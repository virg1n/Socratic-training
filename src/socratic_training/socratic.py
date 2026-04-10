from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .model_manager import StagedModelManager, generate_texts
from .prompts import socratic_hint_prompt
from .types import HintCandidate, JudgeEvaluation, RolloutRecord, TaskExample
from .utils import batched, safe_mean, safe_std


class SocraticRolloutGenerator:
    def __init__(self, config: PipelineConfig, models: StagedModelManager) -> None:
        self.config = config
        self.models = models

    def generate_hints(self, tasks: list[TaskExample]) -> list[list[HintCandidate]]:
        prompts = [socratic_hint_prompt(task) for task in tasks]
        grouped_candidates: list[list[HintCandidate]] = []
        with self.models.use_socratic(mode="inference") as runtime:
            raw_outputs = generate_texts(
                runtime,
                prompts,
                max_new_tokens=self.config.socratic.max_hint_tokens,
                temperature=self.config.socratic.rollout_temperature,
                top_p=self.config.socratic.top_p,
                num_return_sequences=self.config.socratic.group_size,
            )
        for task_index, task in enumerate(tasks):
            start = task_index * self.config.socratic.group_size
            group: list[HintCandidate] = []
            for local_index, text in enumerate(raw_outputs[start : start + self.config.socratic.group_size], start=1):
                group.append(HintCandidate(task=task, prompt=prompts[task_index], hint=text.strip(), index=local_index))
            grouped_candidates.append(group)
        return grouped_candidates


class StagedGRPOTrainer:
    def __init__(self, config: PipelineConfig, models: StagedModelManager) -> None:
        self.config = config
        self.models = models
        self.optimizer_state_path = Path(self.config.paths.checkpoints_dir) / "socratic" / "optimizer.pt"

    def optimize(self, evaluation_groups: list[list[JudgeEvaluation]]) -> dict[str, float]:
        torch = _require_module("torch")
        rollout_records: list[RolloutRecord] = []
        for group in evaluation_groups:
            rewards = [evaluation.final_reward for evaluation in group]
            mean_reward = safe_mean(rewards)
            std_reward = safe_std(rewards)
            for evaluation in group:
                rollout_records.append(
                    RolloutRecord(
                        candidate=evaluation.candidate,
                        reward=evaluation.final_reward,
                        group_advantage=(evaluation.final_reward - mean_reward) / std_reward,
                    )
                )

        losses: list[float] = []
        with self.models.use_socratic(mode="train") as runtime:
            tokenizer = runtime.tokenizer
            model = runtime.model
            optimizer = self._build_optimizer(model)
            if self.optimizer_state_path.exists():
                optimizer.load_state_dict(torch.load(self.optimizer_state_path, map_location="cpu"))
            model.train()
            for _ in range(self.config.socratic.epochs_per_iteration):
                optimizer.zero_grad(set_to_none=True)
                for batch in batched(rollout_records, self.config.socratic.micro_batch_size):
                    old_log_probs = self._sequence_log_probs(model, tokenizer, batch).detach()
                    loss = self._grpo_loss(model, tokenizer, batch, old_log_probs)
                    losses.append(float(loss.detach().cpu().item()))
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.socratic.max_grad_norm)
                optimizer.step()
            checkpoint_dir = Path(self.config.paths.checkpoints_dir) / "socratic" / "latest"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            torch.save(optimizer.state_dict(), self.optimizer_state_path)
        return {
            "mean_reward": safe_mean([record.reward for record in rollout_records]),
            "mean_loss": safe_mean(losses),
            "num_rollouts": float(len(rollout_records)),
        }

    def _sequence_log_probs(self, model, tokenizer, batch: list[RolloutRecord]):
        torch = _require_module("torch")
        encodings = _encode_batch(tokenizer, batch, device=self.config.socratic.train_device)
        outputs = model(input_ids=encodings["input_ids"], attention_mask=encodings["attention_mask"])
        log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        labels = encodings["input_ids"][:, 1:]
        gathered = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        mask = encodings["completion_mask"][:, 1:]
        return (gathered * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)

    def _grpo_loss(self, model, tokenizer, batch: list[RolloutRecord], old_log_probs):
        torch = _require_module("torch")
        new_log_probs = self._sequence_log_probs(model, tokenizer, batch)
        advantages = torch.tensor([record.group_advantage for record in batch], dtype=new_log_probs.dtype, device=new_log_probs.device)
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1.0 - self.config.socratic.clip_range, 1.0 + self.config.socratic.clip_range)
        surrogate = torch.minimum(ratio * advantages, clipped * advantages)
        approx_kl = 0.5 * (new_log_probs - old_log_probs).pow(2)
        return -(surrogate - self.config.socratic.kl_beta * approx_kl).mean()

    def _build_optimizer(self, model):
        if self.config.socratic.paged_optimizer:
            try:
                bnb = _require_module("bitsandbytes")
                return bnb.optim.PagedAdamW32bit(model.parameters(), lr=self.config.socratic.learning_rate, weight_decay=self.config.socratic.weight_decay)
            except Exception:
                pass
        torch = _require_module("torch")
        return torch.optim.AdamW(model.parameters(), lr=self.config.socratic.learning_rate, weight_decay=self.config.socratic.weight_decay)


def _encode_batch(tokenizer, batch: list[RolloutRecord], device: str):
    torch = _require_module("torch")
    prompt_ids = [tokenizer.encode(record.candidate.prompt, add_special_tokens=False) for record in batch]
    completion_ids = [tokenizer.encode(record.candidate.hint + tokenizer.eos_token, add_special_tokens=False) for record in batch]
    full_sequences = [prompt + completion for prompt, completion in zip(prompt_ids, completion_ids, strict=False)]
    max_length = max(len(sequence) for sequence in full_sequences)
    input_ids = []
    attention_mask = []
    completion_mask = []
    pad_id = tokenizer.pad_token_id
    for prompt, completion, sequence in zip(prompt_ids, completion_ids, full_sequences, strict=False):
        pad_len = max_length - len(sequence)
        input_ids.append([pad_id] * pad_len + sequence)
        attention_mask.append([0] * pad_len + [1] * len(sequence))
        completion_mask.append([0] * (pad_len + len(prompt)) + [1] * len(completion))
    return {
        "input_ids": torch.tensor(input_ids, device=device),
        "attention_mask": torch.tensor(attention_mask, device=device),
        "completion_mask": torch.tensor(completion_mask, dtype=torch.float32, device=device),
    }


def _require_module(name: str):
    import importlib

    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise RuntimeError(f"{name} is required for Socratic GRPO training.") from error
