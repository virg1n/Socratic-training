from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from socratic_training.config import AppConfig
from socratic_training.models.loader import load_socratic


@dataclass(frozen=True)
class GrpoTrajectory:
    prompt_ids: List[int]
    completion_ids: List[int]
    reward: float
    group_id: int


@dataclass
class GrpoTrainStats:
    steps: int
    mean_reward: float
    mean_advantage: float
    loss: float


def _logprob_avg(model, input_ids, *, prompt_len: int, completion_len: int):
    import torch

    out = model(input_ids=input_ids)
    logits = out.logits  # [B, T, V]
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    if completion_len <= 0:
        return torch.zeros((input_ids.shape[0],), device=input_ids.device)

    start = max(0, prompt_len - 1)
    end = start + completion_len
    comp_lp = token_logp[:, start:end]
    return comp_lp.mean(dim=-1)  # average per completion token


def _compute_advantages(trajs: List[GrpoTrajectory]) -> List[float]:
    by_group: Dict[int, List[float]] = {}
    for t in trajs:
        by_group.setdefault(t.group_id, []).append(float(t.reward))
    means = {gid: sum(rs) / max(1, len(rs)) for gid, rs in by_group.items()}
    adv = [float(t.reward) - means[t.group_id] for t in trajs]
    # Normalize advantages for stability.
    if len(adv) >= 2:
        m = sum(adv) / len(adv)
        v = sum((x - m) ** 2 for x in adv) / (len(adv) - 1)
        s = math.sqrt(max(v, 1e-8))
        adv = [(x - m) / s for x in adv]
    return adv


def train_socratic_grpo(
    cfg: AppConfig,
    *,
    trajectories: List[GrpoTrajectory],
    output_adapter_dir: Optional[Path] = None,
) -> GrpoTrainStats:
    """
    Minimal GRPO/PPO-style update:
    - Group-relative advantages (reward - group mean), then normalized.
    - PPO clipping based on old policy logprobs computed before updates.

    This is intentionally simple and designed for single-node workstation training.
    For multi-GPU, run via `accelerate launch` (Accelerator will pick it up automatically if available).
    """
    if not trajectories:
        return GrpoTrainStats(steps=0, mean_reward=0.0, mean_advantage=0.0, loss=0.0)

    try:
        from accelerate import Accelerator
    except Exception:  # pragma: no cover
        Accelerator = None  # type: ignore[assignment]

    import torch
    from torch.utils.data import DataLoader

    grpo = cfg.training.grpo
    output_dir = Path(output_adapter_dir or cfg.models.socratic.adapter_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    advantages = _compute_advantages(trajectories)

    # Pack into tensors lazily inside collate.
    def collate(batch_idx: List[int]):
        # micro-batch is indices
        ids = batch_idx
        return ids

    dl = DataLoader(
        list(range(len(trajectories))),
        batch_size=grpo.micro_batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    if Accelerator is not None:
        accelerator = Accelerator(mixed_precision=grpo.mixed_precision if grpo.mixed_precision != "no" else None)
    else:
        accelerator = None

    with load_socratic(cfg.models.socratic, for_training=True) as lm:
        model = lm.model
        tok = lm.tokenizer

        if grpo.use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        # Optimizer: only trainable params (LoRA if enabled).
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=grpo.lr,
            weight_decay=grpo.weight_decay,
        )

        if accelerator is not None:
            model, opt, dl = accelerator.prepare(model, opt, dl)
            device = accelerator.device
        else:
            device = next(model.parameters()).device

        # Compute old policy logprobs once (no grad).
        old_lp: List[float] = []
        model.eval()
        with torch.no_grad():
            for t in trajectories:
                ids = t.prompt_ids + t.completion_ids
                input_ids = torch.tensor([ids], device=device, dtype=torch.long)
                lp = _logprob_avg(
                    model,
                    input_ids,
                    prompt_len=len(t.prompt_ids),
                    completion_len=len(t.completion_ids),
                )
                old_lp.append(float(lp.item()))
        model.train()

        total_loss = 0.0
        steps = 0
        accum = 0
        opt.zero_grad(set_to_none=True)

        for _epoch in range(grpo.epochs_per_iter):
            for batch_ids in dl:
                # batch_ids is list[int]
                loss_batch = 0.0
                for idx in batch_ids:
                    t = trajectories[int(idx)]
                    ids = t.prompt_ids + t.completion_ids
                    input_ids = torch.tensor([ids], device=device, dtype=torch.long)

                    lp_new = _logprob_avg(
                        model,
                        input_ids,
                        prompt_len=len(t.prompt_ids),
                        completion_len=len(t.completion_ids),
                    )
                    lp_old = torch.tensor([old_lp[int(idx)]], device=device, dtype=lp_new.dtype)

                    # PPO ratio on avg logprob (more stable than sum for variable lengths).
                    ratio = torch.exp(lp_new - lp_old).clamp(0.0, 10.0)
                    adv = torch.tensor([advantages[int(idx)]], device=device, dtype=lp_new.dtype)

                    unclipped = ratio * adv
                    clipped = torch.clamp(ratio, 1.0 - grpo.ppo_clip, 1.0 + grpo.ppo_clip) * adv
                    loss = -torch.min(unclipped, clipped).mean()
                    loss_batch = loss_batch + loss

                loss_batch = loss_batch / max(1, len(batch_ids))
                if accelerator is not None:
                    accelerator.backward(loss_batch)
                else:
                    loss_batch.backward()

                accum += 1
                total_loss += float(loss_batch.detach().item())
                if accum % grpo.grad_accum_steps == 0:
                    if accelerator is not None:
                        accelerator.clip_grad_norm_(model.parameters(), grpo.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grpo.max_grad_norm)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    steps += 1

        # Save adapters if using PEFT.
        if hasattr(model, "save_pretrained"):
            try:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_pretrained(str(output_dir))
                else:
                    model.save_pretrained(str(output_dir))
            except Exception:
                # If not a PEFT model, skip.
                pass

    mean_r = sum(t.reward for t in trajectories) / len(trajectories)
    mean_a = sum(advantages) / len(advantages)
    loss = total_loss / max(1, steps * grpo.grad_accum_steps)
    return GrpoTrainStats(steps=steps, mean_reward=float(mean_r), mean_advantage=float(mean_a), loss=float(loss))

