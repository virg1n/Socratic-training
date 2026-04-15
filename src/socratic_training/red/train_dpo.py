from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from socratic_training.config import AppConfig
from socratic_training.curriculum import load_curriculum
from socratic_training.models.loader import load_red
from socratic_training.red.prompts import red_task_training_prompt
from socratic_training.utils.io import read_yaml


@dataclass
class DpoStats:
    steps: int
    loss: float


def _logprob_avg(model, input_ids, *, prompt_len: int, completion_len: int):
    import torch

    out = model(input_ids=input_ids)
    logits = out.logits
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if completion_len <= 0:
        return torch.zeros((input_ids.shape[0],), device=input_ids.device)
    start = max(0, prompt_len - 1)
    end = start + completion_len
    return token_logp[:, start:end].mean(dim=-1)


def train_red_dpo_from_pairs(
    cfg: AppConfig,
    *,
    max_steps: int = 200,
    beta: float = 0.1,
    lr: float = 1e-4,
    micro_batch_size: int = 1,
    grad_accum_steps: int = 8,
    max_length: int = 1536,
) -> DpoStats:
    pairs_path = Path(cfg.logging.red_dpo_pairs_path)
    if not pairs_path.exists():
        raise FileNotFoundError(f"DPO pairs not found: {pairs_path}")
    curriculum = load_curriculum(Path(cfg.curriculum_path))

    pairs: List[Dict[str, Any]] = []
    for line in pairs_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "prompt" in r and "chosen" in r and "rejected" in r:
            pairs.append(r)
    if not pairs:
        raise RuntimeError("No usable DPO pairs found.")

    import torch
    from torch.utils.data import DataLoader

    try:
        from accelerate import Accelerator

        accelerator = Accelerator(mixed_precision="bf16")
    except Exception:  # pragma: no cover
        accelerator = None

    def collate(batch):
        return batch

    dl = DataLoader(pairs, batch_size=micro_batch_size, shuffle=True, collate_fn=collate)

    def _normalize_prompt(record: Dict[str, Any]) -> str:
        prompt = str(record.get("prompt", "") or "").strip()
        if prompt and not prompt.startswith("TOPIC:"):
            return prompt
        topic = str(record.get("topic", "") or "").strip()
        difficulty = str(record.get("difficulty", "") or "").strip()
        if topic and difficulty:
            return red_task_training_prompt(
                curriculum_bucket=curriculum.bucket_prompt(topic=topic, difficulty=difficulty),
                min_tests=cfg.validation.min_tests,
            )
        return prompt

    with load_red(cfg.models.red, for_training=True) as lm:
        model = lm.model
        tok = lm.tokenizer

        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr)

        if accelerator is not None:
            model, opt, dl = accelerator.prepare(model, opt, dl)
            device = accelerator.device
        else:
            device = next(model.parameters()).device

        # Cache reference logprobs from the initial policy (no extra model).
        model.eval()
        ref_cache: List[Tuple[float, float, int, int, List[int], List[int]]] = []
        with torch.no_grad():
            for r in pairs:
                prompt = _normalize_prompt(r)
                chosen = json.dumps(r["chosen"], ensure_ascii=False) if isinstance(r["chosen"], (dict, list)) else str(r["chosen"])
                rejected = (
                    json.dumps(r["rejected"], ensure_ascii=False) if isinstance(r["rejected"], (dict, list)) else str(r["rejected"])
                )

                p_ids = tok(prompt, add_special_tokens=False).input_ids
                c_ids = tok(chosen, add_special_tokens=False).input_ids
                r_ids = tok(rejected, add_special_tokens=False).input_ids

                # Truncate to max_length.
                p_ids = p_ids[: max_length // 2]
                c_ids = c_ids[: max_length - len(p_ids)]
                r_ids = r_ids[: max_length - len(p_ids)]

                ids_c = p_ids + c_ids
                ids_r = p_ids + r_ids
                lp_c = float(
                    _logprob_avg(
                        model, torch.tensor([ids_c], device=device), prompt_len=len(p_ids), completion_len=len(c_ids)
                    ).item()
                )
                lp_r = float(
                    _logprob_avg(
                        model, torch.tensor([ids_r], device=device), prompt_len=len(p_ids), completion_len=len(r_ids)
                    ).item()
                )
                ref_cache.append((lp_c, lp_r, len(p_ids), len(c_ids), p_ids, c_ids))

        model.train()
        total_loss = 0.0
        step = 0
        opt.zero_grad(set_to_none=True)

        for batch in dl:
            for _ in batch:
                # Sample sequentially from ref_cache; this keeps code simple and avoids large per-step tokenization.
                lp_ref_c, lp_ref_r, prompt_len, comp_len, p_ids, c_ids = ref_cache[step % len(ref_cache)]
                # For rejected, reuse prompt and take rejected ids from the original pairs list.
                pair = pairs[step % len(pairs)]
                rejected = (
                    json.dumps(pair["rejected"], ensure_ascii=False)
                    if isinstance(pair["rejected"], (dict, list))
                    else str(pair["rejected"])
                )
                r_ids = tok(rejected, add_special_tokens=False).input_ids[: max_length - len(p_ids)]

                ids_c = p_ids + c_ids
                ids_r = p_ids + r_ids
                input_c = torch.tensor([ids_c], device=device, dtype=torch.long)
                input_r = torch.tensor([ids_r], device=device, dtype=torch.long)

                lp_pi_c = _logprob_avg(model, input_c, prompt_len=prompt_len, completion_len=len(c_ids))
                lp_pi_r = _logprob_avg(model, input_r, prompt_len=prompt_len, completion_len=len(r_ids))

                # DPO with cached reference (from initial policy).
                # delta = (log pi - log ref)_chosen - (log pi - log ref)_rejected
                delta = (lp_pi_c - lp_ref_c) - (lp_pi_r - lp_ref_r)
                loss = -torch.nn.functional.logsigmoid(beta * delta).mean()

                if accelerator is not None:
                    accelerator.backward(loss)
                else:
                    loss.backward()
                total_loss += float(loss.detach().item())

                if (step + 1) % grad_accum_steps == 0:
                    if accelerator is not None:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                step += 1
                if step >= max_steps:
                    break
            if step >= max_steps:
                break

        if getattr(cfg.models.red, "save_adapters", True):
            out_dir = Path(cfg.models.red.adapter_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(model, "save_pretrained"):
                try:
                    if accelerator is not None:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            accelerator.unwrap_model(model).save_pretrained(str(out_dir))
                    else:
                        model.save_pretrained(str(out_dir))
                except Exception:
                    pass

    return DpoStats(steps=step, loss=total_loss / max(1, step))


def run_red_dpo(config_path: Path) -> None:
    cfg = AppConfig.parse_obj(read_yaml(config_path))
    stats = train_red_dpo_from_pairs(cfg)
    print(f"Red DPO done: steps={stats.steps} loss={stats.loss:.4f}")
