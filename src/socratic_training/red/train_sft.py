from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from socratic_training.config import AppConfig
from socratic_training.curriculum import load_curriculum
from socratic_training.models.loader import load_red
from socratic_training.utils.io import read_yaml


@dataclass
class SftStats:
    steps: int
    loss: float


def _format_sft_text(prompt: str, response_json: str) -> str:
    return f"PROMPT:\n{prompt}\n\nRESPONSE_JSON:\n{response_json}\n"


def train_red_sft_from_hard_buffer(
    cfg: AppConfig,
    *,
    max_steps: int = 100,
    lr: float = 2e-4,
    micro_batch_size: int = 1,
    grad_accum_steps: int = 8,
    max_length: int = 1536,
) -> SftStats:
    """
    Trains Red with teacher forcing on the hard-example buffer.
    This updates only LoRA adapters (per loader config).
    """
    curriculum = load_curriculum(Path(cfg.curriculum_path))
    buf_path = Path(cfg.logging.hard_buffer_path)
    if not buf_path.exists():
        raise FileNotFoundError(f"Hard buffer not found: {buf_path}")

    records: List[Dict[str, Any]] = []
    for line in buf_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    if not records:
        raise RuntimeError("Hard buffer is empty; nothing to train Red on.")

    # Build prompt/response pairs.
    examples: List[Dict[str, str]] = []
    for r in records:
        topic = str(r.get("topic", ""))
        difficulty = str(r.get("difficulty", ""))
        task = r.get("task", {})
        if not isinstance(task, dict):
            continue
        prompt = curriculum.bucket_prompt(topic=topic, difficulty=difficulty)
        response = json.dumps(task, ensure_ascii=False)
        examples.append({"prompt": prompt, "response": response})

    if not examples:
        raise RuntimeError("No usable examples in hard buffer.")

    import torch
    from torch.utils.data import DataLoader

    try:
        from accelerate import Accelerator

        accelerator = Accelerator(mixed_precision="bf16")
    except Exception:  # pragma: no cover
        accelerator = None

    def collate(batch: List[Dict[str, str]]):
        return batch

    dl = DataLoader(examples, batch_size=micro_batch_size, shuffle=True, collate_fn=collate)

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

        model.train()
        total_loss = 0.0
        step = 0
        opt.zero_grad(set_to_none=True)

        for batch in dl:
            for ex in batch:
                text = _format_sft_text(ex["prompt"], ex["response"])
                # Tokenize prompt separately to mask labels.
                prompt_ids = tok(f"PROMPT:\n{ex['prompt']}\n\nRESPONSE_JSON:\n", add_special_tokens=False).input_ids
                full = tok(text, add_special_tokens=False, truncation=True, max_length=max_length, return_tensors="pt")
                input_ids = full["input_ids"].to(device)
                attn = full["attention_mask"].to(device)

                labels = input_ids.clone()
                # Mask prompt tokens.
                mask_len = min(len(prompt_ids), labels.shape[1])
                labels[:, :mask_len] = -100

                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss = out.loss
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

        # Save adapters (optional)
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

    return SftStats(steps=step, loss=total_loss / max(1, step))


def run_red_sft(config_path: Path) -> None:
    cfg = AppConfig.parse_obj(read_yaml(config_path))
    stats = train_red_sft_from_hard_buffer(cfg)
    print(f"Red SFT done: steps={stats.steps} loss={stats.loss:.4f}")
