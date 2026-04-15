from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from socratic_training.config import AppConfig
from socratic_training.models.loader import load_socratic
from socratic_training.socratic.prompts import socratic_single_hint_prompt
from socratic_training.utils.chat import build_model_inputs, move_to_device


@dataclass
class HintGenResult:
    hints: List[str]
    prompt_ids: List[int]
    completion_ids: List[List[int]]
    raw_text: str
    errors: Tuple[str, ...] = ()


def _summarize_failure_text(observed_failure: str, *, max_lines: int = 40) -> str:
    """
    Keeps the tail of an interpreter traceback / error output.
    """
    text = (observed_failure or "").strip()
    if not text:
        return "None"
    lines = text.splitlines()
    tail = "\n".join(lines[-max_lines:])
    return tail.strip() or "None"


def generate_hints(
    cfg: AppConfig,
    *,
    topic: str,
    difficulty: str,
    statement: str,
    student_code: str,
    observed_failure: str,
    num_hints: Optional[int] = None,
) -> HintGenResult:
    with load_socratic(cfg.models.socratic, for_training=False) as lm:
        return generate_hints_with_lm(
            cfg,
            lm=lm,
            topic=topic,
            difficulty=difficulty,
            statement=statement,
            student_code=student_code,
            observed_failure=observed_failure,
            num_hints=num_hints,
        )


def generate_hints_with_lm(
    cfg: AppConfig,
    *,
    lm,
    topic: str,
    difficulty: str,
    statement: str,
    student_code: str,
    observed_failure: str,
    num_hints: Optional[int] = None,
) -> HintGenResult:
    n = int(num_hints or cfg.generation.socratic_num_hints)
    prompt = socratic_single_hint_prompt(
        statement=statement,
        student_code=student_code,
        failure_summary=_summarize_failure_text(observed_failure),
        topic=topic,
        difficulty=difficulty,
    )

    errors: List[str] = []
    model = lm.model
    tok = lm.tokenizer

    base_inputs = build_model_inputs(tok, user_text=prompt)
    prompt_ids = base_inputs["input_ids"][0].tolist()
    try:
        device = model.get_input_embeddings().weight.device
    except Exception:  # pragma: no cover
        device = next(model.parameters()).device

    prompt_len = base_inputs["input_ids"].shape[1]
    prompt_ids_tensor = base_inputs["input_ids"][0]
    completion_ids: List[List[int]] = []
    hints: List[str] = []
    eos = tok.eos_token_id
    for _ in range(n):
        inputs = move_to_device(base_inputs, device)
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.generation.socratic_max_new_tokens,
            do_sample=True,
            temperature=1.10,
            top_p=0.95,
            num_return_sequences=1,
            pad_token_id=tok.eos_token_id,
        )
        seq = out[0]
        import torch

        prompt_ids_cmp = prompt_ids_tensor.to(seq.device)
        if seq.numel() >= prompt_len and torch.equal(seq[:prompt_len], prompt_ids_cmp):
            comp_ids = seq[prompt_len:]
        else:
            comp_ids = seq
        comp = comp_ids.tolist()
        if eos in comp:
            comp = comp[: comp.index(eos)]
        # Drop trailing pads/eos leftovers (best-effort).
        while comp and comp[-1] in {eos, tok.pad_token_id}:
            comp.pop()
        completion_ids.append(comp)
        hints.append(tok.decode(comp, skip_special_tokens=True).strip())

    # Keep a raw concatenated view for debugging/logging.
    text = "\n\n".join([f"[{i}] {h}" for i, h in enumerate(hints)])

    hints = [h for h in hints if h]
    if len(hints) < n:
        errors.append(f"expected {n} hints, got {len(hints)} (some were empty)")

    return HintGenResult(
        hints=hints[:n],
        prompt_ids=prompt_ids,
        completion_ids=completion_ids[:n],
        raw_text=text,
        errors=tuple(errors),
    )
