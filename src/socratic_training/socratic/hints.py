from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from socratic_training.config import AppConfig
from socratic_training.models.loader import load_socratic
from socratic_training.socratic.prompts import socratic_single_hint_prompt
from socratic_training.utils.chat import build_model_inputs, move_to_device


@dataclass
class HintGenResult:
    hints: List[str]
    prompt_ids: List[List[int]]
    completion_ids: List[List[int]]
    raw_text: str
    errors: Tuple[str, ...] = ()


_HINT_FOCUS_ROTATION = (
    "Focus on the first failing test and ask what value the code produces versus what the assert expects.",
    "Focus on tracing loop bounds, iteration counts, or stopping conditions by hand on a tiny example.",
    "Focus on how a key state variable changes over time, such as an accumulator, index, counter, or temporary value.",
    "Focus on whether the code takes the correct branch or condition for the failing case.",
    "Focus on whether helper functions receive and return the values the outer function expects.",
    "Focus on checking the smallest edge case that should pass and comparing it with the current behavior.",
)


def _normalize_hint_text(text: str) -> str:
    lowered = (text or "").strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^\w\s]", "", lowered)
    return lowered.strip()


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
    errors: List[str] = []
    model = lm.model
    tok = lm.tokenizer
    try:
        device = model.get_input_embeddings().weight.device
    except Exception:  # pragma: no cover
        device = next(model.parameters()).device

    failure_summary = _summarize_failure_text(observed_failure)
    max_attempts = max(n * 3, n + 4)
    seen_hints = set()
    prompt_ids: List[List[int]] = []
    completion_ids: List[List[int]] = []
    hints: List[str] = []
    raw_items: List[str] = []
    eos = tok.eos_token_id
    attempts = 0

    while len(hints) < n and attempts < max_attempts:
        focus_instruction = _HINT_FOCUS_ROTATION[attempts % len(_HINT_FOCUS_ROTATION)]
        prompt = socratic_single_hint_prompt(
            statement=statement,
            student_code=student_code,
            failure_summary=failure_summary,
            topic=topic,
            difficulty=difficulty,
            focus_instruction=focus_instruction,
            previous_hints=tuple(hints[-3:]),
        )

        base_inputs = build_model_inputs(tok, user_text=prompt)
        prompt_len = base_inputs["input_ids"].shape[1]
        prompt_ids_tensor = base_inputs["input_ids"][0]
        inputs = move_to_device(base_inputs, device)
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.generation.socratic_max_new_tokens,
            do_sample=True,
            temperature=0.95,
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
        while comp and comp[-1] in {eos, tok.pad_token_id}:
            comp.pop()

        hint_text = tok.decode(comp, skip_special_tokens=True).strip()
        raw_items.append(f"[attempt {attempts}] {hint_text}")
        attempts += 1
        if not hint_text:
            errors.append("generated empty hint")
            continue

        normalized = _normalize_hint_text(hint_text)
        if normalized in seen_hints:
            errors.append("generated duplicate hint")
            continue

        seen_hints.add(normalized)
        prompt_ids.append(prompt_ids_tensor.tolist())
        completion_ids.append(comp)
        hints.append(hint_text)

    text = "\n\n".join(raw_items)
    if len(hints) < n:
        errors.append(f"expected {n} distinct hints, got {len(hints)}")

    return HintGenResult(
        hints=hints[:n],
        prompt_ids=prompt_ids[:n],
        completion_ids=completion_ids[:n],
        raw_text=text,
        errors=tuple(errors),
    )
