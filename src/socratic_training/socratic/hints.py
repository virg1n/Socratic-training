from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from socratic_training.config import AppConfig
from socratic_training.models.loader import load_socratic
from socratic_training.socratic.prompts import socratic_single_hint_prompt


@dataclass
class HintGenResult:
    hints: List[str]
    prompt_ids: List[int]
    completion_ids: List[List[int]]
    raw_text: str
    errors: Tuple[str, ...] = ()


def _summarize_failures(buggy_details: dict, max_items: int = 3) -> str:
    failures = buggy_details.get("failures", [])
    if not isinstance(failures, list) or not failures:
        return "No detailed failures available."
    parts = []
    for f in failures[:max_items]:
        if not isinstance(f, dict):
            continue
        i = f.get("i")
        kind = f.get("kind")
        exp = f.get("expected")
        got = f.get("got")
        parts.append(f"- test[{i}]: {kind} (expected={exp!r}, got={got!r})")
    return "\n".join(parts) if parts else "No detailed failures available."


def generate_hints(
    cfg: AppConfig,
    *,
    topic: str,
    difficulty: str,
    statement: str,
    student_code: str,
    buggy_test_details: dict,
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
            buggy_test_details=buggy_test_details,
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
    buggy_test_details: dict,
    num_hints: Optional[int] = None,
) -> HintGenResult:
    n = int(num_hints or cfg.generation.socratic_num_hints)
    prompt = socratic_single_hint_prompt(
        statement=statement,
        student_code=student_code,
        failure_summary=_summarize_failures(buggy_test_details),
        topic=topic,
        difficulty=difficulty,
    )

    errors: List[str] = []
    model = lm.model
    tok = lm.tokenizer

    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt")
    prompt_ids = inputs["input_ids"][0].tolist()
    inputs = inputs.to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=cfg.generation.socratic_max_new_tokens,
        do_sample=True,
        temperature=0.85,
        top_p=0.95,
        num_return_sequences=n,
        pad_token_id=tok.eos_token_id,
    )

    prompt_len = inputs["input_ids"].shape[1]
    completion_ids: List[List[int]] = []
    hints: List[str] = []
    eos = tok.eos_token_id
    for seq in out:
        comp = seq[prompt_len:].tolist()
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
