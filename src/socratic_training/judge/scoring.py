from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from socratic_training.config import AppConfig
from socratic_training.judge.rubric import RUBRIC_FIELDS, judge_prompt
from socratic_training.models.loader import load_judge
from socratic_training.utils.chat import build_model_inputs, move_to_device
from socratic_training.utils.json import extract_first_json


@dataclass
class HintScore:
    id: int
    subscores: Dict[str, int]
    answer_dump: bool
    final_reward: float
    notes: str = ""


@dataclass
class JudgeResult:
    scores: List[HintScore]
    ranking: List[int]
    raw_text: str
    errors: Tuple[str, ...] = ()


def _is_answer_dump_text(hint: str) -> bool:
    h = hint.lower()
    if "```" in h:
        return True
    if "def solve" in h or "return " in h and "def " in h:
        return True
    if len(h.splitlines()) >= 12:
        return True
    return False


def _reward_from_subscores(sub: Dict[str, int], *, answer_dump: bool, beta_dump: float) -> float:
    # Weighted sum; tuned to strongly favor non-disclosure and actionable debugging.
    w = {
        "did_not_give_final_answer": 3.0,
        "pedagogical_value": 2.0,
        "bug_localization_help": 2.0,
        "curriculum_alignment": 1.5,
        "beginner_friendliness": 1.5,
        "actionability_of_hint": 2.0,
    }
    score = 0.0
    for k, weight in w.items():
        score += weight * float(sub.get(k, 0))

    # Hard penalty if the judge thinks it gave the answer.
    if int(sub.get("did_not_give_final_answer", 0)) <= 1:
        score -= 20.0

    # Extra heuristic penalty for obvious dumps.
    if answer_dump:
        score -= beta_dump

    return score


def score_hints(
    cfg: AppConfig,
    *,
    topic: str,
    difficulty: str,
    statement: str,
    student_code: str,
    hints: List[str],
) -> JudgeResult:
    with load_judge(cfg.models.judge) as lm:
        return score_hints_with_lm(
            cfg,
            lm=lm,
            topic=topic,
            difficulty=difficulty,
            statement=statement,
            student_code=student_code,
            hints=hints,
        )


def score_hints_with_lm(
    cfg: AppConfig,
    *,
    lm,
    topic: str,
    difficulty: str,
    statement: str,
    student_code: str,
    hints: List[str],
) -> JudgeResult:
    prompt = judge_prompt(
        topic=topic,
        difficulty=difficulty,
        statement=statement,
        student_code=student_code,
        hints=hints,
    )

    errors: List[str] = []
    model = lm.model
    tok = lm.tokenizer

    inputs = build_model_inputs(tok, user_text=prompt)
    prompt_len = inputs["input_ids"].shape[1]
    try:
        device = model.get_input_embeddings().weight.device
    except Exception:  # pragma: no cover
        device = next(model.parameters()).device
    inputs = move_to_device(inputs, device)
    out = model.generate(
        **inputs,
        max_new_tokens=cfg.generation.judge_max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    import torch

    seq = out[0]
    prompt_ids = inputs["input_ids"][0]
    prompt_ids_cmp = prompt_ids.to(seq.device)
    if seq.numel() >= prompt_len and torch.equal(seq[:prompt_len], prompt_ids_cmp):
        gen_ids = seq[prompt_len:]
    else:
        gen_ids = seq
    text = tok.decode(gen_ids, skip_special_tokens=True)

    try:
        obj = extract_first_json(text)
        if not isinstance(obj, dict):
            raise ValueError("expected a JSON object")
        items = obj.get("items", [])
        ranking = obj.get("ranking", [])
        if not isinstance(items, list) or not isinstance(ranking, list):
            raise ValueError("items/ranking types invalid")
    except Exception as e:
        return JudgeResult(scores=[], ranking=[], raw_text=text, errors=(f"judge_json_parse_error: {e}",))

    scores: List[HintScore] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        hid = int(it.get("id", -1))
        subs = it.get("subscores", {})
        if not isinstance(subs, dict):
            subs = {}
        subs_int = {k: int(subs.get(k, 0)) for k in RUBRIC_FIELDS}

        hint_text = hints[hid] if 0 <= hid < len(hints) else ""
        dump = bool(it.get("answer_dump", False)) or _is_answer_dump_text(hint_text)
        reward = _reward_from_subscores(subs_int, answer_dump=dump, beta_dump=cfg.training.grpo.beta_answer_dump_penalty)
        scores.append(
            HintScore(
                id=hid,
                subscores=subs_int,
                answer_dump=dump,
                final_reward=reward,
                notes=str(it.get("notes", ""))[:400],
            )
        )

    # If ranking missing, derive from reward.
    if not ranking:
        ranking = [s.id for s in sorted(scores, key=lambda x: x.final_reward, reverse=True)]

    # Basic validation: ensure ranking contains known ids.
    known = {s.id for s in scores}
    ranking = [int(x) for x in ranking if int(x) in known]
    if len(ranking) != len(known):
        errors.append("ranking_incomplete_or_invalid")
        # Fill remainder.
        remaining = [s.id for s in sorted(scores, key=lambda x: x.final_reward, reverse=True) if s.id not in ranking]
        ranking.extend(remaining)

    return JudgeResult(scores=scores, ranking=ranking, raw_text=text, errors=tuple(errors))
