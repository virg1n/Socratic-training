from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from socratic_training.config import AppConfig
from socratic_training.judge.rubric import (
    judge_prompt_directness_relevance,
    judge_prompt_hint_validation,
    judge_prompt_pedagogy_localization,
)
from socratic_training.models.loader import load_judge
from socratic_training.utils.chat import build_model_inputs, move_to_device
from socratic_training.utils.json import extract_first_json


@dataclass
class HintScore:
    id: int
    subscores: Dict[str, int]
    answer_dump: bool
    final_reward: float
    valid: bool = True
    validation_issues: Tuple[str, ...] = ()
    notes: str = ""


@dataclass
class JudgeResult:
    scores: List[HintScore]
    ranking: List[int]
    raw_text: str
    errors: Tuple[str, ...] = ()


def _clamp_int(v: object, *, lo: int, hi: int) -> int:
    try:
        iv = int(v)  # type: ignore[arg-type]
    except Exception:
        return lo
    return max(lo, min(hi, iv))


def _coerce_bool(v: object, *, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
    return default


def _coerce_issue_list(v: object) -> Tuple[str, ...]:
    if not isinstance(v, list):
        return ()
    out: List[str] = []
    for item in v:
        s = str(item).strip().lower().replace(" ", "_")
        if s:
            out.append(s)
    return tuple(dict.fromkeys(out))


def _has_think_tags(text: str) -> bool:
    t = (text or "").lower()
    return "<think" in t or "</think>" in t


def _heuristic_answer_dump(text: str) -> bool:
    """
    Conservative "answer dump" heuristic used only as a fallback.
    Prefer the Judge-provided `answer_dump` field when available.
    """
    s = (text or "").strip()
    if not s:
        return False
    nonempty = sum(1 for ln in s.splitlines() if ln.strip())
    if nonempty >= 14:
        return True
    if "```" in s and nonempty >= 6:
        return True
    return False


def _reward_from_subscores(
    cfg: AppConfig,
    sub: Dict[str, int],
    *,
    answer_dump: bool,
    valid: bool,
    validation_issues: Tuple[str, ...],
) -> float:
    """
    Weighted reward with config-driven penalties for answer dumps and validation failures.

    Subscores:
    - not_too_direct: 0..10 (higher is better)
    - relevant: 0..10
    - pedagogical_value: 0..5
    - bug_localization_help: 0..5
    """
    w = {
        "not_too_direct": 3.0,
        "relevant": 2.0,
        "pedagogical_value": 2.0,
        "bug_localization_help": 2.0,
    }
    base = 0.0
    for k, weight in w.items():
        base += weight * float(sub.get(k, 0))

    beta = max(0.0, float(cfg.training.grpo.beta_answer_dump_penalty))
    normalized_issues = set(validation_issues)
    if answer_dump:
        normalized_issues.add("answer_dump")

    issue_penalties = {
        "answer_dump": 25.0,
        "too_direct": 8.0,
        "generic": 4.0,
        "not_actionable": 4.0,
        "chain_of_thought": 10.0,
        "malformed": 10.0,
    }

    penalties = 0.0
    if not valid:
        penalties += beta * 5.0
    penalties += beta * sum(issue_penalties.get(issue, 0.0) for issue in normalized_issues)
    return base - penalties


def score_hints(
    cfg: AppConfig,
    *,
    topic: str,
    difficulty: str,
    statement: str,
    student_code: str,
    observed_failure: str,
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
            observed_failure=observed_failure,
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
    observed_failure: str,
    hints: List[str],
) -> JudgeResult:
    errors: List[str] = []
    model = lm.model
    tok = lm.tokenizer

    def _generate_json(stage: str, prompt: str) -> Tuple[Optional[dict], str]:
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
            return (obj if isinstance(obj, dict) else None), text
        except Exception as e:
            errors.append(f"judge_json_parse_error[{stage}]: {e}")
            return None, text

    prompt_v = judge_prompt_hint_validation(
        topic=topic,
        difficulty=difficulty,
        statement=statement,
        student_code=student_code,
        observed_failure=observed_failure,
        hints=hints,
    )
    obj_v, text_v = _generate_json("validation", prompt_v)

    prompt_a = judge_prompt_directness_relevance(
        topic=topic,
        difficulty=difficulty,
        statement=statement,
        student_code=student_code,
        observed_failure=observed_failure,
        hints=hints,
    )
    obj_a, text_a = _generate_json("directness_relevance", prompt_a)

    prompt_b = judge_prompt_pedagogy_localization(
        topic=topic,
        difficulty=difficulty,
        statement=statement,
        student_code=student_code,
        observed_failure=observed_failure,
        hints=hints,
    )
    obj_b, text_b = _generate_json("pedagogy_localization", prompt_b)

    n = len(hints)
    by_id: Dict[int, Dict[str, object]] = {
        i: {
            "not_too_direct": 0,
            "relevant": 0,
            "pedagogical_value": 0,
            "bug_localization_help": 0,
            "answer_dump": None,
            "valid": True,
            "issues": (),
            "notes": "",
        }
        for i in range(n)
    }

    def _ingest(stage: str, obj: Optional[dict], *, fields: Tuple[str, ...]) -> None:
        if not obj:
            return
        items = obj.get("items", [])
        if not isinstance(items, list):
            errors.append(f"judge_items_invalid[{stage}]")
            return
        for it in items:
            if not isinstance(it, dict):
                continue
            try:
                hid = int(it.get("id", -1))
            except Exception:
                continue
            if hid not in by_id:
                continue
            rec = by_id[hid]
            for f in fields:
                if f not in it:
                    continue
                rec[f] = it.get(f)

    _ingest("directness_relevance", obj_a, fields=("not_too_direct", "relevant", "answer_dump"))
    _ingest("pedagogy_localization", obj_b, fields=("pedagogical_value", "bug_localization_help"))

    if isinstance(obj_v, dict):
        items = obj_v.get("items", [])
        if not isinstance(items, list):
            errors.append("judge_items_invalid[validation]")
        else:
            for it in items:
                if not isinstance(it, dict):
                    continue
                try:
                    hid = int(it.get("id", -1))
                except Exception:
                    continue
                if hid not in by_id:
                    continue
                rec = by_id[hid]
                rec["valid"] = it.get("valid", True)
                rec["issues"] = _coerce_issue_list(it.get("issues", []))
                rec["notes"] = str(it.get("notes", "") or "").strip()

    scores: List[HintScore] = []
    for hid in range(n):
        rec = by_id[hid]
        subs = {
            "not_too_direct": _clamp_int(rec.get("not_too_direct"), lo=0, hi=10),
            "relevant": _clamp_int(rec.get("relevant"), lo=0, hi=10),
            "pedagogical_value": _clamp_int(rec.get("pedagogical_value"), lo=0, hi=5),
            "bug_localization_help": _clamp_int(rec.get("bug_localization_help"), lo=0, hi=5),
        }
        has_think = _has_think_tags(hints[hid])
        validation_issues = _coerce_issue_list(rec.get("issues", ()))
        valid = _coerce_bool(rec.get("valid"), default=True)
        dump_flag = rec.get("answer_dump", None)
        if dump_flag is None:
            dump = _heuristic_answer_dump(hints[hid]) or ("answer_dump" in validation_issues)
        else:
            dump = _coerce_bool(dump_flag, default=False) or ("answer_dump" in validation_issues)
        reward = _reward_from_subscores(
            cfg,
            subs,
            answer_dump=dump,
            valid=valid,
            validation_issues=validation_issues,
        )
        if has_think:
            reward *= 0.2
        scores.append(
            HintScore(
                id=hid,
                subscores=subs,
                answer_dump=dump,
                final_reward=float(reward),
                valid=valid,
                validation_issues=validation_issues,
                notes=str(rec.get("notes", "") or ""),
            )
        )

    ranking = [s.id for s in sorted(scores, key=lambda x: x.final_reward, reverse=True)]
    raw_text = (
        f"=== validation ===\n{text_v}\n\n"
        f"=== directness_relevance ===\n{text_a}\n\n"
        f"=== pedagogy_localization ===\n{text_b}"
    )
    return JudgeResult(scores=scores, ranking=ranking, raw_text=raw_text, errors=tuple(errors))
