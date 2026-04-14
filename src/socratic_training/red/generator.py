from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from pydantic.v1 import ValidationError
except Exception:  # pragma: no cover
    from pydantic import ValidationError  # type: ignore[no-redef]

from socratic_training.config import AppConfig
from socratic_training.curriculum import Curriculum
from socratic_training.models.loader import load_red
from socratic_training.red.prompts import red_task_generation_prompt
from socratic_training.red.schema import RedTask
from socratic_training.utils.chat import build_model_inputs, move_to_device
from socratic_training.utils.json import extract_first_json


@dataclass
class RedGenResult:
    tasks: List[RedTask]
    raw_texts: List[str]
    errors: Tuple[str, ...] = ()


def _coerce_to_task_obj(obj) -> Optional[dict]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        # Some models still return an array despite instructions.
        return obj[0]
    return None


def generate_red_tasks(
    cfg: AppConfig,
    *,
    curriculum: Curriculum,
    topic: str,
    difficulty: str,
    num_tasks: Optional[int] = None,
    lm: Optional[object] = None,
    rng: Optional[random.Random] = None,
    debug: bool = False,
) -> RedGenResult:
    num = int(num_tasks or cfg.generation.red_num_tasks)
    bucket = curriculum.bucket_prompt(topic=topic, difficulty=difficulty)
    rng = rng or random.Random()

    def _nonempty_line_count(code: str) -> int:
        return sum(1 for line in str(code).splitlines() if line.strip())

    def _sample_target_line_policy() -> Tuple[str, Tuple[int, int], Tuple[int, int], int]:
        r = rng.random()
        # 15%: short (<10 lines), 70%: medium (20-25), 15%: long (>30).
        if r < 0.15:
            # strict (7-9), fallback (5-15)
            return ("short", (7, 9), (5, 15), 420)
        if r < 0.85:
            # strict (20-25), fallback (16-32)
            return ("medium", (20, 25), (16, 32), 700)
        # strict (31-40), fallback (28-55)
        return ("long", (31, 40), (28, 55), int(cfg.generation.red_max_new_tokens))

    def _prompt_for_target(target_min: int, target_max: int) -> str:
        return red_task_generation_prompt(
            curriculum_bucket=bucket,
            min_tests=cfg.validation.min_tests,
            code_lines_hint=f"{int(target_min)}-{int(target_max)}",
        )

    attempt_settings = [
        {"do_sample": True, "temperature": 0.7, "top_p": 0.95},
        {"do_sample": False},
        {"do_sample": True, "temperature": 0.2, "top_p": 0.90},
    ]

    errors: List[str] = []
    texts: List[str] = []
    tasks: List[RedTask] = []

    def _generate_one_with_lm(lm_obj, *, call_index: int) -> Optional[RedTask]:
        model = lm_obj.model
        tok = lm_obj.tokenizer

        policy, strict_rng, fallback_rng, policy_max_new = _sample_target_line_policy()
        target_min, target_max = strict_rng
        fb_min, fb_max = fallback_rng
        prompt = _prompt_for_target(target_min, target_max)
        if debug:
            print(
                f"[debug-red-gen] start call={call_index} policy={policy} strict={target_min}-{target_max} "
                f"fallback={fb_min}-{fb_max} max_new_tokens={min(int(cfg.generation.red_max_new_tokens), int(policy_max_new))}"
            )

        text = ""
        for attempt, gen_kwargs in enumerate(attempt_settings, start=1):
            accept_min, accept_max = (target_min, target_max) if attempt == 1 else (fb_min, fb_max)
            user_prompt = prompt
            if attempt > 1:
                user_prompt = (
                    prompt
                    + "\n\nCRITICAL REMINDER: You MAY include a <think>...</think> block, but then output exactly one JSON object and nothing after it. The JSON must start with '{' and end with '}'."
                    + f"\nTarget code length should be about {target_min}-{target_max} non-empty lines (acceptance for this retry: {accept_min}-{accept_max})."
                )

            inputs = build_model_inputs(tok, user_text=user_prompt)
            prompt_len = inputs["input_ids"].shape[1]
            try:
                device = model.get_input_embeddings().weight.device
            except Exception:  # pragma: no cover
                device = next(model.parameters()).device
            inputs = move_to_device(inputs, device)

            out = model.generate(
                **inputs,
                max_new_tokens=min(int(cfg.generation.red_max_new_tokens), int(policy_max_new)),
                pad_token_id=tok.eos_token_id,
                **gen_kwargs,
            )
            import torch

            seq = out[0]
            prompt_ids = inputs["input_ids"][0]
            prompt_ids_cmp = prompt_ids.to(seq.device)
            if seq.numel() >= prompt_len and torch.equal(seq[:prompt_len], prompt_ids_cmp):
                gen_ids = seq[prompt_len:]
            else:
                # Some generation utilities may return only the generated tokens.
                gen_ids = seq
            text = tok.decode(gen_ids, skip_special_tokens=True)
            texts.append(text)

            try:
                obj = extract_first_json(text)
                task_obj: Optional[dict] = None
                # Common wrappers: {"task":{...}} or {"data":{...}}
                if isinstance(obj, dict):
                    if all(k in obj for k in ("topic", "difficulty", "statement", "code")):
                        task_obj = obj
                    else:
                        for key in ("task", "data", "example", "item"):
                            v = obj.get(key)
                            cand = _coerce_to_task_obj(v)
                            if cand is not None:
                                task_obj = cand
                                break
                        if task_obj is None:
                            # String-wrapped JSON (e.g. {"output": "{...}"}).
                            for key in ("output", "result", "response", "task_json", "json"):
                                v = obj.get(key)
                                if not isinstance(v, str):
                                    continue
                                try:
                                    nested = extract_first_json(v)
                                except Exception:
                                    continue
                                cand = _coerce_to_task_obj(nested)
                                if cand is not None:
                                    task_obj = cand
                                    break
                else:
                    task_obj = _coerce_to_task_obj(obj)

                if task_obj is None:
                    raise ValueError("expected a JSON object matching the schema")

                task = RedTask.parse_obj(task_obj)
                lines = _nonempty_line_count(task.code)
                if lines < accept_min or lines > accept_max:
                    errors.append(
                        f"call{call_index}: attempt{attempt}: code_lines_out_of_range "
                        f"policy={policy} expected={accept_min}-{accept_max} got={lines}"
                    )
                    continue
                if debug:
                    print(
                        f"[debug-red-gen] call={call_index} attempt={attempt} policy={policy} "
                        f"lines={lines} strict={target_min}-{target_max} accepted={accept_min}-{accept_max}"
                    )
                return task
            except Exception as e:
                errors.append(f"call{call_index}: attempt{attempt}: json_parse_error: {e}")

        errors.append(f"call{call_index}: json_parse_error: unknown")
        return None

    def _generate_many(lm_obj) -> None:
        nonlocal tasks
        max_calls = max(num * 4, num + 2)
        call_index = 0
        while len(tasks) < num and call_index < max_calls:
            if debug:
                print(f"[debug-red-gen] progress tasks={len(tasks)}/{num} call={call_index+1}/{max_calls}")
            call_index += 1
            task = _generate_one_with_lm(lm_obj, call_index=call_index)
            if task is None:
                continue
            tasks.append(task)

    if lm is not None:
        _generate_many(lm)
    else:
        with load_red(cfg.models.red, for_training=False) as loaded:
            _generate_many(loaded)

    # Best-effort schema validation error reporting already recorded as json_parse_error.
    # Still catch any stray ValidationError (e.g., if RedTask parse fails unexpectedly).
    valid_tasks: List[RedTask] = []
    for i, t in enumerate(tasks):
        try:
            valid_tasks.append(RedTask.parse_obj(t.dict()))
        except ValidationError as e:
            errors.append(f"task[{i}] schema validation failed: {e}")

    return RedGenResult(tasks=valid_tasks[:num], raw_texts=texts, errors=tuple(errors))
