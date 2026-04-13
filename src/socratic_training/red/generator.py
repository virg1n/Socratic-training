from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

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
    raw_text: str
    errors: Tuple[str, ...] = ()

def generate_red_tasks(
    cfg: AppConfig,
    *,
    curriculum: Curriculum,
    topic: str,
    difficulty: str,
    num_tasks: Optional[int] = None,
) -> RedGenResult:
    num = int(num_tasks or cfg.generation.red_num_tasks)
    bucket = curriculum.bucket_prompt(topic=topic, difficulty=difficulty)
    prompt = red_task_generation_prompt(
        curriculum_bucket=bucket,
        num_tasks=num,
        max_tests=cfg.validation.max_tests,
        min_tests=cfg.validation.min_tests,
    )

    attempt_settings = [
        {"do_sample": True, "temperature": 0.7, "top_p": 0.95},
        {"do_sample": False},
        {"do_sample": True, "temperature": 0.2, "top_p": 0.90},
    ]

    errors: List[str] = []
    text = ""
    arr: Optional[list] = None

    with load_red(cfg.models.red, for_training=False) as lm:
        model = lm.model
        tok = lm.tokenizer

        for attempt, gen_kwargs in enumerate(attempt_settings, start=1):
            user_prompt = prompt
            if attempt > 1:
                user_prompt = (
                    prompt
                    + "\n\nCRITICAL REMINDER: Output MUST be STRICT JSON only, starting with '[' and ending with ']'."
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
                max_new_tokens=cfg.generation.red_max_new_tokens,
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

            try:
                obj = extract_first_json(text)
                arr_candidate: Optional[list] = None
                if isinstance(obj, list):
                    arr_candidate = obj
                elif isinstance(obj, dict):
                    # Common wrappers: {"tasks":[...]} or {"items":[...]}
                    for key in ("tasks", "items", "data", "examples"):
                        v = obj.get(key)
                        if isinstance(v, list):
                            arr_candidate = v
                            break
                    # String-wrapped JSON (e.g. {"output": "[{...}]"}).
                    if arr_candidate is None:
                        for key in ("output", "result", "response", "tasks_json", "json"):
                            v = obj.get(key)
                            if not isinstance(v, str):
                                continue
                            try:
                                nested = extract_first_json(v)
                            except Exception:
                                continue
                            if isinstance(nested, list):
                                arr_candidate = nested
                                break
                            if isinstance(nested, dict):
                                for k2 in ("tasks", "items", "data", "examples"):
                                    vv = nested.get(k2)
                                    if isinstance(vv, list):
                                        arr_candidate = vv
                                        break
                                if arr_candidate is not None:
                                    break
                    # Single-task object fallback.
                    if arr_candidate is None and any(k in obj for k in ("statement", "canonical_solution", "buggy_solution", "tests")):
                        arr_candidate = [obj]
                if arr_candidate is None:
                    raise ValueError("expected a JSON array (or object containing a tasks/items list)")
                arr = arr_candidate
                break
            except Exception as e:
                errors.append(f"attempt{attempt}: json_parse_error: {e}")

    if arr is None:
        return RedGenResult(tasks=[], raw_text=text, errors=tuple(errors or ["json_parse_error: unknown"]))

    tasks: List[RedTask] = []
    for i, obj in enumerate(arr):
        try:
            tasks.append(RedTask.parse_obj(obj))
        except ValidationError as e:
            errors.append(f"task[{i}] schema validation failed: {e}")

    return RedGenResult(tasks=tasks, raw_text=text, errors=tuple(errors))
