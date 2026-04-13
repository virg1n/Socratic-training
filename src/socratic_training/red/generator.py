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

    errors: List[str] = []
    with load_red(cfg.models.red, for_training=False) as lm:
        model = lm.model
        tok = lm.tokenizer

        inputs = tok(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]
        try:
            device = model.get_input_embeddings().weight.device
        except Exception:  # pragma: no cover
            device = next(model.parameters()).device
        inputs = inputs.to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.generation.red_max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tok.eos_token_id,
        )
        gen = out[0][prompt_len:]
        text = tok.decode(gen, skip_special_tokens=True)

    try:
        obj = extract_first_json(text)
        if not isinstance(obj, list):
            raise ValueError("expected a JSON array")
        arr = obj
    except Exception as e:
        return RedGenResult(tasks=[], raw_text=text, errors=(f"json_parse_error: {e}",))

    tasks: List[RedTask] = []
    for i, obj in enumerate(arr):
        try:
            tasks.append(RedTask.parse_obj(obj))
        except ValidationError as e:
            errors.append(f"task[{i}] schema validation failed: {e}")

    return RedGenResult(tasks=tasks, raw_text=text, errors=tuple(errors))
