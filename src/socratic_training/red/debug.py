from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Tuple

from socratic_training.config import AppConfig
from socratic_training.curriculum import load_curriculum
from socratic_training.red.generator import generate_red_tasks
from socratic_training.utils.events import append_event
from socratic_training.utils.io import read_yaml
from socratic_training.validation.task_validator import validate_red_task


def run_red_debug(config_path: Path, *, topic: str, difficulty: str) -> None:
    cfg = AppConfig.parse_obj(read_yaml(config_path))
    curriculum = load_curriculum(Path(cfg.curriculum_path))

    out_dir = Path(cfg.logging.out_dir) / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    red = generate_red_tasks(cfg, curriculum=curriculum, topic=topic, difficulty=difficulty)
    raw_debug = "\n\n".join([f"### CALL {i+1}\n{t}" for i, t in enumerate(red.raw_texts[-10:])]) if red.raw_texts else ""
    (out_dir / "red_last_completion.txt").write_text(raw_debug, encoding="utf-8")
    print(f"Red parse errors: {list(red.errors)}")
    print(f"Red tasks parsed: {len(red.tasks)}")

    reasons = Counter()
    ok = 0
    for t in red.tasks:
        v = validate_red_task(cfg, curriculum=curriculum, task=t, seen_fingerprints=set())
        if v.ok:
            ok += 1
        for r in v.reasons:
            reasons[r] += 1

    print(f"Valid tasks: {ok}/{len(red.tasks)}")
    if reasons:
        print("Top reject reasons:")
        for r, c in reasons.most_common(15):
            print(f"- {r}: {c}")
