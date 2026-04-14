from __future__ import annotations

import random
import warnings
from contextlib import ExitStack
from pathlib import Path
from typing import Optional

from socratic_training.config import AppConfig
from socratic_training.curriculum import load_curriculum
from socratic_training.memory import preflight_and_autoscale
from socratic_training.models.loader import load_judge, load_red, load_socratic
from socratic_training.pipeline.bucket_select import choose_bucket
from socratic_training.pipeline.iteration import _ensure_dirs, run_iteration_cfg
from socratic_training.utils.io import read_yaml


def run_loop(
    config_path: Path,
    *,
    topic: str,
    difficulty: str,
    iterations: int,
    seed: Optional[int] = None,
    debug_red: bool = False,
    debug_socratic: bool = False,
    debug_judge: bool = False,
    red_update_every: int = 3,
) -> None:
    """
    Runs multiple iterations in one process.

    When enabled via cfg.execution.keep_* flags, keeps models loaded across iterations (faster),
    relying on cfg.models.*.allowed_gpus to avoid GPU memory conflicts.
    """
    if iterations <= 0:
        raise ValueError("iterations must be >= 1")
    if int(red_update_every) <= 0:
        raise ValueError("red_update_every must be >= 1")

    cfg = AppConfig.parse_obj(read_yaml(config_path))
    curriculum = load_curriculum(Path(cfg.curriculum_path))
    _ensure_dirs(cfg)

    # Preflight + auto-reduce if requested.
    report = preflight_and_autoscale(cfg, curriculum=curriculum, dry_run=True)
    if report.warnings:
        for w in report.warnings:
            warnings.warn(w)

    keep_soc = bool(getattr(cfg.execution, "keep_socratic_loaded", False))
    keep_judge = bool(getattr(cfg.execution, "keep_judge_loaded", False))
    keep_red = bool(getattr(cfg.execution, "keep_red_loaded", False))
    reload_every = int(getattr(cfg.execution, "reload_every_iters", 0) or 0)

    rng = random.Random(int(seed)) if seed is not None else random.Random()
    iter_index = 0

    def _run_chunk(n: int) -> None:
        nonlocal iter_index
        with ExitStack() as stack:
            soc = stack.enter_context(load_socratic(cfg.models.socratic, for_training=True)) if keep_soc else None
            judge = stack.enter_context(load_judge(cfg.models.judge)) if keep_judge else None
            red = stack.enter_context(load_red(cfg.models.red, for_training=False)) if keep_red else None

            for _ in range(int(n)):
                iter_index += 1
                bucket = choose_bucket(curriculum, topic_spec=topic, difficulty_spec=difficulty, rng=rng)
                run_iteration_cfg(
                    cfg,
                    curriculum,
                    topic=bucket.topic,
                    difficulty=bucket.difficulty,
                    socratic_lm=soc,
                    judge_lm=judge,
                    red_lm=red,
                    iteration_index=int(iter_index),
                    debug_red=bool(debug_red),
                    debug_socratic=bool(debug_socratic),
                    debug_judge=bool(debug_judge),
                    red_update_every=int(red_update_every),
                )

    if reload_every > 0:
        remaining = int(iterations)
        while remaining > 0:
            chunk = min(int(reload_every), remaining)
            _run_chunk(chunk)
            remaining -= chunk
    else:
        _run_chunk(int(iterations))
