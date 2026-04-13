from __future__ import annotations

import warnings
from contextlib import ExitStack
from pathlib import Path

from socratic_training.config import AppConfig
from socratic_training.curriculum import load_curriculum
from socratic_training.memory import preflight_and_autoscale
from socratic_training.models.loader import load_judge, load_red, load_socratic
from socratic_training.pipeline.iteration import _ensure_dirs, run_iteration_cfg
from socratic_training.utils.io import read_yaml


def run_loop(config_path: Path, *, topic: str, difficulty: str, iterations: int) -> None:
    """
    Runs multiple iterations in one process.

    When enabled via cfg.execution.keep_* flags, keeps models loaded across iterations (faster),
    relying on cfg.models.*.allowed_gpus to avoid GPU memory conflicts.
    """
    if iterations <= 0:
        raise ValueError("iterations must be >= 1")

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

    with ExitStack() as stack:
        soc = stack.enter_context(load_socratic(cfg.models.socratic, for_training=True)) if keep_soc else None
        judge = stack.enter_context(load_judge(cfg.models.judge)) if keep_judge else None
        red = stack.enter_context(load_red(cfg.models.red, for_training=False)) if keep_red else None

        for _ in range(int(iterations)):
            run_iteration_cfg(
                cfg,
                curriculum,
                topic=topic,
                difficulty=difficulty,
                socratic_lm=soc,
                judge_lm=judge,
                red_lm=red,
            )

