from __future__ import annotations

from pathlib import Path

from socratic_training.config import AppConfig
from socratic_training.curriculum import load_curriculum
from socratic_training.memory import preflight_and_autoscale
from socratic_training.utils.io import read_yaml


def run_preflight(config_path: Path) -> None:
    cfg = AppConfig.parse_obj(read_yaml(config_path))
    curriculum = load_curriculum(Path(cfg.curriculum_path))
    report = preflight_and_autoscale(cfg, curriculum=curriculum, dry_run=True)
    print("Preflight estimates:")
    for k, v in sorted(report.estimates_gb.items()):
        if k.endswith("_params_b"):
            print(f"- {k}: {v:.2f}B params (estimated)")
        else:
            print(f"- {k}: {v:.1f}GB")
    if report.suggested_updates:
        print("Auto-reduction suggested updates:")
        for k, v in report.suggested_updates.items():
            print(f"- {k}: {v}")
    if report.warnings:
        print("Warnings:")
        for w in report.warnings:
            print(f"- {w}")
