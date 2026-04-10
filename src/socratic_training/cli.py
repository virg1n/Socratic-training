from __future__ import annotations

import argparse
import json

from .config import PipelineConfig
from .pipeline import TrainingPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sequential Socratic / Red / Judge training stack.")
    parser.add_argument("--config", default="configs/workstation.toml", help="Path to the TOML configuration file.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preflight", help="Run memory estimation and curriculum parsing.")

    run_iteration = subparsers.add_parser("run-iteration", help="Run one full Red -> Validate -> Socratic -> Judge -> GRPO iteration.")
    run_iteration.add_argument("--topic", default=None, help="Topic name from curriculum.txt.")
    run_iteration.add_argument("--subtopic", default=None, help="Subtopic name from curriculum.txt.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = PipelineConfig.from_toml(args.config)
    pipeline = TrainingPipeline(config)

    if args.command == "preflight":
        print(json.dumps(pipeline.preflight(), indent=2, ensure_ascii=False))
        return

    if args.command == "run-iteration":
        print(json.dumps(pipeline.run_iteration(topic=args.topic, subtopic=args.subtopic), indent=2, ensure_ascii=False))
        return

    parser.error(f"Unknown command: {args.command}")
