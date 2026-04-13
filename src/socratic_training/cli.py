from __future__ import annotations

import argparse
from pathlib import Path

from socratic_training.pipeline.iteration import run_iteration
from socratic_training.pipeline.loop import run_loop
from socratic_training.preflight import run_preflight
from socratic_training.red.debug import run_red_debug
from socratic_training.red.train_dpo import run_red_dpo
from socratic_training.red.train_sft import run_red_sft


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="socratic-train")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("preflight", help="Estimate memory + sanity-check config.")
    p_pre.add_argument("--config", type=Path, required=True)

    p_run = sub.add_parser("run-iteration", help="Run one Red→Validate→Socratic→Judge→GRPO iteration.")
    p_run.add_argument("--config", type=Path, required=True)
    p_run.add_argument("--topic", type=str, required=True)
    p_run.add_argument("--difficulty", type=str, required=True)

    p_loop = sub.add_parser("run-loop", help="Run multiple iterations in one process (optional model reuse).")
    p_loop.add_argument("--config", type=Path, required=True)
    p_loop.add_argument("--topic", type=str, required=True)
    p_loop.add_argument("--difficulty", type=str, required=True)
    p_loop.add_argument("--iterations", type=int, required=True)

    p_rsft = sub.add_parser("train-red-sft", help="Train Red LoRA adapters via SFT on hard buffer.")
    p_rsft.add_argument("--config", type=Path, required=True)

    p_rdpo = sub.add_parser("train-red-dpo", help="Train Red LoRA adapters via DPO on preference pairs.")
    p_rdpo.add_argument("--config", type=Path, required=True)

    p_rdbg = sub.add_parser("debug-red", help="Generate+validate Red tasks (no Socratic/Judge).")
    p_rdbg.add_argument("--config", type=Path, required=True)
    p_rdbg.add_argument("--topic", type=str, required=True)
    p_rdbg.add_argument("--difficulty", type=str, required=True)

    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.cmd == "preflight":
        run_preflight(args.config)
        return

    if args.cmd == "run-iteration":
        run_iteration(args.config, topic=args.topic, difficulty=args.difficulty)
        return

    if args.cmd == "run-loop":
        run_loop(args.config, topic=args.topic, difficulty=args.difficulty, iterations=int(args.iterations))
        return

    if args.cmd == "train-red-sft":
        run_red_sft(args.config)
        return

    if args.cmd == "train-red-dpo":
        run_red_dpo(args.config)
        return

    if args.cmd == "debug-red":
        run_red_debug(args.config, topic=args.topic, difficulty=args.difficulty)
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":  # pragma: no cover
    main()
