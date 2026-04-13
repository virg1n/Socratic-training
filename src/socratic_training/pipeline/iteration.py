from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from socratic_training.buffers.hard_buffer import HardExampleBuffer
from socratic_training.config import AppConfig
from socratic_training.curriculum import load_curriculum
from socratic_training.judge.scoring import score_hints_with_lm
from socratic_training.memory import preflight_and_autoscale
from socratic_training.models.loader import load_judge, load_socratic
from socratic_training.red.generator import generate_red_tasks
from socratic_training.red.schema import RedTask
from socratic_training.rl.grpo import GrpoTrajectory, train_socratic_grpo
from socratic_training.socratic.hints import generate_hints_with_lm
from socratic_training.utils.events import append_event
from socratic_training.utils.io import read_yaml
from socratic_training.validation.task_validator import TaskValidation, validate_red_task


def _ensure_dirs(cfg: AppConfig) -> None:
    Path(cfg.logging.out_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.hard_buffer_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.red_dpo_pairs_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.models.socratic.adapter_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.models.red.adapter_dir).mkdir(parents=True, exist_ok=True)


def run_iteration(config_path: Path, *, topic: str, difficulty: str) -> None:
    cfg = AppConfig.parse_obj(read_yaml(config_path))
    curriculum = load_curriculum(Path(cfg.curriculum_path))
    _ensure_dirs(cfg)

    # Preflight + auto-reduce if requested.
    report = preflight_and_autoscale(cfg, curriculum=curriculum, dry_run=True)
    if report.warnings:
        for w in report.warnings:
            warnings.warn(w)

    append_event(Path(cfg.logging.jsonl_path), {"type": "iteration_start", "topic": topic, "difficulty": difficulty})

    # A-B: Red generates candidates (include buggy solutions).
    red = generate_red_tasks(cfg, curriculum=curriculum, topic=topic, difficulty=difficulty)
    total_generated = len(red.tasks)
    # Strict bucket enforcement: Red must not drift to other topics/difficulties.
    bucket_tasks = [t for t in red.tasks if t.topic == topic and t.difficulty == difficulty]
    if len(bucket_tasks) != len(red.tasks):
        append_event(
            Path(cfg.logging.jsonl_path),
            {
                "type": "red_bucket_mismatch",
                "requested_topic": topic,
                "requested_difficulty": difficulty,
                "kept": len(bucket_tasks),
                "dropped": len(red.tasks) - len(bucket_tasks),
            },
        )
    red.tasks = bucket_tasks
    append_event(
        Path(cfg.logging.jsonl_path),
        {
            "type": "red_generation",
            "topic": topic,
            "difficulty": difficulty,
            "num_tasks_generated": total_generated,
            "num_tasks_kept": len(red.tasks),
            "errors": list(red.errors),
        },
    )

    if not red.tasks:
        append_event(
            Path(cfg.logging.jsonl_path),
            {
                "type": "iteration_abort",
                "reason": "red_no_tasks",
                "topic": topic,
                "difficulty": difficulty,
                "red_errors": list(red.errors),
            },
        )
        if total_generated > 0 and not red.errors:
            raise RuntimeError(
                "Red produced tasks but none matched the requested (topic,difficulty). "
                "Ensure Red outputs topic/difficulty exactly as provided by the curriculum bucket."
            )
        raise RuntimeError(f"Red produced no parsable tasks. Errors: {list(red.errors)}")

    # C: Validate tasks (schema already validated in generator).
    seen: Set[str] = set()
    valid_pairs: List[Tuple[RedTask, TaskValidation]] = []
    validations = []
    for t in red.tasks:
        v = validate_red_task(cfg, curriculum=curriculum, task=t, seen_fingerprints=seen)
        validations.append(v)
        if v.ok:
            valid_pairs.append((t, v))

    append_event(
        Path(cfg.logging.jsonl_path),
        {
            "type": "task_validation",
            "topic": topic,
            "difficulty": difficulty,
            "num_valid": len(valid_pairs),
            "num_total": len(red.tasks),
            "invalid_reasons": [r for v in validations for r in v.reasons][:200],
        },
    )

    valid_tasks = [t for (t, _v) in valid_pairs]
    if not valid_tasks:
        reason_counts = Counter([r for v in validations for r in v.reasons])
        top = reason_counts.most_common(12)
        append_event(
            Path(cfg.logging.jsonl_path),
            {"type": "iteration_abort", "reason": "no_valid_tasks", "topic": topic, "difficulty": difficulty},
        )
        append_event(
            Path(cfg.logging.jsonl_path),
            {
                "type": "task_validation_summary",
                "topic": topic,
                "difficulty": difficulty,
                "reason_counts_top": top,
            },
        )
        raise RuntimeError(f"No valid Red tasks after validation. Top reasons: {top}")

    # D: Socratic generates multiple hints per task (group rollouts).
    task_hints: List[Dict[str, object]] = []
    num_hints = int(cfg.training.grpo.group_size)
    with load_socratic(cfg.models.socratic, for_training=False) as soc:
        for idx, (t, v) in enumerate(valid_pairs):
            hg = generate_hints_with_lm(
                cfg,
                lm=soc,
                topic=t.topic,
                difficulty=t.difficulty,
                statement=t.statement,
                student_code=t.buggy_solution,
                buggy_test_details=v.buggy.details if v.buggy else {},
                num_hints=num_hints,
            )
            task_hints.append(
                {
                    "task_index": idx,
                    "task": t,
                    "hint_gen": hg,
                    "validation": v,
                }
            )

    append_event(
        Path(cfg.logging.jsonl_path),
        {
            "type": "socratic_hint_gen",
            "topic": topic,
            "difficulty": difficulty,
            "num_tasks": len(task_hints),
            "num_hints_each": num_hints,
        },
    )

    # E: Judge evaluates/ranks hints.
    judged: List[Dict[str, object]] = []
    with load_judge(cfg.models.judge) as judge:
        for item in task_hints:
            t: RedTask = item["task"]  # type: ignore[assignment]
            hg = item["hint_gen"]
            hints = getattr(hg, "hints", [])
            jr = score_hints_with_lm(
                cfg,
                lm=judge,
                topic=t.topic,
                difficulty=t.difficulty,
                statement=t.statement,
                student_code=t.buggy_solution,
                hints=list(hints),
            )
            judged.append({**item, "judge_result": jr})

    append_event(
        Path(cfg.logging.jsonl_path),
        {
            "type": "judge_scoring",
            "topic": topic,
            "difficulty": difficulty,
            "num_tasks": len(judged),
        },
    )

    # Build DPO preference pairs for Red: prefer tasks where Socratic struggled.
    bucket_prompt = curriculum.bucket_prompt(topic=topic, difficulty=difficulty)
    best_by_task: List[Tuple[float, Dict[str, object]]] = []
    for item in judged:
        jr = item["judge_result"]
        best = max((s.final_reward for s in getattr(jr, "scores", [])), default=-5.0)
        best_by_task.append((float(best), item))
    best_by_task.sort(key=lambda x: x[0])  # ascending: hardest first
    if len(best_by_task) >= 2:
        easiest = best_by_task[-1][1]
        rejected_task = easiest["task"]
        for best, item in best_by_task[: max(1, len(best_by_task) // 2)]:
            chosen_task = item["task"]
            append_event(
                Path(cfg.logging.red_dpo_pairs_path),
                {
                    "type": "red_dpo_pair",
                    "topic": topic,
                    "difficulty": difficulty,
                    "prompt": bucket_prompt,
                    "chosen": chosen_task.dict(),  # type: ignore[union-attr]
                    "rejected": rejected_task.dict(),  # type: ignore[union-attr]
                    "chosen_best_reward": best,
                    "rejected_best_reward": float(best_by_task[-1][0]),
                },
            )

    # Build trajectories for GRPO.
    trajectories: List[GrpoTrajectory] = []
    per_task_best: List[Tuple[int, float]] = []
    for item in judged:
        task_index = int(item["task_index"])  # type: ignore[arg-type]
        hg = item["hint_gen"]
        jr = item["judge_result"]

        hints: List[str] = list(getattr(hg, "hints", []))
        prompt_ids: List[int] = list(getattr(hg, "prompt_ids", []))
        completion_ids: List[List[int]] = list(getattr(hg, "completion_ids", []))

        # Map hint id -> reward (default pessimistic).
        reward_map: Dict[int, float] = {i: -5.0 for i in range(len(hints))}
        for s in getattr(jr, "scores", []):
            reward_map[int(s.id)] = float(s.final_reward)

        best = max(reward_map.values()) if reward_map else -5.0
        per_task_best.append((task_index, best))

        for hid in range(min(len(hints), len(completion_ids))):
            trajectories.append(
                GrpoTrajectory(
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids[hid],
                    reward=reward_map.get(hid, -5.0),
                    group_id=task_index,
                )
            )

    # F: Update Socratic via GRPO.
    stats = train_socratic_grpo(cfg, trajectories=trajectories, output_adapter_dir=Path(cfg.models.socratic.adapter_dir))
    append_event(
        Path(cfg.logging.jsonl_path),
        {
            "type": "grpo_update",
            "topic": topic,
            "difficulty": difficulty,
            "steps": stats.steps,
            "mean_reward": stats.mean_reward,
            "mean_advantage": stats.mean_advantage,
            "loss": stats.loss,
            "num_trajectories": len(trajectories),
        },
    )

    # G: Add hard examples for Red.
    hard = HardExampleBuffer(Path(cfg.logging.hard_buffer_path))
    for item in judged:
        t: RedTask = item["task"]  # type: ignore[assignment]
        hg = item["hint_gen"]
        jr = item["judge_result"]

        # Heuristic: "performed poorly" if best reward is below the median.
        best = max((s.final_reward for s in getattr(jr, "scores", [])), default=-5.0)
        if best < stats.mean_reward:
            hard.add(
                topic=t.topic,
                difficulty=t.difficulty,
                task=t.dict(),
                socratic_hints=list(getattr(hg, "hints", [])),
                judge={
                    "ranking": getattr(jr, "ranking", []),
                    "scores": [
                        {
                            "id": s.id,
                            "subscores": s.subscores,
                            "answer_dump": s.answer_dump,
                            "final_reward": s.final_reward,
                            "notes": s.notes,
                        }
                        for s in getattr(jr, "scores", [])
                    ],
                    "errors": list(getattr(jr, "errors", ())),
                },
                best_reward=best,
            )

    append_event(
        Path(cfg.logging.jsonl_path),
        {
            "type": "iteration_end",
            "topic": topic,
            "difficulty": difficulty,
            "num_valid_tasks": len(valid_tasks),
            "num_scored_tasks": len(judged),
            "hard_examples_added": True,
        },
    )
