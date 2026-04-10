from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .buffer import HardExampleBuffer, JsonlLogger
from .config import PipelineConfig
from .curriculum import parse_curriculum_file
from .judge import JudgeClient
from .memory import preflight_and_tune
from .model_manager import StagedModelManager
from .red import RedGenerator, RedTrainer
from .socratic import SocraticRolloutGenerator, StagedGRPOTrainer
from .types import CurriculumBucket, HardExample
from .utils import configure_logging, safe_mean, slugify, write_json
from .validators import TaskValidator


class TrainingPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        configure_logging(config.log_level)
        self.config, self.scenarios = preflight_and_tune(config)
        self.curriculum = parse_curriculum_file(self.config.paths.curriculum_file)
        self.models = StagedModelManager(self.config)
        self.validator = TaskValidator(self.config)
        self.red_generator = RedGenerator(self.config, self.models)
        self.red_trainer = RedTrainer(self.config, self.models)
        self.socratic_generator = SocraticRolloutGenerator(self.config, self.models)
        self.socratic_trainer = StagedGRPOTrainer(self.config, self.models)
        self.judge = JudgeClient(self.config, self.models)
        self.hard_buffer = HardExampleBuffer(self.config.paths.hard_buffer_file)
        self.coverage_logger = JsonlLogger(self.config.paths.coverage_log_file)
        self.performance_logger = JsonlLogger(self.config.paths.performance_log_file)
        self.iteration = 0

    def preflight(self) -> dict[str, object]:
        summary = {
            "config": self.config.to_dict(),
            "memory_scenarios": [asdict(scenario) for scenario in self.scenarios],
            "curriculum_buckets": [asdict(bucket) for bucket in self.curriculum.list_buckets()],
        }
        write_json(Path(self.config.paths.output_dir) / "preflight.json", summary)
        return summary

    def run_iteration(self, topic: str | None = None, subtopic: str | None = None) -> dict[str, object]:
        self.iteration += 1
        bucket = self._select_bucket(topic, subtopic)
        existing_hard_examples = self.hard_buffer.load()
        existing_tasks = [example.task for example in existing_hard_examples if example.bucket_id == bucket.bucket_id]

        generated_tasks = self.red_generator.generate_tasks(bucket, existing_tasks)
        validations = []
        seen_tasks = list(existing_tasks)
        for task in generated_tasks:
            validation = self.validator.validate(task, bucket, seen_tasks)
            validations.append(validation)
            if validation.accepted:
                seen_tasks.append(task)
        valid_tasks = [result.task for result in validations if result.accepted][: self.config.max_valid_tasks_per_iteration]

        if not valid_tasks:
            self.red_trainer.record_feedback(bucket, validations, [])
            summary = {
                "iteration": self.iteration,
                "bucket": bucket.bucket_id,
                "accepted_tasks": 0,
                "warnings": ["No valid Red tasks passed validation."],
            }
            self.coverage_logger.log(summary)
            return summary

        hint_groups = self.socratic_generator.generate_hints(valid_tasks)
        judged_groups = [self.judge.evaluate_group(task, group) for task, group in zip(valid_tasks, hint_groups, strict=False)]
        optimization_stats = self.socratic_trainer.optimize(judged_groups)
        hard_examples = self._extract_hard_examples(bucket, judged_groups)
        for example in hard_examples:
            self.hard_buffer.append(example)

        self.red_trainer.record_feedback(bucket, validations, hard_examples)
        self.red_trainer.maybe_train(self.iteration)
        self._log_iteration(bucket, validations, judged_groups, optimization_stats)

        return {
            "iteration": self.iteration,
            "bucket": bucket.bucket_id,
            "accepted_tasks": len(valid_tasks),
            "hard_examples": len(hard_examples),
            "mean_reward": optimization_stats["mean_reward"],
            "mean_loss": optimization_stats["mean_loss"],
        }

    def _select_bucket(self, topic: str | None, subtopic: str | None) -> CurriculumBucket:
        if topic and subtopic:
            return self.curriculum.get_bucket(topic, subtopic)
        buckets = self.curriculum.list_buckets()
        coverage_counts = self._coverage_counts()
        return min(buckets, key=lambda bucket: coverage_counts.get(bucket.bucket_id, 0))

    def _coverage_counts(self) -> dict[str, int]:
        path = Path(self.config.paths.coverage_log_file)
        if not path.exists():
            return {}
        counts: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            import json

            payload = json.loads(line)
            bucket_id = payload.get("bucket")
            if bucket_id:
                counts[bucket_id] = counts.get(bucket_id, 0) + 1
        return counts

    def _extract_hard_examples(self, bucket: CurriculumBucket, judged_groups) -> list[HardExample]:
        hard_examples: list[HardExample] = []
        for group in judged_groups:
            rewards = [evaluation.final_reward for evaluation in group]
            best_reward = max(rewards)
            worst_reward = min(rewards)
            if best_reward <= self.config.socratic.hard_example_threshold:
                hard_examples.append(
                    HardExample(
                        task=group[0].candidate.task,
                        best_reward=best_reward,
                        worst_reward=worst_reward,
                        bucket_id=bucket.bucket_id,
                        metadata={"spread": best_reward - worst_reward},
                    )
                )
        return hard_examples

    def _log_iteration(self, bucket: CurriculumBucket, validations, judged_groups, optimization_stats) -> None:
        self.coverage_logger.log(
            {
                "iteration": self.iteration,
                "bucket": bucket.bucket_id,
                "accepted_tasks": sum(1 for validation in validations if validation.accepted),
                "rejected_tasks": sum(1 for validation in validations if not validation.accepted),
            }
        )
        self.performance_logger.log(
            {
                "iteration": self.iteration,
                "bucket": bucket.bucket_id,
                "mean_reward": optimization_stats["mean_reward"],
                "mean_loss": optimization_stats["mean_loss"],
                "best_reward": max((max(evaluation.final_reward for evaluation in group) for group in judged_groups), default=0.0),
                "mean_best_reward": safe_mean([max(evaluation.final_reward for evaluation in group) for group in judged_groups]),
            }
        )
        tasks_dir = Path(self.config.paths.output_dir) / "iterations" / f"{self.iteration:04d}-{slugify(bucket.bucket_id)}"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        write_json(tasks_dir / "validations.json", [self._validation_payload(validation) for validation in validations])
        write_json(tasks_dir / "judge.json", [[self._judge_payload(evaluation) for evaluation in group] for group in judged_groups])

    @staticmethod
    def _validation_payload(validation) -> dict[str, object]:
        return {
            "task": validation.task.to_dict(),
            "accepted": validation.accepted,
            "reasons": validation.reasons,
            "canonical_execution": asdict(validation.canonical_execution) if validation.canonical_execution else None,
            "buggy_execution": asdict(validation.buggy_execution) if validation.buggy_execution else None,
        }

    @staticmethod
    def _judge_payload(evaluation) -> dict[str, object]:
        return {
            "hint": evaluation.candidate.hint,
            "rank": evaluation.rank,
            "reward": evaluation.final_reward,
            "rationale": evaluation.rationale,
            "subscores": asdict(evaluation.subscores),
            "disclosure_penalty": evaluation.disclosure_penalty,
            "task": evaluation.candidate.task.to_dict(),
        }
