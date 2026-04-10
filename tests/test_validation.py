from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from socratic_training.config import PipelineConfig
from socratic_training.types import CurriculumBucket, TaskExample, TaskTest
from socratic_training.validators import TaskValidator


def build_task() -> TaskExample:
    return TaskExample(
        topic="functions / parameters and return values",
        difficulty="beginner",
        statement="Write a function named solve that converts a number of minutes into seconds and returns the result.",
        canonical_solution="""def solve(minutes):\n    return minutes * 60\n""",
        buggy_solution="""def solve(minutes):\n    return minutes * 10\n""",
        bug_explanation="Uses the wrong conversion factor.",
        tests=[
            TaskTest(name="one_minute", code="assert solve(1) == 60"),
            TaskTest(name="zero_minutes", code="assert solve(0) == 0"),
            TaskTest(name="three_minutes", code="assert solve(3) == 180"),
        ],
        expected_learning_objectives=["define a function with parameters", "return a computed result"],
    )


class ValidationTests(unittest.TestCase):
    def test_valid_task_passes(self) -> None:
        validator = TaskValidator(PipelineConfig())
        bucket = CurriculumBucket(
            topic="functions",
            subtopic="parameters and return values",
            description="defining functions",
            difficulties=["beginner"],
            objectives=["define a function with parameters", "return a computed result"],
            keywords=["def", "return"],
            forbidden=["decorators"],
        )
        result = validator.validate(build_task(), bucket, [])
        self.assertTrue(result.accepted, msg=result.reasons)
        self.assertTrue(result.canonical_execution.all_passed)
        self.assertEqual(result.buggy_execution.failed, 2)

    def test_identical_buggy_solution_is_rejected(self) -> None:
        validator = TaskValidator(PipelineConfig())
        bucket = CurriculumBucket(
            topic="functions",
            subtopic="parameters and return values",
            description="defining functions",
            difficulties=["beginner"],
            objectives=["define a function with parameters", "return a computed result"],
            keywords=["def", "return"],
            forbidden=["decorators"],
        )
        task = build_task()
        task.buggy_solution = task.canonical_solution
        result = validator.validate(task, bucket, [])
        self.assertFalse(result.accepted)
        self.assertIn("Buggy solution passes every test.", result.reasons)


if __name__ == "__main__":
    unittest.main()
