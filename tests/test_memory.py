from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from socratic_training.config import PipelineConfig
from socratic_training.memory import estimate_judge_inference, preflight_and_tune


class MemoryTests(unittest.TestCase):
    def test_preflight_returns_all_scenarios(self) -> None:
        config = PipelineConfig()
        tuned, scenarios = preflight_and_tune(config)
        self.assertEqual(len(scenarios), 4)
        self.assertEqual(tuned.socratic.group_size, 4)

    def test_judge_estimate_has_name(self) -> None:
        estimate = estimate_judge_inference(PipelineConfig())
        self.assertEqual(estimate.name, "judge_inference")
        self.assertGreater(estimate.gpu_gb, 0.0)


if __name__ == "__main__":
    unittest.main()
