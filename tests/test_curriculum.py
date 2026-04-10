from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from socratic_training.curriculum import parse_curriculum_file


SAMPLE = """\
TOPIC: functions
DESCRIPTION: defining small functions
DIFFICULTIES: beginner, easy
FORBIDDEN:
- decorators
SUBTOPIC: parameters and return values
OBJECTIVES:
- define a function with parameters
- return a computed result
KEYWORDS:
- def
- return
---
TOPIC: loops
DESCRIPTION: simple counting loops
DIFFICULTIES:
- beginner
SUBTOPIC: for loops
OBJECTIVES:
- iterate with range
KEYWORDS: for, range
"""


class CurriculumParserTests(unittest.TestCase):
    def test_parser_builds_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "curriculum.txt"
            path.write_text(SAMPLE, encoding="utf-8")
            catalog = parse_curriculum_file(path)

        buckets = catalog.list_buckets()
        self.assertEqual(len(buckets), 2)
        self.assertEqual(buckets[0].topic, "functions")
        self.assertIn("decorators", buckets[0].forbidden)
        self.assertIn("return a computed result", buckets[0].objectives)


if __name__ == "__main__":
    unittest.main()
