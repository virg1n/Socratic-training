from __future__ import annotations

import json

from .types import CurriculumBucket, HintCandidate, TaskExample


def red_task_generation_prompt(bucket: CurriculumBucket, candidate_count: int, existing_statements: list[str]) -> str:
    excluded = "\n".join(f"- {statement}" for statement in existing_statements[:12]) or "- none yet"
    schema = {
        "topic": f"{bucket.topic} / {bucket.subtopic}",
        "difficulty": bucket.difficulties[0] if bucket.difficulties else "beginner",
        "statement": "Student-facing Python problem statement.",
        "canonical_solution": "Python code defining solve(...).",
        "buggy_solution": "",
        "bug_explanation": "",
        "tests": [{"name": "test_case_name", "code": "assert solve(...) == ..."}],
        "expected_learning_objectives": bucket.objectives[:2] or ["Use the current curriculum objective."],
    }
    return f"""You are Red, a curriculum-bounded hard-example generator for a Python tutoring model.

Your job is to create EXACTLY {candidate_count} candidate tasks in JSON array form.

Curriculum constraints:
- Topic: {bucket.topic}
- Subtopic: {bucket.subtopic}
- Allowed difficulties: {", ".join(bucket.difficulties) or "beginner"}
- Required learning objectives:
{_bullet_list(bucket.objectives)}
- Recommended keywords:
{_bullet_list(bucket.keywords)}
- Forbidden or out-of-scope areas:
{_bullet_list(bucket.forbidden)}

Pedagogical constraints:
- Stay inside a standard Python course.
- Do not use pathological edge cases, obscure interpreter behavior, metaprogramming, or advanced libraries.
- Make tasks useful for tutoring; they should expose common student reasoning mistakes.
- Each canonical solution must define a single callable `solve`.
- Keep solutions compact and beginner-readable.
- Provide 3 to 6 assert-based tests per task.
- `expected_learning_objectives` must be chosen from the curriculum objectives.

Similarity exclusions:
{excluded}

Output format:
- Return a JSON array only.
- Every element must contain this schema:
{json.dumps(schema, indent=2)}
- Leave `buggy_solution` empty. It will be generated in a second stage.
- Do not wrap the JSON in markdown fences.
"""


def red_buggy_solution_prompt(task: TaskExample, bucket: CurriculumBucket) -> str:
    return f"""You are Red. Produce a buggy but syntactically valid Python solution for the task below.

Curriculum bucket:
- Topic: {bucket.topic}
- Subtopic: {bucket.subtopic}
- Objectives:
{_bullet_list(bucket.objectives)}
- Forbidden:
{_bullet_list(bucket.forbidden)}

Task statement:
{task.statement}

Canonical solution:
```python
{task.canonical_solution}
```

Tests:
{_bullet_list(test.code for test in task.tests)}

Requirements:
- Keep the same `solve` signature as the canonical solution.
- Introduce a logical bug, not a syntax error.
- The buggy solution should fail at least one meaningful test but still look plausible for a beginner.
- Do not add imports that go beyond a beginner Python course.
- Output JSON only with keys `buggy_solution` and `bug_explanation`.
"""


def socratic_hint_prompt(task: TaskExample) -> str:
    return f"""You are the Socratic tutor. Give exactly one short hint for the student.

Rules:
- Do NOT reveal the final answer.
- Do NOT provide the full corrected code.
- Focus on the student's next reasoning step.
- Point toward the likely bug or misconception.
- Keep the tone beginner-friendly and actionable.
- Keep the hint under 90 words.

Task statement:
{task.statement}

Student's buggy solution:
```python
{task.buggy_solution}
```

Learning objectives:
{_bullet_list(task.expected_learning_objectives)}

Respond with the hint text only.
"""


def judge_single_hint_prompt(task: TaskExample, hint: str) -> str:
    schema = {
        "did_not_give_final_answer": 0.0,
        "pedagogical_value": 0.0,
        "bug_localization_help": 0.0,
        "curriculum_alignment": 0.0,
        "beginner_friendliness": 0.0,
        "actionability_of_hint": 0.0,
        "rationale": "Short explanation.",
    }
    return f"""You are Judge, a strict evaluator for Socratic tutoring hints.

Grade the hint on a 0.0 to 1.0 scale for each rubric item:
- did_not_give_final_answer: high only if the hint avoids solving the problem.
- pedagogical_value: high if it develops student reasoning.
- bug_localization_help: high if it points the student toward the likely mistake.
- curriculum_alignment: high if it stays inside the stated curriculum.
- beginner_friendliness: high if the wording suits a beginner.
- actionability_of_hint: high if the student can act on it immediately.

Hard rule:
- If the hint effectively gives the final answer or corrected implementation, `did_not_give_final_answer` must be near 0.

Task statement:
{task.statement}

Buggy solution:
```python
{task.buggy_solution}
```

Canonical solution for evaluator reference:
```python
{task.canonical_solution}
```

Learning objectives:
{_bullet_list(task.expected_learning_objectives)}

Hint to judge:
{hint}

Return JSON only:
{json.dumps(schema, indent=2)}
"""


def judge_ranking_prompt(task: TaskExample, hints: list[HintCandidate]) -> str:
    joined_hints = "\n\n".join(f"Hint {candidate.index}:\n{candidate.hint}" for candidate in hints)
    schema = {"ranking": [candidate.index for candidate in hints], "rationale": "Brief ranking explanation."}
    return f"""You are Judge. Rank the hint candidates from best to worst for this task.

Priorities, in order:
1. Does not reveal the final answer.
2. Helps the student find the bug or next step.
3. Stays inside the curriculum.
4. Is actionable and beginner-friendly.

Task:
{task.statement}

Hints:
{joined_hints}

Return JSON only:
{json.dumps(schema, indent=2)}
"""


def red_sft_prompt(bucket: CurriculumBucket, task: TaskExample) -> str:
    return f"""Create a curriculum-valid Python tutoring task for:
- Topic: {bucket.topic}
- Subtopic: {bucket.subtopic}
- Objectives: {", ".join(bucket.objectives)}

Return a structured task package that stays inside the curriculum and includes tests.
"""


def red_dpo_prompt(bucket: CurriculumBucket) -> str:
    return f"""Generate a Python tutoring task for {bucket.topic} / {bucket.subtopic}.

Stay within:
- Objectives: {", ".join(bucket.objectives)}
- Forbidden: {", ".join(bucket.forbidden)}
"""


def _bullet_list(items) -> str:
    values = list(items)
    if not values:
        return "- none"
    return "\n".join(f"- {item}" for item in values)
