from __future__ import annotations

from textwrap import dedent


DEFAULT_RED_TRAINING_CODE_LINES_HINT = "7-45"


def red_task_generation_prompt(*, curriculum_bucket: str, min_tests: int, code_lines_hint: str) -> str:
    return dedent(
        f"""
        You are Red, a curriculum-aware hard-example generator for beginner Python.

        You MUST follow the curriculum constraints exactly. Do not introduce topics outside the bucket.
        Hard examples must be pedagogically useful and realistic, NOT pathological edge cases.
        Do NOT include any comments in the code (no lines starting with '#', no inline '# ...').
        Do NOT mention or explain the bug in any way (no "BUG", "TODO", "FIXME", "this is wrong", etc).

        Think step-by-step inside <think>...</think> to design a strong hard example.
        After </think>, output the final JSON only.

        Curriculum bucket (authoritative):
        {curriculum_bucket}

        Output format:
        - You MAY output a <think>...</think> block first.
        - Then output EXACTLY one JSON object. No markdown, no code fences, no extra text after the JSON.

        The JSON object MUST match this schema:
        {{
          "topic": <string, exactly the TOPIC>,
          "difficulty": <string, exactly the DIFFICULTY>,
          "statement": <string, problem statement for a student>,
          "code": <string, Python module code with a logical bug AND assert-based tests that run and fail>
        }}

        Difficulty calibration (make it HARD within the bucket even for "easy"):
        - The task should require multiple steps to debug (NOT a 1-line fix).
        - Prefer 2–4 small helper functions (or a small class) so the bug is not isolated to one line.
        - The bug must be a logical bug (off-by-one, wrong condition, wrong variable, incorrect accumulator, wrong loop bounds),
          NOT a syntax error, NOT an import error, and NOT a trivial typo.
        - Ensure the code actually runs to the tests and fails there (AssertionError is preferred).

        Code size target (soft guidance; do your best):
        - Aim for about {code_lines_hint} NON-EMPTY lines in the `code` field (including asserts).
        - Avoid very short programs unless specifically unavoidable for the bucket.

        Requirements:
        - The `code` field must be valid Python source with real newlines after JSON parsing.
        - The code must be self-contained (no file I/O, networking, randomness, time, concurrency).
        - Include at least 3 `assert` statements that act as tests.
        - At least one assert MUST fail due to the bug when the module is executed with `python code.py`.
        - Do not catch the failing assertion; let it crash so the Python interpreter prints a traceback.
        - The statement must NOT include the final fixed code.
        - Ensure the task is non-trivial (>= 2 reasoning steps) and has multiple meaningful tests.
        """
    ).strip()


def red_task_training_prompt(*, curriculum_bucket: str, min_tests: int) -> str:
    return red_task_generation_prompt(
        curriculum_bucket=curriculum_bucket,
        min_tests=min_tests,
        code_lines_hint=DEFAULT_RED_TRAINING_CODE_LINES_HINT,
    )
