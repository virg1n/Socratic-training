from __future__ import annotations

from textwrap import dedent


SCORE_FIELDS = [
    "not_too_direct",
    "relevant",
    "pedagogical_value",
    "bug_localization_help",
]


def _hints_block(hints: list[str]) -> str:
    return "\n".join([f"[{i}] {h}" for i, h in enumerate(hints)])


def judge_prompt_directness_relevance(
    *,
    topic: str,
    difficulty: str,
    statement: str,
    student_code: str,
    hints: list[str],
) -> str:
    """
    Scores:
    - not_too_direct: 0..10 (0 = direct instruction / points exact bug; 10 = Socratic, non-revealing)
    - relevant: 0..10 (0 = generic/off-topic; 10 = specific to THIS code+error)
    Also outputs:
    - answer_dump: bool (true if it effectively gives the final solution or dumps code)
    """
    hints_block = _hints_block(hints)
    return dedent(
        f"""
        You are Judge, a strict evaluator of Socratic tutoring hints for beginner Python.

        Think step-by-step inside <think>...</think>.
        After </think>, output JSON only (no markdown, no extra text).

        Context:
        Topic: {topic}
        Difficulty: {difficulty}

        Problem:
        {statement}

        Student code (buggy):
        {student_code}

        Hints to evaluate:
        {hints_block}

        Scoring rules:
        - not_too_direct (0..10):
          - 0 = directly tells what to change or exactly points at the bug ("change X to Y", "the bug is on line ...")
          - 10 = Socratic questions/hints that guide discovery without revealing the fix
        - relevant (0..10):
          - 0 = generic advice not tied to this code/error
          - 10 = clearly tied to this specific code and observed failure
        - answer_dump (bool):
          - true if the hint gives the full solution, provides full working code, or includes substantial code dumps

        Output EXACTLY this JSON schema:
        {{
          "items": [
            {{
              "id": 0,
              "not_too_direct": 0,
              "relevant": 0,
              "answer_dump": false
            }}
          ]
        }}

        Requirements:
        - Provide one item for every hint id 0..{len(hints)-1}.
        - Use the full range when appropriate (avoid giving the same score to everything).
        """
    ).strip()


def judge_prompt_pedagogy_localization(
    *,
    topic: str,
    difficulty: str,
    statement: str,
    student_code: str,
    hints: list[str],
) -> str:
    """
    Scores:
    - pedagogical_value: 0..5
    - bug_localization_help: 0..5
    """
    hints_block = _hints_block(hints)
    return dedent(
        f"""
        You are Judge, a strict evaluator of Socratic tutoring hints for beginner Python.

        Think step-by-step inside <think>...</think>.
        After </think>, output JSON only (no markdown, no extra text).

        Context:
        Topic: {topic}
        Difficulty: {difficulty}

        Problem:
        {statement}

        Student code (buggy):
        {student_code}

        Hints to evaluate:
        {hints_block}

        Scoring rules:
        - pedagogical_value (0..5):
          - 0 = confusing or unhelpful
          - 5 = teaches a useful idea or debugging approach for beginners
        - bug_localization_help (0..5):
          - 0 = does not narrow down where the bug is
          - 5 = strongly helps pinpoint the bug location/condition without giving the full fix

        Output EXACTLY this JSON schema:
        {{
          "items": [
            {{
              "id": 0,
              "pedagogical_value": 0,
              "bug_localization_help": 0
            }}
          ]
        }}

        Requirements:
        - Provide one item for every hint id 0..{len(hints)-1}.
        - Use the full range when appropriate (avoid giving the same score to everything).
        """
    ).strip()

