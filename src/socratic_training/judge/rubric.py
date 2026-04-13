from __future__ import annotations

from textwrap import dedent


RUBRIC_FIELDS = [
    "did_not_give_final_answer",
    "pedagogical_value",
    "bug_localization_help",
    "curriculum_alignment",
    "beginner_friendliness",
    "actionability_of_hint",
]


def judge_prompt(
    *,
    topic: str,
    difficulty: str,
    statement: str,
    student_code: str,
    hints: list[str],
) -> str:
    hints_block = "\n".join([f"[{i}] {h}" for i, h in enumerate(hints)])
    rubric = "\n".join([f"- {k}: integer 0..5" for k in RUBRIC_FIELDS])
    return dedent(
        f"""
        You are Judge, a strict evaluator of Socratic tutoring hints for beginner Python.

        You MUST follow this rubric and return valid JSON only.
        Prefer pairwise comparisons between hints when deciding the final ranking.

        Rubric subscores (0..5; higher is better):
        {rubric}

        Critical rule:
        - If a hint reveals the full solution (full working code, or an answer-dump that effectively solves the task),
          then did_not_give_final_answer must be 0 and the hint must be ranked last.

        Context:
        Topic: {topic}
        Difficulty: {difficulty}

        Problem:
        {statement}

        Student code (buggy):
        {student_code}

        Hints to evaluate:
        {hints_block}

        Output EXACTLY this JSON object (no extra text):
        {{
          "items": [
            {{
              "id": 0,
              "subscores": {{
                "did_not_give_final_answer": 0,
                "pedagogical_value": 0,
                "bug_localization_help": 0,
                "curriculum_alignment": 0,
                "beginner_friendliness": 0,
                "actionability_of_hint": 0
              }},
              "answer_dump": false,
              "notes": "<short>"
            }}
          ],
          "ranking": [0]
        }}
        """
    ).strip()
