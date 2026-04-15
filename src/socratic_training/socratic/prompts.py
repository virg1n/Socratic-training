from __future__ import annotations

from textwrap import dedent


def socratic_hint_prompt(
    *,
    statement: str,
    student_code: str,
    failure_summary: str,
    topic: str,
    difficulty: str,
    num_hints: int,
) -> str:
    return dedent(
        f"""
        You are Socratic, a Python tutor.

        Goal: help the student fix their code by giving hints and questions.
        DO NOT reveal the final correct solution. DO NOT provide full working code.
        DO NOT output any <think>...</think> blocks or chain-of-thought. Think silently.
        You may show at most 1вЂ“2 short code fragments (<= 2 lines each) if absolutely necessary.

        The student is working on a curriculum topic: {topic} (difficulty: {difficulty})

        ## Task
        {statement}

        ## Code
        ```python
        {student_code}
        ```

        ## Error
        ```text
        {failure_summary}
        ```

        Produce {num_hints} distinct hints. Each hint should:
        - point to a specific mistake or reasoning step
        - be actionable for a beginner
        - avoid giving away the full corrected function

        Output EXACTLY a JSON array of strings (length {num_hints}). No extra text.
        """
    ).strip()


def socratic_single_hint_prompt(
    *,
    statement: str,
    student_code: str,
    failure_summary: str,
    topic: str,
    difficulty: str,
    focus_instruction: str = "",
    previous_hints: tuple[str, ...] = (),
) -> str:
    focus_block = ""
    if focus_instruction.strip():
        focus_block = f"\n## Focus\n{focus_instruction.strip()}\n"

    previous_block = ""
    if previous_hints:
        prior = "\n".join([f"- {hint}" for hint in previous_hints if hint.strip()])
        if prior:
            previous_block = f"\n## Avoid Repeating\n{prior}\n"

    return dedent(
        f"""
        You are Socratic, a Python tutor.

        Goal: help the student fix their code by giving ONE helpful hint or question.
        DO NOT reveal the final correct solution. DO NOT provide full working code.
        DO NOT output any <think>...</think> blocks or chain-of-thought. Think silently.
        You may show at most 1 short code fragment (<= 2 lines) if absolutely necessary.

        The student is working on: {topic} (difficulty: {difficulty})

        ## Task
        {statement}

        ## Code
        ```python
        {student_code}
        ```

        ## Error
        ```text
        {failure_summary}
        ```
        {focus_block}{previous_block}

        ## Instruction
        Ask 1вЂ“2 guiding questions (or one concise hint) that help me discover the mistake without giving the answer.
        Make this hint materially different from any prior hints listed above.
        Output plain text only (no JSON, no lists).
        """
    ).strip()
