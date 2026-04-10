from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .types import CurriculumCatalog, CurriculumSubtopic, CurriculumTopic


HEADER_KEYS = {"TOPIC", "DESCRIPTION", "DIFFICULTIES", "FORBIDDEN", "SUBTOPIC", "OBJECTIVES", "KEYWORDS"}


class CurriculumParseError(ValueError):
    pass


def parse_curriculum_file(path: str | Path) -> CurriculumCatalog:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    topics: dict[str, CurriculumTopic] = {}
    current_topic: CurriculumTopic | None = None
    current_subtopic: CurriculumSubtopic | None = None
    current_list_key: str | None = None

    def flush_subtopic() -> None:
        nonlocal current_subtopic, current_topic
        if current_subtopic is None:
            return
        if not current_subtopic.objectives:
            raise CurriculumParseError(f"Subtopic '{current_subtopic.name}' must declare at least one objective.")
        if current_topic is None:
            raise CurriculumParseError("Found SUBTOPIC before TOPIC.")
        current_topic.subtopics.append(current_subtopic)
        current_subtopic = None

    def flush_topic() -> None:
        nonlocal current_topic
        flush_subtopic()
        if current_topic is None:
            return
        if not current_topic.subtopics:
            raise CurriculumParseError(f"Topic '{current_topic.name}' must include at least one subtopic.")
        topics[current_topic.name] = current_topic
        current_topic = None

    def assign_value(key: str, value: str) -> None:
        nonlocal current_topic, current_subtopic, current_list_key
        current_list_key = None
        if key == "TOPIC":
            flush_topic()
            current_topic = CurriculumTopic(name=value)
            return
        if current_topic is None:
            raise CurriculumParseError(f"{key} declared before TOPIC.")
        if key == "DESCRIPTION":
            current_topic.description = value
            return
        if key == "SUBTOPIC":
            flush_subtopic()
            current_subtopic = CurriculumSubtopic(name=value)
            return
        target = current_subtopic if current_subtopic is not None else current_topic
        if key == "DIFFICULTIES":
            target.difficulties.extend(_parse_inline_list(value))
            return
        if key == "FORBIDDEN":
            target.forbidden.extend(_parse_inline_list(value))
            return
        if current_subtopic is None:
            raise CurriculumParseError(f"{key} must appear inside a SUBTOPIC block.")
        if key == "OBJECTIVES":
            current_subtopic.objectives.extend(_parse_inline_list(value))
            return
        if key == "KEYWORDS":
            current_subtopic.keywords.extend(_parse_inline_list(value))
            return
        raise CurriculumParseError(f"Unsupported key: {key}")

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "---":
            flush_topic()
            current_list_key = None
            continue
        if line.startswith("- "):
            if current_list_key is None:
                raise CurriculumParseError(f"Unexpected list item at line {line_number}: {raw_line}")
            assign_list_item(current_topic, current_subtopic, current_list_key, line[2:].strip())
            continue
        if ":" not in line:
            raise CurriculumParseError(f"Expected KEY: VALUE at line {line_number}: {raw_line}")
        key, value = [part.strip() for part in line.split(":", 1)]
        upper_key = key.upper()
        if upper_key not in HEADER_KEYS:
            raise CurriculumParseError(f"Unknown key '{key}' at line {line_number}.")
        if not value and upper_key in {"DIFFICULTIES", "FORBIDDEN", "OBJECTIVES", "KEYWORDS"}:
            current_list_key = upper_key
            continue
        assign_value(upper_key, value)

    flush_topic()
    if not topics:
        raise CurriculumParseError("Curriculum file is empty.")
    normalized = {name: _deduplicate_topic(topic) for name, topic in topics.items()}
    return CurriculumCatalog(topics=normalized, source_path=path)


def assign_list_item(
    current_topic: CurriculumTopic | None,
    current_subtopic: CurriculumSubtopic | None,
    key: str,
    value: str,
) -> None:
    if current_topic is None:
        raise CurriculumParseError(f"{key} list declared before TOPIC.")
    target = current_subtopic if current_subtopic is not None else current_topic
    if key == "DIFFICULTIES":
        target.difficulties.append(value)
        return
    if key == "FORBIDDEN":
        target.forbidden.append(value)
        return
    if current_subtopic is None:
        raise CurriculumParseError(f"{key} list must appear inside a SUBTOPIC block.")
    if key == "OBJECTIVES":
        current_subtopic.objectives.append(value)
        return
    if key == "KEYWORDS":
        current_subtopic.keywords.append(value)
        return
    raise CurriculumParseError(f"Unsupported list key: {key}")


def _parse_inline_list(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _deduplicate_topic(topic: CurriculumTopic) -> CurriculumTopic:
    subtopics = []
    for subtopic in topic.subtopics:
        subtopics.append(
            replace(
                subtopic,
                difficulties=_dedupe(subtopic.difficulties),
                objectives=_dedupe(subtopic.objectives),
                keywords=_dedupe(subtopic.keywords),
                forbidden=_dedupe(subtopic.forbidden),
            )
        )
    return replace(topic, difficulties=_dedupe(topic.difficulties), forbidden=_dedupe(topic.forbidden), subtopics=subtopics)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        output.append(normalized)
    return output
