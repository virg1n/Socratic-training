from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Subtopic:
    name: str
    objectives: Tuple[str, ...]
    keywords: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Topic:
    name: str
    description: str
    difficulties: Tuple[str, ...]
    forbidden: Tuple[str, ...] = ()
    subtopics: Tuple[Subtopic, ...] = ()

    def all_objectives(self) -> Tuple[str, ...]:
        seen: Dict[str, None] = {}
        for s in self.subtopics:
            for obj in s.objectives:
                seen.setdefault(obj, None)
        return tuple(seen.keys())

    def all_keywords(self) -> Tuple[str, ...]:
        seen: Dict[str, None] = {}
        for s in self.subtopics:
            for kw in s.keywords:
                seen.setdefault(kw, None)
        return tuple(seen.keys())


@dataclass(frozen=True)
class Curriculum:
    topics: Tuple[Topic, ...]
    forbidden_global: Tuple[str, ...] = ()

    def topic_index(self) -> Dict[str, Topic]:
        return {t.name: t for t in self.topics}

    def get_topic(self, name: str) -> Topic:
        idx = self.topic_index()
        if name not in idx:
            raise KeyError(f"Unknown topic: {name!r}. Known: {sorted(idx.keys())}")
        return idx[name]

    def buckets(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for t in self.topics:
            for d in t.difficulties:
                out.append((t.name, d))
        return out

    def bucket_prompt(self, topic: str, difficulty: str) -> str:
        t = self.get_topic(topic)
        if difficulty not in t.difficulties:
            raise KeyError(f"Topic {topic!r} does not allow difficulty {difficulty!r}: {t.difficulties}")
        forbidden = list(self.forbidden_global) + list(t.forbidden)
        objectives = list(t.all_objectives())
        keywords = list(t.all_keywords())
        lines: List[str] = []
        lines.append(f"TOPIC: {t.name}")
        lines.append(f"DIFFICULTY: {difficulty}")
        lines.append(f"TOPIC_DESCRIPTION: {t.description}")
        lines.append("ALLOWED_LEARNING_OBJECTIVES:")
        lines += [f"- {o}" for o in objectives] if objectives else ["- (none specified)"]
        lines.append("TOPIC_KEYWORDS:")
        lines += [f"- {k}" for k in keywords] if keywords else ["- (none specified)"]
        lines.append("FORBIDDEN_OR_OUT_OF_SCOPE:")
        lines += [f"- {x}" for x in forbidden] if forbidden else ["- (none specified)"]
        return "\n".join(lines)


class CurriculumFormatError(ValueError):
    pass


_LIST_KEYS = {"DIFFICULTIES", "FORBIDDEN", "OBJECTIVES", "KEYWORDS", "FORBIDDEN_GLOBAL", "OUT_OF_SCOPE"}


def _strip_comment(line: str) -> str:
    s = line.strip()
    if s.startswith("#"):
        return ""
    return line.rstrip("\n")


def _parse_list(lines: Sequence[str], start_i: int) -> Tuple[List[str], int]:
    items: List[str] = []
    i = start_i
    while i < len(lines):
        raw = lines[i].strip()
        if not raw:
            i += 1
            continue
        if raw == "---":
            break
        if ":" in raw and not raw.startswith("-"):
            break
        if raw.startswith("-"):
            item = raw[1:].strip()
            if item:
                items.append(item)
        i += 1
    return items, i


def load_curriculum(path: Path) -> Curriculum:
    text = path.read_text(encoding="utf-8")
    raw_lines = [_strip_comment(l) for l in text.splitlines()]
    lines = [l for l in raw_lines if l.strip()]

    topics: List[Topic] = []
    forbidden_global: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Global forbidden/out-of-scope (optional)
        if line.startswith("FORBIDDEN_GLOBAL:") or line.startswith("OUT_OF_SCOPE:"):
            key = line.split(":", 1)[0].strip()
            lst, i = _parse_list(lines, i + 1)
            forbidden_global.extend(lst)
            continue

        if line == "---":
            i += 1
            continue

        if not line.startswith("TOPIC:"):
            raise CurriculumFormatError(f"Expected 'TOPIC:' or global header at line {i+1}: {lines[i]!r}")

        topic_name = line.split(":", 1)[1].strip()
        description = ""
        difficulties: List[str] = []
        forbidden: List[str] = []
        subtopics: List[Subtopic] = []

        i += 1
        current_subtopic_name: Optional[str] = None
        current_objectives: List[str] = []
        current_keywords: List[str] = []

        def flush_subtopic() -> None:
            nonlocal current_subtopic_name, current_objectives, current_keywords
            if current_subtopic_name is None:
                return
            subtopics.append(
                Subtopic(
                    name=current_subtopic_name,
                    objectives=tuple(current_objectives),
                    keywords=tuple(current_keywords),
                )
            )
            current_subtopic_name = None
            current_objectives = []
            current_keywords = []

        while i < len(lines):
            line = lines[i].strip()
            if line == "---":
                i += 1
                break

            if ":" not in line:
                raise CurriculumFormatError(f"Malformed line {i+1}: {lines[i]!r}")

            key, val = [x.strip() for x in line.split(":", 1)]

            if key == "TOPIC":
                raise CurriculumFormatError(f"Nested TOPIC at line {i+1}. Use '---' separators.")

            if key == "DESCRIPTION":
                description = val
                i += 1
                continue

            if key in {"DIFFICULTIES", "FORBIDDEN"}:
                lst, i = _parse_list(lines, i + 1)
                if key == "DIFFICULTIES":
                    difficulties.extend(lst)
                else:
                    forbidden.extend(lst)
                continue

            if key == "SUBTOPIC":
                flush_subtopic()
                current_subtopic_name = val
                i += 1
                continue

            if key in {"OBJECTIVES", "KEYWORDS"}:
                if current_subtopic_name is None:
                    raise CurriculumFormatError(f"{key} before SUBTOPIC at line {i+1}")
                lst, i = _parse_list(lines, i + 1)
                if key == "OBJECTIVES":
                    current_objectives.extend(lst)
                else:
                    current_keywords.extend(lst)
                continue

            raise CurriculumFormatError(f"Unknown key {key!r} at line {i+1}")

        flush_subtopic()

        if not topic_name:
            raise CurriculumFormatError("Empty TOPIC name.")
        if not description:
            description = "(no description)"
        if not difficulties:
            raise CurriculumFormatError(f"TOPIC {topic_name!r} missing DIFFICULTIES.")

        topics.append(
            Topic(
                name=topic_name,
                description=description,
                difficulties=tuple(difficulties),
                forbidden=tuple(forbidden),
                subtopics=tuple(subtopics),
            )
        )

    return Curriculum(topics=tuple(topics), forbidden_global=tuple(forbidden_global))


def objectives_for_bucket(curriculum: Curriculum, topic: str, difficulty: str) -> Tuple[str, ...]:
    _ = difficulty  # currently topic-level; reserved for future difficulty-specific curriculum.
    return curriculum.get_topic(topic).all_objectives()


def forbidden_for_bucket(curriculum: Curriculum, topic: str, difficulty: str) -> Tuple[str, ...]:
    _ = difficulty  # reserved for future.
    t = curriculum.get_topic(topic)
    return tuple(list(curriculum.forbidden_global) + list(t.forbidden))


def keywords_for_bucket(curriculum: Curriculum, topic: str, difficulty: str) -> Tuple[str, ...]:
    _ = difficulty  # reserved for future.
    return curriculum.get_topic(topic).all_keywords()
