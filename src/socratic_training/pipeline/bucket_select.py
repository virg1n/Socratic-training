from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from socratic_training.curriculum import Curriculum


def _norm(s: str) -> str:
    return str(s).strip().lower()


def _parse_slash_choices(spec: str) -> Optional[Tuple[str, ...]]:
    """
    Parses CLI choice specs:
      - "random" -> None
      - "a/b/c"  -> ("a","b","c")
      - "loops"  -> ("loops",)
    """
    s = _norm(spec)
    if s == "random":
        return None
    parts = [p.strip() for p in str(spec).split("/") if p.strip()]
    if not parts:
        raise ValueError("Empty choice spec.")
    return tuple(_norm(p) for p in parts)


@dataclass(frozen=True)
class BucketChoice:
    topic: str
    difficulty: str


def choose_bucket(
    curriculum: Curriculum,
    *,
    topic_spec: str,
    difficulty_spec: str,
    rng: random.Random,
) -> BucketChoice:
    """
    Chooses a (topic, difficulty) bucket given CLI specs and a curriculum.

    Rules:
    - topic_spec can be "random" or "a/b/c" to sample topics.
    - difficulty_spec can be "random" or "medium/hard" to sample among allowed difficulties.
    - If the chosen topic doesn't allow any requested difficulty, we retry with another topic
      *only if* topic_spec is not a single fixed topic.
    """
    if not curriculum.topics:
        raise ValueError("Curriculum has no topics.")

    topic_choices = _parse_slash_choices(topic_spec)
    diff_choices = _parse_slash_choices(difficulty_spec)

    # Map normalized topic -> canonical topic name from curriculum.
    topic_norm_to_name: Dict[str, str] = {_norm(t.name): t.name for t in curriculum.topics}

    if topic_choices is None:
        topic_pool = [t.name for t in curriculum.topics]
    else:
        missing = [t for t in topic_choices if t not in topic_norm_to_name]
        if missing:
            raise ValueError(f"Unknown topic(s): {missing}. Known: {sorted(topic_norm_to_name.keys())}")
        topic_pool = [topic_norm_to_name[t] for t in topic_choices]

    # Compute eligible buckets deterministically so we never "randomly fail" to find an intersection.
    eligible: list[tuple[str, tuple[str, ...]]] = []
    for topic in topic_pool:
        topic_obj = curriculum.get_topic(topic)
        allowed_diffs = tuple(topic_obj.difficulties)
        if not allowed_diffs:
            continue

        if diff_choices is None:
            eligible.append((topic, allowed_diffs))
            continue

        allowed_norm_to_name = {_norm(d): d for d in allowed_diffs}
        allowed = tuple(allowed_norm_to_name[d] for d in diff_choices if d in allowed_norm_to_name)
        if allowed:
            eligible.append((topic, allowed))

    if not eligible:
        desired = "random" if diff_choices is None else "/".join(diff_choices)
        if topic_choices is None:
            topic_desc = "random"
        else:
            topic_desc = "/".join(topic_choices)

        if topic_choices is not None and len(topic_pool) == 1 and diff_choices is not None:
            topic = topic_pool[0]
            allowed = list(curriculum.get_topic(topic).difficulties)
            raise ValueError(
                f"Topic {topic!r} does not allow any of requested difficulties {list(diff_choices)!r}. "
                f"Allowed: {allowed!r}"
            )

        raise ValueError(f"Could not find any bucket matching topic={topic_desc!r} difficulty={desired!r}.")

    topic, diffs = rng.choice(eligible)
    return BucketChoice(topic=topic, difficulty=rng.choice(list(diffs)))
