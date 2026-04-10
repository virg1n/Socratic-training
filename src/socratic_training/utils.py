from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


LOGGER = logging.getLogger("socratic_training")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def slugify(value: str) -> str:
    compact = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return compact or "item"


def normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"`+", " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def token_set(value: str) -> set[str]:
    return {token for token in normalize_text(value).split() if token}


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def extract_first_json(value: str) -> Any:
    decoder = json.JSONDecoder()
    for index, character in enumerate(value):
        if character not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(value[index:])
            return payload
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON object or array found in model output.")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = asdict(payload) if is_dataclass(payload) else payload
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    mean_value = safe_mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) or 1.0
