from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import HardExample
from .utils import append_jsonl


class HardExampleBuffer:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, example: HardExample) -> None:
        append_jsonl(self.path, example.to_dict())

    def load(self) -> list[HardExample]:
        if not self.path.exists():
            return []
        records: list[HardExample] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            from .types import TaskExample

            records.append(
                HardExample(
                    task=TaskExample.from_dict(payload["task"]),
                    best_reward=payload["best_reward"],
                    worst_reward=payload["worst_reward"],
                    bucket_id=payload["bucket_id"],
                    metadata=payload.get("metadata", {}),
                )
            )
        return records


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.path, payload)
