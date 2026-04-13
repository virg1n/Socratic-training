from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class HardExample:
    topic: str
    difficulty: str
    task: Dict[str, Any]
    socratic_hints: List[str]
    judge: Dict[str, Any]
    best_reward: float
    created_at: str


class HardExampleBuffer:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, ex: HardExample) -> None:
        rec = {
            "topic": ex.topic,
            "difficulty": ex.difficulty,
            "task": ex.task,
            "socratic_hints": ex.socratic_hints,
            "judge": ex.judge,
            "best_reward": ex.best_reward,
            "created_at": ex.created_at,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def add(
        self,
        *,
        topic: str,
        difficulty: str,
        task: Dict[str, Any],
        socratic_hints: List[str],
        judge: Dict[str, Any],
        best_reward: float,
    ) -> None:
        self.append(
            HardExample(
                topic=topic,
                difficulty=difficulty,
                task=task,
                socratic_hints=socratic_hints,
                judge=judge,
                best_reward=float(best_reward),
                created_at=_utc_now_iso(),
            )
        )

    def iter(self) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return iter(())
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue

    def to_sft_records(self) -> List[Dict[str, str]]:
        """
        Converts hard examples to (prompt, response) pairs for Red SFT.
        This is intentionally minimal: training scripts can enrich later.
        """
        out: List[Dict[str, str]] = []
        for r in self.iter():
            task = r.get("task", {})
            if not isinstance(task, dict):
                continue
            out.append(
                {
                    "prompt": f"Generate a task for topic={r.get('topic')} difficulty={r.get('difficulty')}",
                    "response": json.dumps(task, ensure_ascii=False),
                }
            )
        return out

