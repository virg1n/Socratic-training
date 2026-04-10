from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CurriculumSubtopic:
    name: str
    difficulties: list[str] = field(default_factory=list)
    objectives: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CurriculumTopic:
    name: str
    description: str = ""
    difficulties: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    subtopics: list[CurriculumSubtopic] = field(default_factory=list)


@dataclass(slots=True)
class CurriculumBucket:
    topic: str
    subtopic: str
    description: str
    difficulties: list[str]
    objectives: list[str]
    keywords: list[str]
    forbidden: list[str]

    @property
    def bucket_id(self) -> str:
        return f"{self.topic}/{self.subtopic}"


@dataclass(slots=True)
class CurriculumCatalog:
    topics: dict[str, CurriculumTopic]
    source_path: Path

    def list_buckets(self) -> list[CurriculumBucket]:
        buckets: list[CurriculumBucket] = []
        for topic in self.topics.values():
            for subtopic in topic.subtopics:
                buckets.append(
                    CurriculumBucket(
                        topic=topic.name,
                        subtopic=subtopic.name,
                        description=topic.description,
                        difficulties=subtopic.difficulties or topic.difficulties,
                        objectives=subtopic.objectives,
                        keywords=subtopic.keywords,
                        forbidden=sorted({*topic.forbidden, *subtopic.forbidden}),
                    )
                )
        return buckets

    def get_bucket(self, topic_name: str, subtopic_name: str) -> CurriculumBucket:
        for bucket in self.list_buckets():
            if bucket.topic == topic_name and bucket.subtopic == subtopic_name:
                return bucket
        raise KeyError(f"Unknown curriculum bucket: {topic_name}/{subtopic_name}")

    def to_dict(self) -> dict[str, Any]:
        return {"source_path": str(self.source_path), "topics": {name: asdict(topic) for name, topic in self.topics.items()}}


@dataclass(slots=True)
class TaskTest:
    name: str
    code: str


@dataclass(slots=True)
class TaskExample:
    topic: str
    difficulty: str
    statement: str
    canonical_solution: str
    buggy_solution: str
    tests: list[TaskTest]
    expected_learning_objectives: list[str]
    bug_explanation: str = ""
    entrypoint: str = "solve"
    source_model: str = "red"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tests"] = [asdict(test) for test in self.tests]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskExample":
        tests = [TaskTest(**test) for test in payload.get("tests", [])]
        return cls(
            topic=payload["topic"],
            difficulty=payload["difficulty"],
            statement=payload["statement"],
            canonical_solution=payload["canonical_solution"],
            buggy_solution=payload.get("buggy_solution", ""),
            bug_explanation=payload.get("bug_explanation", ""),
            tests=tests,
            expected_learning_objectives=list(payload.get("expected_learning_objectives", [])),
            entrypoint=payload.get("entrypoint", "solve"),
            source_model=payload.get("source_model", "red"),
        )


@dataclass(slots=True)
class ExecutionResult:
    passed: int
    failed: int
    stdout: str
    stderr: str
    failures: list[str]

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


@dataclass(slots=True)
class TaskValidationResult:
    task: TaskExample
    accepted: bool
    reasons: list[str]
    canonical_execution: ExecutionResult | None = None
    buggy_execution: ExecutionResult | None = None


@dataclass(slots=True)
class HintCandidate:
    task: TaskExample
    prompt: str
    hint: str
    index: int


@dataclass(slots=True)
class JudgeSubscores:
    did_not_give_final_answer: float
    pedagogical_value: float
    bug_localization_help: float
    curriculum_alignment: float
    beginner_friendliness: float
    actionability_of_hint: float

    def weighted_sum(self, weights: dict[str, float]) -> float:
        return (
            self.did_not_give_final_answer * weights["did_not_give_final_answer"]
            + self.pedagogical_value * weights["pedagogical_value"]
            + self.bug_localization_help * weights["bug_localization_help"]
            + self.curriculum_alignment * weights["curriculum_alignment"]
            + self.beginner_friendliness * weights["beginner_friendliness"]
            + self.actionability_of_hint * weights["actionability_of_hint"]
        )


@dataclass(slots=True)
class JudgeEvaluation:
    candidate: HintCandidate
    subscores: JudgeSubscores
    rationale: str
    disclosure_penalty: float
    final_reward: float
    rank: int = 0


@dataclass(slots=True)
class RolloutRecord:
    candidate: HintCandidate
    reward: float
    group_advantage: float


@dataclass(slots=True)
class HardExample:
    task: TaskExample
    best_reward: float
    worst_reward: float
    bucket_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "best_reward": self.best_reward,
            "worst_reward": self.worst_reward,
            "bucket_id": self.bucket_id,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ScenarioEstimate:
    name: str
    gpu_gb: float
    cpu_gb: float
    fits_gpu: bool
    fits_cpu: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return self.fits_gpu and self.fits_cpu
