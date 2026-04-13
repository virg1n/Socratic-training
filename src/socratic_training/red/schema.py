from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # pydantic v2 provides a v1 compat layer
    from pydantic.v1 import BaseModel, Field, validator
except Exception:  # pragma: no cover
    from pydantic import BaseModel, Field, validator  # type: ignore[no-redef]


JsonValue = Any


class TestCase(BaseModel):
    args: List[JsonValue] = Field(default_factory=list)
    kwargs: Dict[str, JsonValue] = Field(default_factory=dict)
    expected: Optional[JsonValue] = None
    raises: Optional[str] = Field(
        default=None,
        description="Optional exception class name (e.g. ValueError) expected to be raised.",
    )

    @validator("raises", pre=True, always=True)
    def _empty_to_none(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None


class RedTask(BaseModel):
    # Curriculum alignment
    topic: str
    difficulty: str
    expected_learning_objectives: List[str]

    # Problem
    statement: str
    function_name: str = "solve"

    # Solutions
    canonical_solution: str
    buggy_solution: str
    bug_explanation: Optional[str] = None

    # Tests
    tests: List[TestCase]

    @validator("statement")
    def _statement_not_empty(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 40:
            raise ValueError("statement too short")
        return v

    @validator("function_name")
    def _valid_function_name(cls, v: str) -> str:
        v = v.strip()
        if not v.isidentifier():
            raise ValueError("function_name must be a valid identifier")
        return v

    @validator("tests")
    def _tests_nonempty(cls, v: List[TestCase]) -> List[TestCase]:
        if len(v) < 3:
            raise ValueError("need at least 3 tests")
        return v
