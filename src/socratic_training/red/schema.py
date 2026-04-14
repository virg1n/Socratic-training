from __future__ import annotations

import codecs
import json
import re
from typing import Any, Optional

try:  # pydantic v2 provides a v1 compat layer
    from pydantic.v1 import BaseModel, Field, validator
except Exception:  # pragma: no cover
    from pydantic import BaseModel, Field, validator  # type: ignore[no-redef]

from socratic_training.utils.code import strip_python_comments


JsonValue = Any


class RedTask(BaseModel):
    # Curriculum alignment
    topic: str
    difficulty: str
    statement: str
    # Full (buggy) module code, including assert-based tests that are executed.
    code: str

    @validator("topic", "difficulty", pre=True)
    def _strip_fields(cls, v: str) -> str:
        return str(v).strip().lower()

    @validator("code", pre=True)
    def _strip_code_fences(cls, v: str) -> str:
        s = str(v).strip()
        if s.startswith("```"):
            lines = s.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()
        s = s.replace("\r\n", "\n")

        # Some models double-escape code strings inside JSON. That yields Python like:
        #   def f(x):\n    return x
        # which causes: "unexpected character after line continuation character".
        # Heuristic: if it looks like source but contains backslash-escapes, decode once.
        looks_like_code = bool(re.search(r"^\s*(def|class|import|from)\b", s))
        has_escapes = any(tok in s for tok in ("\\n", "\\t", "\\r", '\\"', "\\'"))
        if looks_like_code and has_escapes:
            # If the whole field is itself a quoted JSON string literal, decode it.
            if (s.startswith('"') and s.endswith('"')) and len(s) >= 2:
                try:
                    inner = json.loads(s)
                    if isinstance(inner, str) and re.search(r"^\s*(def|class)\s+", inner):
                        s = inner
                except Exception:
                    pass

            try:
                cand = codecs.decode(s, "unicode_escape")
                if re.search(r"^\s*(def|class)\s+", cand) and cand.count("\\") < s.count("\\"):
                    s = cand
            except Exception:
                pass

        # User request: strip ALL comments from Red code.
        s = strip_python_comments(s)

        return s

    @validator("statement")
    def _statement_not_empty(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 40:
            raise ValueError("statement too short")
        return v

    @validator("code")
    def _code_not_empty(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 40:
            raise ValueError("code too short")
        return v
