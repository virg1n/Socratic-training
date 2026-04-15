from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from socratic_training.config import AppConfig
from socratic_training.curriculum import Curriculum
from socratic_training.red.schema import RedTask
from socratic_training.validation.sandbox import ExecResult, run_code_in_subprocess


@dataclass
class TaskValidation:
    ok: bool
    reasons: Tuple[str, ...]
    exec: Optional[ExecResult] = None
    fingerprint: Optional[str] = None
    observed_failure: Optional[str] = None


_BANNED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "pathlib",
    "importlib",
    "socket",
    "requests",
    "urllib",
    "http",
    "asyncio",
    "threading",
    "multiprocessing",
    "pickle",
}

_BANNED_NAMES = {
    "open",
    "eval",
    "exec",
    "__import__",
    "compile",
    "input",
}


def _fingerprint(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _ast_safety_check(code: str) -> List[str]:
    reasons: List[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax_error: {e.msg}"]

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                base = alias.name.split(".")[0]
                if base in _BANNED_IMPORTS:
                    reasons.append(f"banned_import:{base}")
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            reasons.append(f"banned_name:{node.id}")
        if isinstance(node, ast.Attribute) and node.attr in {"system", "popen", "walk", "remove"}:
            reasons.append(f"suspicious_attr:{node.attr}")
    return reasons


def _extract_observed_failure(stderr: str, stdout: str) -> str:
    text = (stderr or stdout or "").strip()
    if not text:
        return ""
    # Keep the tail (exception name/message) + a bit of context.
    lines = text.splitlines()
    tail = "\n".join(lines[-30:])
    return tail


def validate_red_task(
    cfg: AppConfig,
    *,
    curriculum: Curriculum,
    task: RedTask,
    seen_fingerprints: Optional[Set[str]] = None,
) -> TaskValidation:
    reasons: List[str] = []

    fp = _fingerprint(task.statement)

    # Safety checks on code
    # Accept syntax errors as "valid buggy tasks" (they'll fail on execution),
    # but only run AST-based safety checks when parsing succeeds.
    try:
        ast.parse(task.code)
    except SyntaxError:
        pass
    else:
        reasons += _ast_safety_check(task.code)

    # If already bad, skip execution.
    if reasons:
        return TaskValidation(ok=False, reasons=tuple(reasons), fingerprint=fp)

    # Execute code and capture interpreter error output.
    exec_res = run_code_in_subprocess(code=task.code, timeout_s=cfg.validation.python_timeout_s)
    observed = _extract_observed_failure(exec_res.stderr, exec_res.stdout)

    if exec_res.ok:
        reasons.append("code_passes_all_tests")
        return TaskValidation(ok=False, reasons=tuple(reasons), exec=exec_res, fingerprint=fp, observed_failure=observed)

    if reasons:
        return TaskValidation(ok=False, reasons=tuple(reasons), exec=exec_res, fingerprint=fp, observed_failure=observed)

    return TaskValidation(ok=True, reasons=tuple(), exec=exec_res, fingerprint=fp, observed_failure=observed)
