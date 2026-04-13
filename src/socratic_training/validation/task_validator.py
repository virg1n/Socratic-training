from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from socratic_training.config import AppConfig
from socratic_training.curriculum import Curriculum, forbidden_for_bucket, objectives_for_bucket
from socratic_training.red.schema import RedTask
from socratic_training.validation.sandbox import SandboxResult, run_tests_in_subprocess


@dataclass
class TaskValidation:
    ok: bool
    reasons: Tuple[str, ...]
    canonical: Optional[SandboxResult] = None
    buggy: Optional[SandboxResult] = None
    fingerprint: Optional[str] = None


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


def _ensure_function_defined(code: str, function_name: str) -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax_error: {e.msg}"]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return []
    return [f"missing_function_def:{function_name}"]


def validate_red_task(
    cfg: AppConfig,
    *,
    curriculum: Curriculum,
    task: RedTask,
    seen_fingerprints: Optional[Set[str]] = None,
) -> TaskValidation:
    reasons: List[str] = []

    if task.canonical_solution.strip() == task.buggy_solution.strip():
        reasons.append("buggy_identical_to_canonical")

    # Curriculum alignment
    try:
        topic_obj = curriculum.get_topic(task.topic)
    except KeyError:
        reasons.append("unknown_topic")
        return TaskValidation(ok=False, reasons=tuple(reasons))

    if task.difficulty not in topic_obj.difficulties:
        reasons.append("difficulty_not_allowed")

    allowed_obj = list(objectives_for_bucket(curriculum, task.topic, task.difficulty))
    if task.expected_learning_objectives:
        def _norm(s: str) -> str:
            return re.sub(r"\s+", " ", s.strip().lower())

        allowed_norm = {_norm(x) for x in allowed_obj}
        for obj in task.expected_learning_objectives:
            if _norm(obj) not in allowed_norm:
                reasons.append("objectives_out_of_bucket")
                break

    forbidden = set(forbidden_for_bucket(curriculum, task.topic, task.difficulty))
    # Quick keyword-based scope check.
    lowered = (task.statement + "\n" + task.canonical_solution + "\n" + task.buggy_solution).lower()
    for f in forbidden:
        if f.lower() in lowered:
            reasons.append(f"mentions_forbidden:{f}")
            break

    # Basic non-triviality
    if len(task.tests) < cfg.validation.min_tests:
        reasons.append("too_few_tests")

    fp = _fingerprint(task.statement)
    if seen_fingerprints is not None:
        if fp in seen_fingerprints:
            reasons.append("duplicate_statement")
        else:
            seen_fingerprints.add(fp)

    # Safety checks on code
    reasons += _ast_safety_check(task.canonical_solution)
    reasons += _ast_safety_check(task.buggy_solution)
    reasons += _ensure_function_defined(task.canonical_solution, task.function_name)
    reasons += _ensure_function_defined(task.buggy_solution, task.function_name)

    # If already bad, skip execution.
    if reasons:
        return TaskValidation(ok=False, reasons=tuple(reasons), fingerprint=fp)

    tests = [t.dict() for t in task.tests][: cfg.validation.max_tests]

    canonical = run_tests_in_subprocess(
        code=task.canonical_solution,
        function_name=task.function_name,
        tests=tests,
        timeout_s=cfg.validation.python_timeout_s,
    )
    if not canonical.ok:
        reasons.append("canonical_fails_tests")
        return TaskValidation(ok=False, reasons=tuple(reasons), canonical=canonical, fingerprint=fp)

    buggy = run_tests_in_subprocess(
        code=task.buggy_solution,
        function_name=task.function_name,
        tests=tests,
        timeout_s=cfg.validation.python_timeout_s,
    )
    if buggy.ok:
        reasons.append("buggy_passes_all_tests")
        return TaskValidation(ok=False, reasons=tuple(reasons), canonical=canonical, buggy=buggy, fingerprint=fp)

    if cfg.validation.require_buggy_passes_some and buggy.passed == 0:
        reasons.append("buggy_fails_all_tests")
        return TaskValidation(ok=False, reasons=tuple(reasons), canonical=canonical, buggy=buggy, fingerprint=fp)

    return TaskValidation(ok=True, reasons=tuple(), canonical=canonical, buggy=buggy, fingerprint=fp)
