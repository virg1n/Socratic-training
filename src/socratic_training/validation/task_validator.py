from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from socratic_training.config import AppConfig
from socratic_training.curriculum import Curriculum, forbidden_for_bucket
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


def _count_asserts(tree: ast.AST) -> int:
    return sum(1 for n in ast.walk(tree) if isinstance(n, ast.Assert))


def _extract_first_file_lineno(stderr: str) -> Optional[int]:
    """
    Best-effort parse of a Python traceback line number:
      File ".../submission.py", line 123, in ...
    """
    m = re.search(r'File ".*submission\\.py", line (\\d+)', stderr)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


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

    # Curriculum alignment
    try:
        topic_obj = curriculum.get_topic(task.topic)
    except KeyError:
        reasons.append("unknown_topic")
        return TaskValidation(ok=False, reasons=tuple(reasons))

    if task.difficulty not in topic_obj.difficulties:
        reasons.append("difficulty_not_allowed")

    forbidden = set(forbidden_for_bucket(curriculum, task.topic, task.difficulty))
    # Quick keyword-based scope check.
    lowered = (task.statement + "\n" + task.code).lower()
    for f in forbidden:
        if f.lower() in lowered:
            reasons.append(f"mentions_forbidden:{f}")
            break

    fp = _fingerprint(task.statement)
    if seen_fingerprints is not None:
        if fp in seen_fingerprints:
            reasons.append("duplicate_statement")
        else:
            seen_fingerprints.add(fp)

    # Safety checks on code
    try:
        tree = ast.parse(task.code)
    except SyntaxError as e:
        reasons.append(f"syntax_error: {e.msg}")
        return TaskValidation(ok=False, reasons=tuple(reasons), fingerprint=fp)

    reasons += _ast_safety_check(task.code)

    # Basic non-triviality: require assert-based tests embedded in code.
    num_asserts = _count_asserts(tree)
    if num_asserts < int(cfg.validation.min_tests):
        reasons.append("too_few_asserts")

    # If already bad, skip execution.
    if reasons:
        return TaskValidation(ok=False, reasons=tuple(reasons), fingerprint=fp)

    # Execute code and capture interpreter error output.
    exec_res = run_code_in_subprocess(code=task.code, timeout_s=cfg.validation.python_timeout_s)
    observed = _extract_observed_failure(exec_res.stderr, exec_res.stdout)

    if exec_res.ok:
        reasons.append("code_passes_all_tests")
        return TaskValidation(ok=False, reasons=tuple(reasons), exec=exec_res, fingerprint=fp, observed_failure=observed)

    # Reject trivial failures: syntax/import problems rather than failing asserts.
    err = (exec_res.stderr or exec_res.stdout or "").lower()
    if any(x in err for x in ("syntaxerror", "indentationerror")):
        reasons.append("code_syntax_error_at_runtime")
    if any(x in err for x in ("modulenotfounderror", "importerror")):
        reasons.append("code_import_error")

    # Prefer assert-driven failures (matches the target dataset style).
    if "assertionerror" not in err:
        reasons.append("non_assert_failure")

    if cfg.validation.require_buggy_passes_some:
        fail_line = _extract_first_file_lineno(exec_res.stderr or exec_res.stdout or "")
        if fail_line is not None:
            assert_lines = sorted([n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert) and hasattr(n, "lineno")])
            asserts_before = sum(1 for ln in assert_lines if int(ln) < int(fail_line))
            if asserts_before <= 0:
                reasons.append("fails_before_any_assert")

    if reasons:
        return TaskValidation(ok=False, reasons=tuple(reasons), exec=exec_res, fingerprint=fp, observed_failure=observed)

    return TaskValidation(ok=True, reasons=tuple(), exec=exec_res, fingerprint=fp, observed_failure=observed)
