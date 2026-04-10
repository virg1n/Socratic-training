from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import asdict
from typing import Iterable

from .config import PipelineConfig
from .types import CurriculumBucket, ExecutionResult, TaskExample, TaskValidationResult
from .utils import jaccard_similarity, normalize_text, token_set


FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "input", "breakpoint", "__import__"}


class UnsafeCodeVisitor(ast.NodeVisitor):
    def __init__(self, allowed_imports: set[str]) -> None:
        self.allowed_imports = allowed_imports
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module = alias.name.split(".", 1)[0]
            if module not in self.allowed_imports:
                self.errors.append(f"Import '{alias.name}' is not allowed.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = (node.module or "").split(".", 1)[0]
        if module not in self.allowed_imports:
            self.errors.append(f"Import '{node.module}' is not allowed.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            self.errors.append(f"Call to '{node.func.id}' is not allowed.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.errors.append("Dunder attribute access is not allowed.")
        self.generic_visit(node)


class TaskValidator:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def validate(self, task: TaskExample, bucket: CurriculumBucket, seen_tasks: Iterable[TaskExample]) -> TaskValidationResult:
        reasons: list[str] = []
        self._validate_schema(task, reasons)
        self._validate_lengths(task, reasons)
        self._validate_curriculum(task, bucket, reasons)
        self._validate_uniqueness(task, seen_tasks, reasons)
        self._validate_code(task, reasons)

        canonical_execution = None
        buggy_execution = None
        if not reasons:
            canonical_execution = self._execute_candidate(task.canonical_solution, task.tests, task.entrypoint)
            buggy_execution = self._execute_candidate(task.buggy_solution, task.tests, task.entrypoint)
            if not canonical_execution.all_passed:
                reasons.append("Canonical solution does not pass all tests.")
            if buggy_execution.failed == 0:
                reasons.append("Buggy solution passes every test.")
            if buggy_execution.passed == 0:
                reasons.append("Buggy solution fails every test; reject as pathological or non-pedagogical.")
            if normalize_text(task.buggy_solution) == normalize_text(task.canonical_solution):
                reasons.append("Buggy solution is identical to canonical solution.")
        return TaskValidationResult(task=task, accepted=not reasons, reasons=reasons, canonical_execution=canonical_execution, buggy_execution=buggy_execution)

    def _validate_schema(self, task: TaskExample, reasons: list[str]) -> None:
        required_text_fields = {
            "topic": task.topic,
            "difficulty": task.difficulty,
            "statement": task.statement,
            "canonical_solution": task.canonical_solution,
            "buggy_solution": task.buggy_solution,
        }
        for field_name, value in required_text_fields.items():
            if not value.strip():
                reasons.append(f"Missing required field: {field_name}.")
        if not task.tests:
            reasons.append("Task must include tests.")
        if not task.expected_learning_objectives:
            reasons.append("Task must include expected learning objectives.")

    def _validate_lengths(self, task: TaskExample, reasons: list[str]) -> None:
        statement_length = len(task.statement.strip())
        if statement_length < self.config.validation.min_statement_chars or statement_length > self.config.validation.max_statement_chars:
            reasons.append("Statement length is outside the accepted range.")
        if len(task.canonical_solution.splitlines()) > self.config.validation.max_solution_lines:
            reasons.append("Canonical solution is too long for the configured curriculum band.")
        if len(task.buggy_solution.splitlines()) > self.config.validation.max_solution_lines:
            reasons.append("Buggy solution is too long for the configured curriculum band.")
        if len(task.tests) < 3:
            reasons.append("At least 3 tests are required.")

    def _validate_curriculum(self, task: TaskExample, bucket: CurriculumBucket, reasons: list[str]) -> None:
        bucket_tokens = token_set(" ".join([bucket.topic, bucket.subtopic, *bucket.keywords, *bucket.objectives]))
        task_tokens = token_set(" ".join([task.topic, task.statement, *task.expected_learning_objectives]))
        overlap = len(bucket_tokens & task_tokens) / max(1, len(bucket_tokens))
        if overlap < 0.10:
            reasons.append("Task does not match the selected curriculum bucket.")
        forbidden_hits = [item for item in bucket.forbidden if normalize_text(item) in normalize_text(task.statement + " " + task.canonical_solution)]
        if forbidden_hits:
            reasons.append(f"Task uses forbidden content: {', '.join(forbidden_hits)}.")
        normalized_objectives = {normalize_text(objective) for objective in bucket.objectives}
        objective_hits = sum(1 for objective in task.expected_learning_objectives if normalize_text(objective) in normalized_objectives)
        if objective_hits == 0:
            reasons.append("Task objectives do not map to the curriculum bucket.")

    def _validate_uniqueness(self, task: TaskExample, seen_tasks: Iterable[TaskExample], reasons: list[str]) -> None:
        for seen_task in seen_tasks:
            if jaccard_similarity(task.statement, seen_task.statement) >= self.config.validation.similarity_threshold:
                reasons.append("Task is too similar to a previously accepted task.")
                return

    def _validate_code(self, task: TaskExample, reasons: list[str]) -> None:
        for label, source in {"canonical": task.canonical_solution, "buggy": task.buggy_solution}.items():
            try:
                tree = ast.parse(source)
            except SyntaxError as error:
                reasons.append(f"{label.title()} solution has syntax error: {error.msg}.")
                continue
            if not any(isinstance(node, ast.FunctionDef) and node.name == task.entrypoint for node in tree.body):
                reasons.append(f"{label.title()} solution must define `{task.entrypoint}`.")
            visitor = UnsafeCodeVisitor(set(self.config.validation.allowed_imports))
            visitor.visit(tree)
            if visitor.errors:
                reasons.extend(f"{label.title()} solution unsafe: {message}" for message in visitor.errors)

    def _execute_candidate(self, source: str, tests, entrypoint: str) -> ExecutionResult:
        runner = _sandbox_runner_source()
        payload = {
            "source": source,
            "tests": [asdict(test) for test in tests],
            "entrypoint": entrypoint,
            "timeout_seconds": self.config.validation.timeout_seconds,
            "memory_limit_mb": self.config.validation.memory_limit_mb,
            "allowed_imports": self.config.validation.allowed_imports,
        }
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", runner],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.config.validation.timeout_seconds + 1,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(passed=0, failed=len(tests), stdout="", stderr="Timed out.", failures=["Execution timed out."])
        if completed.returncode != 0:
            return ExecutionResult(passed=0, failed=len(tests), stdout=completed.stdout, stderr=completed.stderr, failures=[completed.stderr.strip() or "Sandbox failed."])
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return ExecutionResult(passed=0, failed=len(tests), stdout=completed.stdout, stderr=completed.stderr, failures=["Sandbox returned invalid JSON."])
        return ExecutionResult(
            passed=result["passed"],
            failed=result["failed"],
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            failures=result.get("failures", []),
        )


def _sandbox_runner_source() -> str:
    return r"""
import contextlib
import importlib
import io
import json
import sys
import traceback

try:
    import resource
except ImportError:
    resource = None

SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

payload = json.loads(sys.stdin.read())
allowed_imports = set(payload["allowed_imports"])

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = name.split(".", 1)[0]
    if module not in allowed_imports:
        raise ImportError(f"Import '{name}' is blocked.")
    return importlib.import_module(name)

SAFE_BUILTINS["__import__"] = safe_import

if resource is not None:
    memory_bytes = int(payload["memory_limit_mb"]) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    cpu_seconds = int(payload["timeout_seconds"])
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))

stdout_buffer = io.StringIO()
stderr_buffer = io.StringIO()
namespace = {"__builtins__": SAFE_BUILTINS}
failures = []
passed = 0

try:
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        exec(compile(payload["source"], "<candidate>", "exec"), namespace, namespace)
        target = namespace.get(payload["entrypoint"])
        if not callable(target):
            raise RuntimeError(f"Entrypoint {payload['entrypoint']} is not callable.")
        for test in payload["tests"]:
            try:
                exec(compile(test["code"], f"<{test['name']}>", "exec"), namespace, namespace)
                passed += 1
            except Exception:
                failures.append(f"{test['name']}: {traceback.format_exc(limit=1).strip()}")
except Exception:
    failures.append(traceback.format_exc(limit=2).strip())

result = {
    "passed": passed,
    "failed": len(payload["tests"]) - passed,
    "failures": failures,
    "stdout": stdout_buffer.getvalue(),
    "stderr": stderr_buffer.getvalue(),
}
print(json.dumps(result))
"""
