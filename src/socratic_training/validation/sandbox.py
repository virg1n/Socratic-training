from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Dict, List, Optional


@dataclass
class SandboxResult:
    ok: bool
    passed: int
    failed: int
    stdout: str
    stderr: str
    details: Dict[str, Any]
    wall_s: float


@dataclass
class ExecResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    wall_s: float


def run_tests_in_subprocess(
    *,
    code: str,
    function_name: str,
    tests: List[Dict[str, Any]],
    timeout_s: float,
    python_executable: Optional[str] = None,
) -> SandboxResult:
    """
    Executes candidate code in a temporary directory with timeouts and stdout/stderr capture.

    Security notes:
    - This is *not* a hardened sandbox. It is a best-effort "safety harness" intended
      to catch accidental bad code from Red. Run on an isolated training box.
    """
    py = python_executable or sys.executable
    start = monotonic()

    runner = textwrap.dedent(
        f"""
        import importlib.util
        import json
        import os
        import sys
        import traceback
        from pathlib import Path

        def _limit_resources():
            try:
                import resource
            except Exception:
                return
            # CPU seconds
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
            except Exception:
                pass
            # Address space (bytes) - best effort
            try:
                mem = 512 * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            except Exception:
                pass
            # File size
            try:
                resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
            except Exception:
                pass

        _limit_resources()

        tests = json.loads(Path("tests.json").read_text(encoding="utf-8"))
        fn_name = {function_name!r}

        spec = importlib.util.spec_from_file_location("submission", "submission.py")
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
        except Exception:
            print(json.dumps({{"error": "import_error", "traceback": traceback.format_exc()}}))
            raise

        if not hasattr(mod, fn_name):
            print(json.dumps({{"error": "missing_function", "function_name": fn_name}}))
            raise SystemExit(2)

        fn = getattr(mod, fn_name)

        passed = 0
        failed = 0
        failures = []
        for idx, t in enumerate(tests):
            args = t.get("args", [])
            kwargs = t.get("kwargs", {{}})
            expected = t.get("expected", None)
            exp_exc = t.get("raises", None)
            try:
                got = fn(*args, **kwargs)
                if exp_exc is not None:
                    failed += 1
                    failures.append({{"i": idx, "kind": "expected_exception", "expected": exp_exc}})
                    continue
                if got == expected:
                    passed += 1
                else:
                    failed += 1
                    failures.append({{"i": idx, "kind": "wrong_answer", "expected": expected, "got": got}})
            except Exception as e:
                if exp_exc is None:
                    failed += 1
                    failures.append({{"i": idx, "kind": "raised", "exc_type": type(e).__name__, "msg": str(e)}})
                else:
                    if type(e).__name__ == exp_exc:
                        passed += 1
                    else:
                        failed += 1
                        failures.append(
                            {{"i": idx, "kind": "wrong_exception", "expected": exp_exc, "got": type(e).__name__}}
                        )

        ok = failed == 0
        print(json.dumps({{"ok": ok, "passed": passed, "failed": failed, "failures": failures}}))
        raise SystemExit(0 if ok else 1)
        """
    ).strip()

    with tempfile.TemporaryDirectory(prefix="socratic_sandbox_") as td:
        tdp = Path(td)
        (tdp / "submission.py").write_text(code, encoding="utf-8")
        (tdp / "tests.json").write_text(json.dumps(tests), encoding="utf-8")
        (tdp / "runner.py").write_text(runner, encoding="utf-8")

        env = os.environ.copy()
        env.update(
            {
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
        )

        cmd = [py, "-I", "-S", str(tdp / "runner.py")]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(tdp),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            wall = monotonic() - start
            return SandboxResult(
                ok=False,
                passed=0,
                failed=len(tests),
                stdout=e.stdout or "",
                stderr=(e.stderr or "") + "\nTIMEOUT",
                details={"error": "timeout"},
                wall_s=wall,
            )

        wall = monotonic() - start
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        details: Dict[str, Any] = {}
        try:
            details = json.loads(stdout.splitlines()[-1]) if stdout else {}
        except Exception:
            details = {"raw_stdout": stdout}

        return SandboxResult(
            ok=bool(details.get("ok", False)) and proc.returncode == 0,
            passed=int(details.get("passed", 0)) if isinstance(details, dict) else 0,
            failed=int(details.get("failed", len(tests))) if isinstance(details, dict) else len(tests),
            stdout=stdout,
            stderr=stderr,
            details=details if isinstance(details, dict) else {"details": details},
            wall_s=wall,
        )


def run_code_in_subprocess(
    *,
    code: str,
    timeout_s: float,
    python_executable: Optional[str] = None,
) -> ExecResult:
    """
    Executes a Python module as a subprocess and captures stdout/stderr.

    This is used for Red-generated "code with asserts" tasks where the code itself
    contains tests and should crash with a Python traceback.
    """
    py = python_executable or sys.executable
    start = monotonic()

    with tempfile.TemporaryDirectory(prefix="socratic_sandbox_") as td:
        tdp = Path(td)
        (tdp / "submission.py").write_text(code, encoding="utf-8")

        env = os.environ.copy()
        env.update(
            {
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
        )

        cmd = [py, "-I", "-S", str(tdp / "submission.py")]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(tdp),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            wall = monotonic() - start
            return ExecResult(
                ok=False,
                returncode=124,
                stdout=e.stdout or "",
                stderr=(e.stderr or "") + "\nTIMEOUT",
                wall_s=wall,
            )

        wall = monotonic() - start
        return ExecResult(
            ok=proc.returncode == 0,
            returncode=int(proc.returncode),
            stdout=(proc.stdout or "").strip(),
            stderr=(proc.stderr or "").strip(),
            wall_s=wall,
        )
