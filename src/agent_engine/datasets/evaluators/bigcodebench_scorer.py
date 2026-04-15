"""BigCodeBench test-harness runner.

Assembles the generated function implementation with the BigCodeBench test
harness and executes it in a subprocess to determine pass/fail.
"""

import os
import re
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from ...utils.logging import get_logger

logger = get_logger(__name__)

# When set, assembled test files are written here (workspace-relative) instead
# of /tmp so they are visible on shared filesystems (e.g. SLURM compute nodes).
# Set to None to use the system temp dir (files are deleted after each run).
DEBUG_EVAL_DIR: Optional[str] = None  # e.g. "./cache/bigcodebench_eval_debug"


def _strip_markdown_fences(text: str) -> str:
    """Remove ```python ... ``` or ``` ... ``` fences from text."""
    pattern = r"```(?:python)?\n?(.*?)\n?```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def evaluate_bigcodebench(
    generated_code: str,
    metadata: Dict[str, Any],
    timeout: int = 30,
) -> Dict[str, Any]:
    """Run the BigCodeBench test harness on generated_code.

    Args:
        generated_code: Model's predicted implementation (may include markdown fences).
        metadata: Must contain ``code_prompt``, ``test``, and ``entry_point``.
            The ``test`` field is a ``unittest.TestCase`` class that calls the
            function directly by name — no ``check()`` wrapper.
        timeout: Subprocess timeout in seconds.

    Returns:
        Dict with keys: correct (bool), score (float), task_id (str), error (str|None).
    """
    task_id = metadata.get("task_id", "unknown")
    code_prompt = metadata.get("code_prompt", "")
    test_harness = metadata.get("test", "")
    entry_point = metadata.get("entry_point", "")

    impl = _strip_markdown_fences(generated_code)

    # If the prediction already contains the full function definition (anywhere in
    # the code) use it directly; otherwise append it after code_prompt.
    # Use re.search so imports/comments before the def are allowed.
    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", impl, re.MULTILINE):
        full_code = impl
    else:
        full_code = code_prompt.rstrip() + "\n" + impl

    # The test field is a unittest.TestCase class that calls the function directly
    # by name. Drive it with unittest.main so the subprocess exits 0 on pass, 1 on fail.
    assembled = (
        full_code.rstrip()
        + "\n\n"
        + test_harness.strip()
        + "\n\n"
        + "import unittest\n"
        + "unittest.main(argv=[''], exit=True, verbosity=0)\n"
    )

    tmp_path = None
    keep_file = DEBUG_EVAL_DIR is not None
    try:
        if DEBUG_EVAL_DIR:
            debug_dir = Path(DEBUG_EVAL_DIR)
            debug_dir.mkdir(parents=True, exist_ok=True)
            safe_id = task_id.replace("/", "_")
            tmp_path = str(debug_dir / f"{safe_id}.py")
            with open(tmp_path, "w", encoding="utf-8") as tmp:
                tmp.write(assembled)
            logger.debug("BigCodeBench assembled test file: %s", tmp_path)
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(assembled)
                tmp_path = tmp.name

        process = subprocess.Popen(
            ["python", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            close_fds=True,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                if hasattr(os, "setsid"):
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=5)
            except Exception:
                pass
            logger.warning("BigCodeBench timeout for task %s", task_id)
            return {
                "correct": False,
                "score": 0.0,
                "task_id": task_id,
                "error": f"Timeout after {timeout}s",
            }

        if process.returncode == 0:
            logger.debug("BigCodeBench task %s: PASS", task_id)
            return {"correct": True, "score": 1.0, "task_id": task_id, "error": None}

        # Extract last meaningful stderr line
        error_lines = [l.strip() for l in stderr.strip().splitlines() if l.strip()]
        error_msg = error_lines[-1][:200] if error_lines else "Unknown error"
        logger.debug("BigCodeBench task %s: FAIL — %s", task_id, error_msg)
        return {"correct": False, "score": 0.0, "task_id": task_id, "error": error_msg}

    except Exception as exc:
        logger.error("BigCodeBench scorer error for task %s: %s", task_id, exc, exc_info=True)
        return {"correct": False, "score": 0.0, "task_id": task_id, "error": str(exc)}
    finally:
        if tmp_path and not keep_file and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
