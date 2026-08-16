"""``bigcodebench_scorer`` — fence stripping, the double-prepend guard, and the
subprocess contract.

CLAUDE.md calls out the ``re.search`` full-definition detection as a real
subtlety: get it wrong in either direction and you either lose the imports the
stub carries or you define the function twice.  Both directions are pinned here.

These tests really do spawn subprocesses, but only over stdlib code — no
network, no dataset, no model.  Each runs in well under a second except the
timeout case, which is bounded by an explicit ``timeout=1``.
"""

import shutil

import pytest

from agent_engine.datasets.evaluators.bigcodebench_scorer import (
    _strip_markdown_fences,
    evaluate_bigcodebench,
)

# The scorer shells out to a bare ``python``.  Skip rather than fail where that
# name is not on PATH -- see test_the_scorer_shells_out_to_a_bare_python below.
needs_python = pytest.mark.skipif(
    shutil.which("python") is None, reason="scorer invokes a bare 'python'"
)


PASSING_TEST = """
import unittest

class TestCases(unittest.TestCase):
    def test_it(self):
        self.assertEqual(task_func(2), 4)
"""

FAILING_TEST = """
import unittest

class TestCases(unittest.TestCase):
    def test_it(self):
        self.assertEqual(task_func(2), 5)
"""


def _metadata(test=PASSING_TEST, code_prompt="def task_func(n):\n", task_id="BigCodeBench/1"):
    return {
        "task_id": task_id,
        "code_prompt": code_prompt,
        "test": test,
        "entry_point": "task_func",
    }


# --- fence stripping ------------------------------------------------------


def test_a_python_fence_is_stripped():
    assert _strip_markdown_fences("```python\nx = 1\n```") == "x = 1"


def test_a_bare_fence_is_stripped():
    assert _strip_markdown_fences("```\nx = 1\n```") == "x = 1"


def test_prose_around_a_fence_is_dropped():
    text = "Here is my solution:\n```python\nx = 1\n```\nHope that helps!"
    assert _strip_markdown_fences(text) == "x = 1"


def test_unfenced_text_is_returned_stripped():
    assert _strip_markdown_fences("  x = 1  ") == "x = 1"


def test_only_the_first_fence_survives():
    """``re.search`` is non-greedy and takes one match, so a reply that shows
    the solution and then an example usage block loses the second block."""
    text = "```python\nx = 1\n```\nand then\n```python\nprint(x)\n```"
    assert _strip_markdown_fences(text) == "x = 1"


# --- the double-prepend guard ---------------------------------------------


@needs_python
def test_a_full_definition_is_used_without_the_stub():
    """Prediction already defines the function, so ``code_prompt`` must not be
    prepended -- doing so would produce two defs and shadow the real one."""
    prediction = "```python\nimport math\n\ndef task_func(n):\n    return n * 2\n```"
    # A stub that would break the file if it were prepended.
    result = evaluate_bigcodebench(prediction, _metadata(code_prompt="def task_func(n):"))
    assert result["correct"] is True


@needs_python
def test_top_level_code_is_appended_after_the_stub():
    """The prepend path, exercised with a prediction that is itself top-level:
    the stub supplies ``task_func``, the prediction supplies its helper."""
    prediction = "def helper(n):\n    return n * 2"
    result = evaluate_bigcodebench(
        prediction, _metadata(code_prompt="def task_func(n):\n    return helper(n)\n")
    )
    assert result["correct"] is True


@needs_python
@pytest.mark.xfail(
    reason="pre-existing bug: _strip_markdown_fences() strips the body's indentation",
    strict=True,
)
def test_a_bare_body_is_appended_after_the_stub():
    """PRE-EXISTING BUG — reported, deliberately not fixed under this refactor.

    ``_strip_markdown_fences`` ends in ``.strip()`` (and ``.strip()`` on the
    captured group), which removes the leading whitespace of the *first* line.
    A prediction that is a bare function body -- exactly what the ``code_prompt``
    prepend path exists to handle -- therefore arrives de-indented::

        "    return n * 2"          -> "return n * 2"
        "    x = 1\\n    return x"   -> "x = 1\\n    return x"

    Appending either to ``def task_func(n):`` yields an IndentationError, so
    every such prediction scores 0 regardless of correctness.  Only the
    full-definition branch works today.  See docs/known-issues.md.
    """
    prediction = "```python\n    return n * 2\n```"
    result = evaluate_bigcodebench(prediction, _metadata(code_prompt="def task_func(n):\n"))
    assert result["correct"] is True


@needs_python
def test_imports_before_the_def_still_count_as_a_full_definition():
    """This is exactly why the check is ``re.search`` with MULTILINE rather
    than ``startswith``: real predictions put imports first."""
    prediction = (
        "```python\n"
        "import math\n"
        "# a comment\n"
        "def task_func(n):\n"
        "    return int(math.pow(n, 2))\n"
        "```"
    )
    result = evaluate_bigcodebench(prediction, _metadata(code_prompt="THIS WOULD NOT PARSE"))
    assert result["correct"] is True


@needs_python
def test_an_indented_definition_counts_too():
    """``^\\s*def`` allows leading whitespace on the line."""
    prediction = "if True:\n    def task_func(n):\n        return n * 2"
    result = evaluate_bigcodebench(prediction, _metadata(code_prompt="THIS WOULD NOT PARSE"))
    assert result["correct"] is True


@needs_python
def test_the_entry_point_name_decides_which_branch_runs():
    """The pattern interpolates ``entry_point``, so a prediction defining only
    a differently-named function falls through to the prepend path.

    Both directions are asserted with the *same* stub, which is valid Python
    only when it is actually prepended: the first call must ignore it, the
    second must use it."""
    defines_entry_point = "def task_func(n):\n    return n * 2"
    defines_helper_only = "def helper(n):\n    return n * 2"
    stub = "def task_func(n):\n    return helper(n)\n"

    # Direct branch: the stub is dropped, so the missing `helper` never matters.
    assert evaluate_bigcodebench(defines_entry_point, _metadata(code_prompt=stub))["correct"]
    # Prepend branch: the stub supplies task_func, the prediction supplies helper.
    assert evaluate_bigcodebench(defines_helper_only, _metadata(code_prompt=stub))["correct"]


# --- result contract ------------------------------------------------------


@needs_python
def test_a_passing_run_reports_score_one_and_no_error():
    result = evaluate_bigcodebench("def task_func(n):\n    return n * 2", _metadata())
    assert result == {
        "correct": True,
        "score": 1.0,
        "task_id": "BigCodeBench/1",
        "error": None,
    }


@needs_python
def test_a_failing_assertion_reports_the_last_stderr_line():
    result = evaluate_bigcodebench(
        "def task_func(n):\n    return n * 3", _metadata(test=FAILING_TEST)
    )
    assert result["correct"] is False
    assert result["score"] == 0.0
    assert result["error"]  # unittest's summary line, not None
    assert len(result["error"]) <= 200


@needs_python
def test_a_syntax_error_is_a_failure_not_an_exception():
    result = evaluate_bigcodebench("def task_func(n):\n    this is not python", _metadata())
    assert result["correct"] is False
    assert result["error"] is not None


@needs_python
def test_a_missing_task_id_defaults_to_unknown():
    metadata = _metadata()
    del metadata["task_id"]
    prediction = "def task_func(n):\n    return n * 2"
    assert evaluate_bigcodebench(prediction, metadata)["task_id"] == "unknown"


@needs_python
def test_a_hanging_solution_is_killed_and_reported_as_a_timeout():
    prediction = "import time\n\ndef task_func(n):\n    time.sleep(30)\n    return n * 2"
    result = evaluate_bigcodebench(prediction, _metadata(), timeout=1)
    assert result == {
        "correct": False,
        "score": 0.0,
        "task_id": "BigCodeBench/1",
        "error": "Timeout after 1s",
    }


@needs_python
def test_the_temp_file_is_removed_after_a_run(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    evaluate_bigcodebench("def task_func(n):\n    return n * 2", _metadata())
    assert list(tmp_path.iterdir()) == []


def test_the_scorer_shells_out_to_a_bare_python():
    """Recorded, not fixed: environment sensitivity worth knowing about.

    ``subprocess.Popen(["python", ...])`` resolves through PATH, so predictions
    are executed by whatever ``python`` the shell finds -- not necessarily the
    interpreter running the suite.  On this cluster that is the system Python,
    several minor versions behind the project's, so a prediction using newer
    syntax would be scored wrong for reasons unrelated to the model.  Anyone
    reporting BigCodeBench numbers should check which ``python`` is on PATH.
    """
    import inspect

    from agent_engine.datasets.evaluators import bigcodebench_scorer

    source = inspect.getsource(bigcodebench_scorer.evaluate_bigcodebench)
    assert '["python", tmp_path]' in source
