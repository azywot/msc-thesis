"""Jobs 004, 005 and 006 take an optional SUBSET that points them at a subset folder."""

import re
import subprocess
from pathlib import Path

JOBS = Path(__file__).resolve().parents[2] / "jobs" / "rsi"
VARS = ("COLLECT_DIR", "RESCUE_DIR", "SUBSET_ARG")


def _resolve(job, env):
    """Run the job's SUBSET handling lines (everything that sets one of VARS) and echo them."""
    text = (JOBS / job).read_text()
    start = text.index("# --- subset handling begin")
    end = text.index("# --- subset handling end")
    block = text[start:end]
    script = block + "\n" + "\n".join(f'echo "{v}=[${{{v}:-}}]"' for v in VARS)
    out = subprocess.run(
        ["bash", "-c", script],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return dict(re.fullmatch(r"(\w+)=\[(.*)\]", line).groups() for line in out.splitlines())


def test_bash_syntax():
    for job in ("004_collect_8b.job", "005_rescue_8b.job", "006_rescue_32b.job"):
        subprocess.run(["bash", "-n", str(JOBS / job)], check=True)


def test_collect_job_paths():
    got = _resolve("004_collect_8b.job", {})
    assert got["COLLECT_DIR"] == "data/training/rsi_phase2/collect_8b" and got["SUBSET_ARG"] == ""
    got = _resolve("004_collect_8b.job", {"SUBSET": "200"})
    assert got["COLLECT_DIR"] == "data/training/rsi_phase2/collect_8b_subset200"
    assert got["SUBSET_ARG"] == "--subset 200"


def test_rescue_jobs_paths():
    for job, name in (("005_rescue_8b.job", "rescue_8b"), ("006_rescue_32b.job", "rescue_32b")):
        got = _resolve(job, {})
        assert got["COLLECT_DIR"] == "data/training/rsi_phase2/collect_8b"
        assert got["RESCUE_DIR"] == f"experiments/results/rsi/phase2/{name}"
        assert got["SUBSET_ARG"] == ""
        got = _resolve(job, {"SUBSET": "50"})
        assert got["COLLECT_DIR"] == "data/training/rsi_phase2/collect_8b_subset50"
        assert got["RESCUE_DIR"] == f"experiments/results/rsi/phase2/{name}_subset50"
        assert got["SUBSET_ARG"] == "--subset 50"


def test_jobs_use_the_variables():
    collect = (JOBS / "004_collect_8b.job").read_text()
    assert '--output-dir "$COLLECT_DIR"' in collect and "$SUBSET_ARG" in collect
    for job in ("005_rescue_8b.job", "006_rescue_32b.job"):
        text = (JOBS / job).read_text()
        assert '"$COLLECT_DIR"/collected_*.jsonl' in text
        assert '--out-dir "$RESCUE_DIR"' in text and "$SUBSET_ARG" in text
