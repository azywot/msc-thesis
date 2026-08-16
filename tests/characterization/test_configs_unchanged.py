"""B1: `generate_configs.py` output must not change.

This compares the generator against a snapshot of its OWN output, not against
the committed ``experiments/configs/`` tree.  The two have drifted (some
committed LoRA-inference configs were hand-edited after generation), so a
generator-vs-committed diff would fail here for reasons that have nothing to do
with any refactor.  See the design spec's Open Decisions.

The generator always writes to a temp directory: this test never touches
``experiments/configs/``.
"""

import hashlib
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

from .conftest import assert_matches_fixture

REPO = Path(__file__).resolve().parents[2]


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "_generate_configs", REPO / "scripts" / "generate_configs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_root_defaults_to_experiments_configs():
    """`--output-root` is additive: with no flags the script writes where it always did.

    The manifest test above can only ever exercise the temp-directory path, so
    this guards the default separately -- without running the generator, which
    would overwrite the committed tree.
    """
    module = _load_generator()
    assert module.CONFIGS_ROOT == REPO / "experiments" / "configs"
    default = inspect.signature(module.generate_suite).parameters["configs_root"].default
    assert default == module.CONFIGS_ROOT


def _manifest(root: Path) -> str:
    """path -> sha256, sorted, one per line.  Content-addressed and order-stable."""
    lines = []
    for p in sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml")):
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{p.relative_to(root).as_posix()}  {digest}")
    return "\n".join(lines) + "\n"


def test_generated_configs_unchanged(tmp_path, update_fixtures):
    out = tmp_path / "configs"
    out.mkdir()
    result = subprocess.run(
        [sys.executable, "scripts/generate_configs.py", "--output-root", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    manifest = _manifest(out)
    assert manifest.strip(), "generator produced no config files"
    assert_matches_fixture("configs.manifest", manifest, update_fixtures)
