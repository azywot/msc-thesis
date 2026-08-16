"""B2: exported prompts, tool schemas, and prompt templates must not change.

Two complementary locks:

``test_exported_prompts_unchanged``
    Runs ``scripts/export_prompts.py`` and snapshots its JSON.  This is the
    only check that covers the **tool schemas**, and it exercises the prompt
    templates as ``PromptBuilder`` actually loads them.

``test_prompt_templates_unchanged``
    Hashes every template file on disk.  The export script hardcodes
    ``["base", "gaia", "gpqa", "math"]`` (``scripts/export_prompts.py:42``), so
    on its own it leaves the ``*_baseline.yaml`` templates -- the entire
    baseline arm of the AgentFlow-vs-baseline comparison -- unlocked.  This
    manifest closes that gap and also makes a newly added template show up as a
    fixture diff rather than passing silently.

Neither test writes anywhere but a temp directory.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from .conftest import assert_matches_fixture

REPO = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO / "src" / "agent_engine" / "prompts" / "templates"


def test_exported_prompts_unchanged(tmp_path, update_fixtures):
    out = tmp_path / "prompts.json"
    result = subprocess.run(
        [sys.executable, "scripts/export_prompts.py", "--output", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["system_prompts"], "export produced no system prompts"
    assert data["tool_schemas"], "export produced no tool schemas"

    canonical = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    assert_matches_fixture("prompts.json", canonical, update_fixtures)


def test_prompt_templates_unchanged(update_fixtures):
    lines = []
    for p in sorted(TEMPLATE_DIR.rglob("*.yaml")):
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{p.relative_to(TEMPLATE_DIR).as_posix()}  {digest}")
    assert lines, f"no templates found under {TEMPLATE_DIR}"

    assert_matches_fixture("prompt_templates.manifest", "\n".join(lines) + "\n", update_fixtures)
