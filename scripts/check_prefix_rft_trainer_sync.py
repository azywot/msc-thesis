"""Check the copied methods still match the sources they were copied from.

Prefix-RFT copies two long methods it cannot extend by subclassing:

- ``PrefixRFTActor.update_policy`` from verl, because verl's fixed ``select_keys``
  drops ``prefix_mask``;
- ``PrefixRFTTrainer._train_step`` from the vendored AgentFlow, because it is one
  method with no smaller seam and VENDORED.md forbids editing it in place.

Either source changing under us would leave the copy silently stale: training
would keep running, on the old body. This re-derives both copies and diffs them.

Run under cosmas-train, where verl lives.

Usage:
    python scripts/check_prefix_rft_trainer_sync.py
"""

from __future__ import annotations

import difflib
import sys

from verl_ext.prefix_rft import actor_edits, trainer_edits


def _check(name: str, expected_fn, actual_fn, guidance: str) -> bool:
    try:
        expected = expected_fn()
    except ValueError as exc:
        print(f"FAILED [{name}]: could not apply the Prefix-RFT edits.")
        print(f"  {exc}")
        print(f"  {guidance}")
        return False

    actual = actual_fn()
    if actual == expected:
        print(f"PASSED [{name}]: in sync ({len(expected.splitlines())} lines).")
        return True

    print(f"FAILED [{name}]: the copy has drifted from its source.\n")
    sys.stdout.writelines(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=f"expected ({name})",
            tofile=f"actual ({name})",
        )
    )
    print(f"\n  {guidance}")
    return False


def _check_agentflow_base() -> bool:
    """The Prefix-RFT config inlines the vendored AgentFlow config's keys.

    It cannot compose them: the vendored file declares hydra.searchpath, which
    Hydra allows only in a primary config. A re-vendor that changes those keys
    would otherwise change GRPO's training setup and leave Prefix-RFT on the old
    one, with no signal.
    """
    from pathlib import Path

    import yaml

    import fine_tuning.agentflow.verl as af
    import verl_ext.prefix_rft as pr

    vendored = yaml.safe_load((Path(af.__file__).parent / "config.yaml").read_text())
    ours = yaml.safe_load(
        (Path(pr.__file__).parent / "config" / "prefix_rft_trainer.yaml").read_text()
    )

    mismatches = []
    for key in ("agentflow", "data", "actor_rollout_ref"):
        if vendored.get(key) != ours.get(key):
            mismatches.append(key)

    if not mismatches:
        print("PASSED [config AGENTFLOW BASE]: inlined keys match the vendored config.")
        return True

    print("FAILED [config AGENTFLOW BASE]: inlined keys have drifted.\n")
    for key in mismatches:
        print(f"  {key}:")
        print(f"    vendored: {vendored.get(key)}")
        print(f"    ours:     {ours.get(key)}")
    print(
        "\n  Copy the vendored block into the AGENTFLOW BASE section of "
        "prefix_rft_trainer.yaml."
    )
    return False


def main() -> int:
    ok = True
    ok &= _check_agentflow_base()
    ok &= _check(
        "actor.update_policy",
        actor_edits.expected_prefix_rft_update_policy,
        actor_edits.actual_prefix_rft_update_policy,
        "verl's update_policy has changed. Regenerate actor.py against the new "
        "original and re-check each edit still does what it says.",
    )
    ok &= _check(
        "trainer._train_step",
        trainer_edits.expected_prefix_rft_train_step,
        trainer_edits.actual_prefix_rft_train_step,
        "The vendored AgentFlow _train_step has changed, most likely a re-vendor. "
        "Regenerate trainer.py against the new vendored file.",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
