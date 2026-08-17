"""Check that the copied update_policy in actor.py still matches the installed verl.

``PrefixRFTActor.update_policy`` is a verbatim copy of verl's, plus three marked
edits, because verl's fixed ``select_keys`` drops ``prefix_mask`` and there is no
smaller seam. A verl upgrade that changes the original would otherwise leave our
copy silently stale: training would keep running, on the old loss body.

This re-derives the expected copy from whatever verl is installed and diffs it
against what is in the file. Run it under cosmas-train, where verl lives.

Usage:
    python scripts/check_prefix_rft_actor_sync.py
"""

from __future__ import annotations

import difflib
import sys

from verl_ext.prefix_rft.actor_edits import (
    actual_prefix_rft_update_policy,
    expected_prefix_rft_update_policy,
)


def main() -> int:
    try:
        expected = expected_prefix_rft_update_policy()
    except ValueError as exc:
        print("FAILED: could not apply the Prefix-RFT edits to verl's update_policy.")
        print(f"  {exc}")
        print("\nverl's update_policy has changed shape. Re-derive actor.py against the")
        print("new original, re-check each edit still does what it says, and update")
        print("actor_edits.py.")
        return 1

    actual = actual_prefix_rft_update_policy()
    if actual == expected:
        n = len(expected.splitlines())
        print(f"PASSED: actor.py's update_policy matches the installed verl ({n} lines).")
        return 0

    print("FAILED: actor.py's update_policy has drifted from the installed verl.\n")
    diff = difflib.unified_diff(
        expected.splitlines(keepends=True),
        actual.splitlines(keepends=True),
        fromfile="expected (verl + Prefix-RFT edits)",
        tofile="actual (src/verl_ext/prefix_rft/actor.py)",
    )
    sys.stdout.writelines(diff)
    print("\nRegenerate actor.py from the current verl rather than hand-patching it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
