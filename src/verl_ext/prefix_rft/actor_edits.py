"""The three edits that turn verl's update_policy into the Prefix-RFT one.

Kept as data, in its own verl-free module, so that ``actor.py`` and the test that
checks ``actor.py`` is still in sync with the installed verl apply exactly the
same transformation. If these ever diverge the guard would be checking itself.
"""

from __future__ import annotations

EDITS = [
    # 1. keep prefix_mask through data.select
    (
        '            select_keys.append("rollout_log_probs")\n',
        '            select_keys.append("rollout_log_probs")\n'
        "        # PREFIX-RFT EDIT 1: verl's fixed select_keys drops prefix_mask, which the\n"
        "        # entropy clip needs. Everything else about the selection is unchanged.\n"
        '        if "prefix_mask" in data.batch.keys():\n'
        '            select_keys.append("prefix_mask")\n',
    ),
    # 2. entropy is mandatory for the clip
    (
        "                    calculate_entropy = self.config.calculate_entropy or "
        "(entropy_coeff != 0)\n",
        "                    # PREFIX-RFT EDIT 2: the clip ranks prefix tokens by current-policy\n"
        "                    # entropy, so it must be computed whatever the config says.\n"
        "                    calculate_entropy = True\n",
    ),
    # 3. the clip itself, immediately before the policy loss
    (
        "                    # Compute policy loss (any function is expected to return 2 values)\n",
        "                    # PREFIX-RFT EDIT 3: zero the advantage of all but the top-k%\n"
        "                    # highest-entropy prefix tokens (paper section 3, A.2).\n"
        '                    if "prefix_mask" in model_inputs:\n'
        "                        advantages, n_zeroed = clip_prefix_advantage_by_entropy(\n"
        "                            advantages,\n"
        '                            model_inputs["prefix_mask"],\n'
        "                            entropy,\n"
        "                            keep_ratio=self.prefix_keep_ratio,\n"
        "                        )\n"
        '                        micro_batch_metrics["actor/prefix_tokens_zeroed"] = n_zeroed\n'
        '                        micro_batch_metrics["actor/prefix_tokens_total"] = int(\n'
        '                            model_inputs["prefix_mask"].sum().item()\n'
        "                        )\n"
        "\n"
        "                    # Compute policy loss (any function is expected to return 2 values)\n",
    ),
]


def apply_edits(source: str, edits=EDITS) -> str:
    """Apply each edit exactly once, failing loudly if an anchor is gone."""
    for old, new in edits:
        count = source.count(old)
        if count != 1:
            raise ValueError(
                f"anchor appears {count} times, expected exactly 1: {old!r}"
            )
        source = source.replace(old, new)
    return source


def extract_verl_update_policy() -> str:
    """Return verl's ``DataParallelPPOActor.update_policy`` source, verbatim.

    Read from the file rather than via ``inspect.getsource``: the method carries a
    ``@GPUMemoryLogger`` decorator that does not preserve ``__wrapped__``, so
    ``getsource`` returns the decorator's inner function instead.
    """
    import verl.workers.actor.dp_actor as dp_actor

    lines = open(dp_actor.__file__).read().splitlines(keepends=True)
    start = next(
        i for i, ln in enumerate(lines) if ln.startswith("    def update_policy(self, data")
    )
    end = start + 1
    while end < len(lines):
        ln = lines[end]
        if ln.strip() and not ln.startswith("        ") and not ln.startswith("    #"):
            break
        end += 1
    return "".join(lines[start:end]).rstrip() + "\n"


def expected_prefix_rft_update_policy() -> str:
    """What ``actor.py``'s update_policy must equal, derived from the installed verl."""
    return apply_edits(extract_verl_update_policy(), EDITS)


def actual_prefix_rft_update_policy() -> str:
    """The update_policy currently in ``actor.py``. It is the last thing in the file."""
    from pathlib import Path

    path = Path(__file__).with_name("actor.py")
    text = path.read_text()
    marker = "    def update_policy(self, data: DataProto):"
    idx = text.index(marker)
    return text[idx:].rstrip() + "\n"
