"""The two edits that turn the vendored _train_step into the Prefix-RFT one.

Same arrangement as ``actor_edits.py``: the edits are data, so ``trainer.py`` and
the check that ``trainer.py`` is still in sync with the vendored AgentFlow apply
exactly the same transformation.

Why a copy at all: ``AgentFlowTrainer._train_step`` is one long method with no
smaller seam, and ``VENDORED.md`` forbids editing the vendored file in place. The
alternative, rebinding the vendored module's ``compute_advantage`` attribute, was
rejected as an invisible patch.
"""

from __future__ import annotations

EDITS = [
    # 1. promote the daemon and tell it the current step, before anything else
    (
        "        metrics = {}\n        timing_raw = {}\n",
        "        metrics = {}\n"
        "        timing_raw = {}\n"
        "\n"
        "        # PREFIX-RFT EDIT 1: the vendored fit() constructs a plain AgentModeDaemon\n"
        "        # (trainer.py:427), so promote it here, and hand it the step the cosine\n"
        "        # prefix schedule needs.\n"
        "        if self.config.get('prefix_rft', {}).get('enable', False):\n"
        "            self._ensure_prefix_daemon()\n"
        "            self.agent_mode_daemon.set_global_step(self.global_steps)\n",
    ),
    # 2. rewrite the advantage on prefix tokens, right after verl computed it
    (
        "            # after advantages are assinged, we begin to drop "
        "(1) long prompt (2) floor to ppo minisize\n",
        "            # PREFIX-RFT EDIT 2: overwrite the advantage on replayed teacher\n"
        "            # tokens (paper section 3). Questions with no prefixed rollout are\n"
        "            # left exactly as verl computed them.\n"
        "            self._apply_prefix_advantage(batch, metrics)\n"
        "\n"
        "            # after advantages are assinged, we begin to drop "
        "(1) long prompt (2) floor to ppo minisize\n",
    ),
]


def apply_edits(source: str, edits=EDITS) -> str:
    """Apply each edit exactly once, failing loudly if an anchor is gone."""
    for old, new in edits:
        count = source.count(old)
        if count != 1:
            raise ValueError(f"anchor appears {count} times, expected exactly 1: {old!r}")
        source = source.replace(old, new)
    return source


def extract_vendored_train_step() -> str:
    """Return ``AgentFlowTrainer._train_step`` source, verbatim, from the vendored file."""
    import fine_tuning.agentflow.verl.trainer as vendored

    lines = open(vendored.__file__).read().splitlines(keepends=True)
    start = next(
        i for i, ln in enumerate(lines) if ln.startswith("    def _train_step(self, batch_dict")
    )
    end = start + 1
    while end < len(lines):
        ln = lines[end]
        if ln.strip() and not ln.startswith("        ") and not ln.startswith("    #"):
            break
        end += 1
    return "".join(lines[start:end]).rstrip() + "\n"


def expected_prefix_rft_train_step() -> str:
    """What ``trainer.py``'s _train_step must equal, derived from the vendored file."""
    return apply_edits(extract_vendored_train_step(), EDITS)


def actual_prefix_rft_train_step() -> str:
    """The _train_step currently in ``trainer.py``. It is the last thing in the file."""
    from pathlib import Path

    text = Path(__file__).with_name("trainer.py").read_text()
    marker = "    def _train_step(self, batch_dict: dict) -> dict:"
    return text[text.index(marker) :].rstrip() + "\n"
