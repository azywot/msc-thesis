"""The four edits that turn the vendored _async_set_up into the Prefix-RFT one.

Same arrangement as ``actor_edits.py`` and ``trainer_edits.py``: the edits are data, so
``daemon.py`` and the check that ``daemon.py`` is still in sync with the vendored
AgentFlow apply exactly the same transformation.

Why a copy at all: the vendored loop builds ONE ``original_sample`` dict per question and
hands the same object to every rollout of it. ``prefix_k`` differs between rollouts of a
question by construction (one of n is prefixed), so a shared dict cannot carry it. There
is no seam inside the loop to override, and ``VENDORED.md`` forbids editing the vendored
file in place.

A note on ``original_sample``: ``self._task_id_to_original_sample`` contains that name as
a substring, so a blanket rename would corrupt the attribute. Every edit below anchors on
a full line for that reason.
"""

from __future__ import annotations

EDITS = [
    # 1. accumulate the dispatched k values so they can be summarised afterwards
    (
        "        for i in range(num_samples):\n",
        "        # PREFIX-RFT EDIT 1: collect the dispatched prefix lengths for metrics.\n"
        "        ks = []\n"
        "\n"
        "        for i in range(num_samples):\n",
    ),
    # 2. per-rollout copy carrying prefix_k, in place of the shared dict
    (
        "            # For training, each sample is rolled out multiple times\n"
        "            for j in range(rollouts_per_sample):\n"
        '                task_metadata = {"data_id": data_id, "is_train": is_train}\n',
        "            # For training, each sample is rolled out multiple times\n"
        "            for j in range(rollouts_per_sample):\n"
        "                # PREFIX-RFT EDIT 2: the vendored loop reuses one dict for every\n"
        "                # rollout of a question. prefix_k differs per rollout, so take a\n"
        "                # copy and stamp it here.\n"
        "                sample = dict(original_sample)\n"
        '                sample["prefix_k"] = self._prefix_k_for(sample, j, is_train)\n'
        '                ks.append(sample["prefix_k"])\n'
        "\n"
        '                task_metadata = {"data_id": data_id, "is_train": is_train}\n',
    ),
    # 3. queue the per-rollout copy, not the shared dict
    (
        "                    sample=original_sample,\n",
        "                    sample=sample,  # PREFIX-RFT EDIT 3: the per-rollout copy\n",
    ),
    # 4. remember the per-rollout copy, so prefix_k survives into batch reconstruction
    (
        "                self._task_id_to_original_sample[rollout_id] = original_sample\n",
        "                # PREFIX-RFT EDIT 4: store the copy; get_train_data_batch needs\n"
        "                # the per-rollout prefix_k to rebuild prefix_mask.\n"
        "                self._task_id_to_original_sample[rollout_id] = sample\n",
    ),
    # 5. summarise the dispatch once every task is queued
    (
        '        print(f"Total tasks queued: {self._total_tasks_queued}")\n',
        '        print(f"Total tasks queued: {self._total_tasks_queued}")\n'
        "\n"
        "        # PREFIX-RFT EDIT 5: record what was actually dispatched. Without this the\n"
        "        # run cannot be distinguished from plain GRPO by its logs. print(), not\n"
        "        # logger.info(): INFO from this package does not reach the SLURM log, which\n"
        "        # is why job 25753032 could not be diagnosed from its output. The vendored\n"
        "        # daemon prints for the same reason.\n"
        "        self.last_prefix_metrics = self._summarise_prefix_dispatch(ks)\n"
        "        print(\n"
        '            f"Prefix dispatch: {sum(1 for k in ks if k > 0)} of {len(ks)} rollouts "\n'
        '            f"prefixed (is_train={is_train}); ks={ks}"\n'
        "        )\n",
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


def extract_vendored_async_set_up() -> str:
    """Return ``AgentModeDaemon._async_set_up`` source, verbatim, from the vendored file."""
    import fine_tuning.agentflow.verl.daemon as vendored

    lines = open(vendored.__file__).read().splitlines(keepends=True)
    marker = "    async def _async_set_up(self, data, server_addresses, is_train=True):"
    start = next(i for i, ln in enumerate(lines) if ln.startswith(marker))
    end = start + 1
    while end < len(lines):
        ln = lines[end]
        if ln.strip() and not ln.startswith("        ") and not ln.startswith("    #"):
            break
        end += 1
    return "".join(lines[start:end]).rstrip() + "\n"


def expected_prefix_rft_async_set_up() -> str:
    """What ``daemon.py``'s _async_set_up must equal, derived from the vendored file."""
    return apply_edits(extract_vendored_async_set_up(), EDITS)


def actual_prefix_rft_async_set_up() -> str:
    """The _async_set_up currently in ``daemon.py``."""
    from pathlib import Path

    lines = Path(__file__).with_name("daemon.py").read_text().splitlines(keepends=True)
    marker = "    async def _async_set_up(self, data, server_addresses, is_train=True):"
    start = next(i for i, ln in enumerate(lines) if ln.startswith(marker))
    end = start + 1
    while end < len(lines):
        ln = lines[end]
        if ln.strip() and not ln.startswith("        ") and not ln.startswith("    #"):
            break
        end += 1
    return "".join(lines[start:end]).rstrip() + "\n"
