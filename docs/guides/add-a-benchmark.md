# Add a benchmark

Three pieces: a **loader** that turns files into `DatasetExample`s, a
**`DatasetSpec` row** that says how the benchmark is prompted and reported, and
optionally a **prompt template**. Nothing else changes — the orchestrator and
the runner never learn your benchmark's name.

> This guide was executed end to end by adding a two-example `toybench`, running
> it through the loader, the prompt builder and the metrics path, and deleting
> it afterwards. The transcript is at the bottom.

---

## 1. Write the loader

`src/agent_engine/datasets/loaders/toybench.py`:

```python
"""Toy benchmark loader."""

import json
from typing import Any, Dict, List

from ..base import BaseDataset, DatasetExample, DatasetRegistry
from ..evaluators.metrics import evaluate_answer


@DatasetRegistry.register("toybench")
class ToyBenchDataset(BaseDataset):
    def load(self) -> List[DatasetExample]:
        data_path = self.config.data_dir / "ToyBench" / f"{self.config.split}.jsonl"
        if not data_path.exists():
            raise FileNotFoundError(f"ToyBench not found at: {data_path}")

        examples = []
        with open(data_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                row = json.loads(line)
                examples.append(
                    DatasetExample(
                        question_id=idx,
                        question=row["question"],
                        answer=row["answer"],
                        metadata={"level": row.get("level", "unknown")},
                    )
                )
        return examples

    def evaluate(self, prediction: str, ground_truth: str,
                 metadata: Dict[str, Any]) -> Dict[str, Any]:
        correct = prediction.strip().lower() == ground_truth.strip().lower()
        return {"correct": correct, "score": 1.0 if correct else 0.0}
```

Conventions the existing loaders follow, worth matching:

- **Path shape** is `data_dir / "<Name>" / f"{split}.jsonl"`. `data_dir` comes
  from the config and defaults to `./data`.
- **A missing file raises `FileNotFoundError` with the expected path in the
  message.** This is the error a new user hits first; make it say where to put
  the file.
- **A malformed row is logged and skipped, not fatal.** One bad line should not
  lose a 200-question run.
- **`answer` is a string.** `question_scorer` calls `is_float(ground_truth)`,
  which raises `TypeError` on `None` — see [known-issues](../known-issues.md).
- **Anything you want to slice by later goes in `metadata`** — level, category,
  year, subject.

Register the module so the decorator runs, in
`src/agent_engine/datasets/__init__.py`:

```python
from .loaders import bigcodebench, gaia, gpqa, hle, math, qa, toybench
```

Skipping this is the single most common mistake: `DatasetRegistry` will not know
the name, and the failure appears at dataset-construction time, not import time.

## 2. Add the `DatasetSpec` row

`src/agent_engine/datasets/spec.py`:

```python
DATASET_SPECS: Dict[str, DatasetSpec] = {
    ...
    "toybench": DatasetSpec("gaia", True, "level"),
}
```

The four fields:

| Field | Meaning |
|---|---|
| `template` | Prompt-template stem. `None` = no mapping; the raw name is tried, then the base template with a warning. |
| `stratified` | Whether `compute_metrics` emits a `per_level` breakdown. |
| `level_field` | The `metadata` key holding the level. |
| `level_fallback_field` | Consulted only when `level_field` is **absent** — not when it is present-but-`None`. |

Reuse an existing template when the task shape matches: GAIA, HLE and MuSiQue
all use `gaia`; AIME, MATH500, AMC and DeepMath all use `math`. A new template
is only needed for a genuinely new answer format.

> **Lookup is case-sensitive.** `get_spec("toybench")` resolves; `"ToyBench"`
> silently returns the default spec — no template mapping and no stratification.
> Use the exact registered spelling in your config. This is preserved
> deliberately from the code the spec table replaced, so it is not going to be
> "fixed".

## 3. Add a prompt template (only if needed)

Templates live in `src/agent_engine/prompts/templates/system/` and come in
pairs — `<name>.yaml` for AgentFlow and `<name>_baseline.yaml` for baseline.
Currently: `base`, `gaia`, `gpqa`, `math`, `bigcodebench`.

**Both files are required.** `PromptBuilder` appends `_baseline` in baseline
mode, so a missing baseline file falls back to `base` with a warning in
`experiment.log` — the run still completes, silently comparing your benchmark's
AgentFlow prompt against the generic baseline prompt. That is a broken
comparison that produces plausible-looking numbers, which is worse than a crash.

## 4. Write the config

```yaml
name: toybench_smoke
models:
  orchestrator: {name: Qwen3-8B, family: qwen3, path_or_id: Qwen/Qwen3-8B, role: orchestrator}
dataset:
  name: toybench
  split: validation
  subset_num: 2
tools:
  enabled_tools: [web_search]
```

For a suite of configs, add a suite to `scripts/generate_configs.py` rather than
writing them by hand — see [configuration.md](../configuration.md#where-configs-live).

## 5. Verify before spending GPU time

```bash
python -c "
from agent_engine.config.schema import DatasetConfig
from agent_engine.datasets.base import DatasetRegistry
import agent_engine.datasets
ds = DatasetRegistry.get(DatasetConfig(name='toybench', split='validation'))
print(len(ds.load()), 'examples')
"
pytest tests/unit/test_wiring_invariants.py -q
```

Note `DatasetRegistry.get` takes a **`DatasetConfig`, not a name**, and returns
an instance rather than the class.

The wiring-invariant tests check that every `DATASET_SPECS` key resolves to a
non-empty template in **both** modes, which catches the missing-`_baseline`
mistake above.

When checking a template by hand, pass **real tool schemas** to
`build_system_prompt`. With `tool_schemas=[]` every template renders to the same
~130-character stub and the AgentFlow and baseline versions look identical, so an
empty-schema check proves nothing.

---

## Walkthrough transcript

Adding `toybench` with two examples, exactly as above:

```
loader           : 2 examples
first question   : 'What is 2 + 2?'  answer='4'  metadata={'level': 'easy'}
spec             : DatasetSpec(template='gaia', stratified=True, level_field='level',
                               level_fallback_field=None)
get_spec("ToyBench") -> DatasetSpec(template=None, stratified=False, ...)   # case-sensitive
level_key        : easy | hard

template agentflow: 2250 chars, mentions sub_goal=True
template baseline : 1813 chars, mentions sub_goal=False
toybench template == gaia template: True

overall          : {'accuracy': 0.5, 'num_correct': '1 of 2'}
per_level        : {"easy": "1 of 1", "hard": "0 of 1"}
tool_usage       : {'web_search': 1}
```

The `per_level` block is what `stratified=True` plus `level_field="level"`
buys; with `stratified=False` the metrics carry the overall block alone. The
template lines confirm the spec row worked: `toybench` renders byte-identical to
`gaia`, and the AgentFlow and baseline variants really are different documents.

> **Two API details the walkthrough corrected.** `DatasetRegistry.get` takes a
> `DatasetConfig`, not a name string. And `compute_metrics` reads each result
> row's **`evaluation`** sub-dict — `{"correct": True}` at the top level is
> ignored and everything scores zero. The row shape the runner writes is
> `{"question_id", "evaluation", "tool_counts", "token_usage", ...}`; if your
> accuracy comes out 0.0 with obviously-correct predictions, check that first.

`toybench` was deleted afterwards, which is why it is not in the tree.
