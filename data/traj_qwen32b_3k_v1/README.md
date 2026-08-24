# traj_qwen32b_3k_v1

Qwen3-32B teacher trajectories (AgentFlow, sub-agent mode) collected on the
`data/training_30k` question pool. Used for SFT, prefix, and GRPO with Qwen3-8B.

| file | rows | purpose |
|---|---:|---|
| `train.parquet` | 3000 | SFT / prefix training; also usable as a GRPO training pool |
| `val_200.parquet` | 200 | SFT validation — same distribution as train |
| `val_60.parquet` | 60 | GRPO validation — questions only, includes AIME |

## Verification status (2026-08-24)

Re-checked against the shipped files rather than against this README. What holds:
schemas and row counts, zero nulls, zero duplicate questions, every split-disjointness
claim (train x `val_200`, train x `val_60`, `val_200` x `val_60`, `val_60` x
`val_combined`, `val_60` x `val_aime` all zero overlap), message structure (every
trajectory is `[system, user, ...]` ending on an assistant turn, every `tool` turn
preceded by an assistant turn), complete `<think>` removal, and the composition and
token statistics below. The stored system prompts are byte-identical to what the current
`PromptBuilder` renders in sub-agent mode (`deepmath` to the math template, `hotpotqa`
and `nq` to the gaia template), so there is no drift between training context and
inference context, and the collection-time suffix is confirmed stripped.

Two blockers were found. Both are flagged in place below; read them before launching:

1. `val_200.parquet` fails the folded-SFT pre-flight gate (one over-length row).
2. The `data.val_files` list in the GRPO snippet raises at dataset construction
   (`extra_info.difficulty` dtype mismatch).

`train.parquet` itself is clean and passes every gate.

## Schema

`train.parquet` and `val_200.parquet`: `data_source, question, result, extra_info, messages`
— identical to `combined_train.parquet` plus `messages`, so one file serves both consumers.

| column | used by |
|---|---|
| `messages` | SFT / prefix — the full conversation |
| `data_source`, `question`, `result`, `extra_info` | GRPO — question + ground truth |

`val_60.parquet` omits `messages` (GRPO never reads it).

## Statistics

| | train | val_200 |
|---|---:|---:|
| rows / unique questions | 3000 | 200 |
| deepmath | 1500 (50.0%) | 99 (49.5%) |
| hotpotqa | 1273 (42.4%) | 85 (42.5%) |
| nq | 227 (7.6%) | 16 (8.0%) |
| **math : search** | 1500 : 1500 | 99 : 101 |
| median tokens / trajectory | 1471 | 1505 |
| p95 tokens / trajectory | 2119 | 2067 |
| median turns | 2 | 2 |
| uses ≥1 tool | 72.8% | 73.0% |
| solved only after a failed attempt | 15.3% | 15.5% |

Composition matches the source pool (deepmath 50.0%, hotpotqa 42.5%, nq 7.5%).
`val_200` is stratified on source x tool-use x difficulty jointly, so it mirrors train
on all three axes rather than source alone. Token counts are post-`<think>`-strip.

Reading the last three rows: token counts include the system prompt (~700 tokens) and
reproduce under the Qwen3-8B tokenizer (median 1470 measured vs 1471 stated; p95 comes
out slightly higher, 2239, depending on how the turns are joined). "Turns" counts
orchestrator turns after the planning turn, so the median trajectory holds 3 assistant
decisions: a plan, one tool call, and an answer. Median tool calls per trajectory is 1.

`val_60.parquet`: 20 deepmath, 17 hotpotqa, 3 nq, 20 AIME —
the same proportions as `val_combined.parquet`. Its 40 non-AIME questions were drawn from
the 22,775 pool questions the teacher **never attempted**, so it carries no
teacher-solvability selection bias. Disjoint from train, `val_200`, `val_combined`, `val_aime`.

## How to use

### SFT — fold first

`messages` is the stored multi-turn transcript, which the orchestrator **never sees at
inference**: every turn is rebuilt as a fresh `[system, user]` memory prompt by
`AgenticOrchestrator._build_memory_prompt`. Training on the raw transcript optimises a
context that never occurs at evaluation — this is why `007` was superseded by `008`,
where loss fell while task performance regressed.

```bash
python scripts/build_sft_parquet.py \
    --from-parquet data/traj_qwen32b_3k_v1/train.parquet \
    --format folded \
    --output-dir data/traj_qwen32b_3k_v1/folded \
    --output-name folded_train.parquet
```

Folding yields ~3 rows per question, so the 50:50 math:search balance holds at the
*question* level but not the *row* level (search trajectories take more turns).
Measured: 3000 trajectories to 9457 rows (3.15x), one row per assistant decision.

**That command folds `train.parquet` only.** `_refold_parquet` looks for a sibling val
split by stripping a `_train` suffix off the input stem, and this file is named
`train.parquet`, so the check never fires and `val_200` is skipped without a warning.
Run it a second time with `--from-parquet data/traj_qwen32b_3k_v1/val_200.parquet`.

Then gate both outputs. This is not optional; it is the check that catches the
format bug behind the `007` regression:

```bash
python scripts/check_sft_folded_format.py \
    --folded data/traj_qwen32b_3k_v1/folded/folded_train.parquet \
    --native data/traj_qwen32b_3k_v1/train.parquet
```

`folded_train` passes all six gates (9457 rows, max 10,070 tokens, well inside a
16,384 `data.max_length`).

**Blocker: folded `val_200` fails the gate.** One deepmath trajectory folds to a row of
18,871 tokens whose user turn alone is 17,648, so right-truncation at `max_length=16384`
consumes the entire 464-token target and the row reports
`row_with_zero_supervised_tokens`. Before using `val_200` for SFT validation, drop that
question, filter rows over `max_length` at fold time, or raise `max_length`. `train`
is unaffected; only `val_200` carries an over-length trajectory (native max 18,717
tokens vs 10,352 for train).

### GRPO — use as-is; never the folded files

```yaml
data.train_files: 'data/traj_qwen32b_3k_v1/train.parquet'
data.val_files:
  - 'data/traj_qwen32b_3k_v1/val_60.parquet'
  - '${BASE_DATA_DIR}/val/val_combined.parquet'   # keep: comparable to earlier runs
```

**Blocker: that `val_files` list does not load.** `RLHFDataset` builds one dataset per
file and calls `datasets.concatenate_datasets`, which aligns features strictly.
`extra_info.difficulty` is `double` in these files and `int64` in `val_combined.parquet`,
so construction raises:

```
ValueError: The features can't be aligned because the key difficulty ...
has unexpected type - Value('int64') (expected either Value('float64') or Value("null"))
```

The same applies to any run that pairs a file from this directory with one from
`data/training/` in a single `train_files` or `val_files` list. Cast one side before
use: every `difficulty` value here is integral, so `int64` with a null sentinel is the
lossless direction (nulls are 1500 / 101 / 40 rows in train / `val_200` / `val_60`, all
from the search sources, which carry no difficulty label). Using either file on its own
is fine.

`messages` is ignored by the GRPO data path. **Never** pass the folded files to GRPO —
~3 duplicate rows per prompt would corrupt the group-relative advantage estimates.

Caution on using `train.parquet` as a GRPO *training* pool: these 3,000 questions were
already solved by the teacher, so they are a selected-easy subset (~42% dead groups vs
~49.5% for unselected questions, measured on earlier rollouts). If the model is SFT'd on
them first, GRPO also sees few unseen problems. Prefer `combined_train.parquet` minus
these uids.

### Prefix-RFT — convert to the demonstration store first

Prefix-RFT does not read this parquet. `verl_ext.prefix_rft.demos.DemoStore` reads a
separate one-row-per-question store with its own schema (`question_key, question_id,
data_source, question, n_steps, steps[{response, tool_name, tool_result}]`), and
`scripts/build_prefix_demos.py` builds it from a `collected_*.jsonl`, filtering on
`correct` and carrying `question_id`. Neither field exists in `train.parquet`, so
convert first:

```python
import sys
import pandas as pd

sys.path[:0] = ["src", "scripts"]
from build_prefix_demos import records_to_demo_rows

df = pd.read_parquet("data/traj_qwen32b_3k_v1/train.parquet")
records = [
    {
        "question_id": i,
        "correct": True,
        "question": r["question"],
        "data_source": r["data_source"],
        "messages": [dict(m) for m in r["messages"]],
    }
    for i, r in enumerate(df.to_dict("records"))
]
pd.DataFrame(records_to_demo_rows(records)).to_parquet(
    "data/training/prefix_rft/prefix_demos_traj3k.parquet", index=False
)
```

`correct=True` is sound here: only verified-correct trajectories were written to this
file (see Provenance). `question_id` is positional and used for diagnostics and ordering
only. **Do not substitute `extra_info.idx` for it**: 170 of its values collide across
data sources in this file, which is exactly the failure `demos.py` warns about, attaching
a maths demonstration to a search question with nothing downstream to notice. The lookup
key is `question_key`, the SHA-1 of the question text.

Gate the result before training:

```bash
python scripts/check_prefix_demos.py \
    --demos data/training/prefix_rft/prefix_demos_traj3k.parquet
```

Verified 2026-08-24: all 3000 records convert, 9457 decisions, mean 3.15 per question,
gate `PASSED`. The conversion reuses `_classify_turns` and `_strip_thinking` from
`build_sft_parquet.py`, the same helpers the folded-SFT path uses, so the two views of a
trajectory cannot drift apart. The 9457 decisions match the 9457 folded SFT rows exactly.

Point training at it with `PREFIX_DEMOS_PATH` (see `scripts/train_orchestrator.py`).

**Coverage is what to check, not the format.** The store is matched to the RL training
questions by question text, so what matters is the overlap with `data.train_files`, not
the size of this file:

| RL train set + store | questions | with a demonstration |
|---|---:|---:|
| `combined_train.parquet` + existing `prefix_demos.parquet` | 1800 | 1358 (75%) |
| `combined_train.parquet` + this file | 1800 | 180 (10%) |
| this file as its own train set | 3000 | 3000 (100%) |

Only 180 of the 3000 questions here also appear in `combined_train.parquet`. Swapping the
store to this file while keeping `combined_train.parquet` as the RL training set therefore
drops prefix coverage from 75% to 10%. A miss is not an error (that question trains as
ordinary GRPO, which is what `demo_ratio` does), but at 10% the prefix mechanism barely
fires. Either merge the two stores, or take the drop deliberately and say so.

Using this file as its own RL training set gives full coverage, but then every question is
one the teacher already solved *and* supplies the prefix for, which compounds the
selected-easy problem noted under GRPO above. Prefer merging.

634 of the 3000 trajectories (21%) hold a single decision and can never carry a prefix;
2366 are prefixable. The existing store has the same property (273 of 1358).

### The validation sets are not interchangeable

| set | n | metric | AIME |
|---|---:|---|---|
| `val_200.parquet` | 200 | SFT token loss | no |
| `val_60.parquet` | 60 | GRPO task reward | yes (20) |
| `val_combined.parquet` | 50 | GRPO task reward | yes (20) |

`val_200` must stay i.i.d. with `train.parquet` — that is the only condition under which
val loss detects overfitting. AIME is deliberately excluded: token loss on AIME traces
measures teacher-imitation, not problem-solving, and mixing distributions makes a rising
curve uninterpretable.

## Provenance

- Teacher Qwen3-32B, thinking enabled, T=0.6 / top_p=0.95 / top_k=20; sub-agents Qwen3-1.7B greedy.
- `<think>` blocks and the collection-time `system_prompt_suffix` are stripped; the student trains with thinking disabled.
- One trajectory per question; tool-using traces preferred over tool-free.
- Every answer verified: ~90% exact match after normalisation, the rest individually reviewed (97.4% reviewer agreement on re-judged records). 158 trajectories with incorrect-but-passing answers excluded.
- All three files verified byte-identical to `build_sft_parquet.py` output.
- Source collection log: `data/training_30k/sft_collection_raw.jsonl` (12,195 records incl. failures, with `collection_run` provenance).
- Exclusion list `data/results/sft_exclude_final.json`; verdicts `data/results/sft_review_final.jsonl`.
- Held-out evaluation uses `data/training_30k/test/`, disjoint from everything here.

**None of the four paths named above currently exist in this repo** (`data/training_30k/`,
`sft_collection_raw.jsonl`, `data/results/sft_exclude_final.json`,
`data/results/sft_review_final.jsonl`), checked 2026-08-24. The parquets themselves are
intact and independently verified, but the audit trail behind the review, exclusion and
byte-identity claims cannot be re-run from here, and neither can
`build_prefix_demos.py`'s documented entry point, which is why the Prefix-RFT section
above converts from the parquet instead. Restore the collection log, or treat those four
bullets as a record of work done elsewhere rather than something reproducible.

## Known limitations

- 27.2% of train trajectories use no tool (the 32B answered from memory, mostly deepmath). An 8B cannot reproduce this, so these examples risk teaching unsupported guessing.
- 221 `code_generator` calls returned empty output (5.4% of trajectories). Genuine at collection time, not a processing artefact.
- `extra_info.difficulty` is stored as `double` here vs `int64` in the pool (nulls force the promotion). No information is lost, but the dtype is not cosmetic: it makes these files un-concatenable with the `data/training/` ones, which breaks the `val_files` list in the GRPO section above. Cast before mixing.
- `extra_info` field order also differs from `combined_train.parquet` (`golden_answers` and `difficulty` are swapped), so strict `pyarrow.concat_tables` fails on order alone; `promote_options="permissive"` and `pandas.concat` both handle it.
- One `val_200` trajectory is 18,717 tokens and exceeds a 16,384 `data.max_length` after folding. See the SFT section.
- `extra_info.idx` is not unique (2830 distinct values over 3000 rows) and collides across data sources 170 times. Key on the question text, never on `idx`.
- Ground truth is taken as correct; dataset-level label noise passes through.
- Only trajectories marked correct were verified; false negatives among the ~6,800 failures were not audited.
