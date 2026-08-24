# traj_qwen32b_3k_v1

Qwen3-32B teacher trajectories (AgentFlow, sub-agent mode) collected on the
`data/training_30k` question pool. This is the single pool behind every Qwen3-8B training
stage: SFT, Prefix-RFT and GRPO all read from this directory. See
"What using one pool for everything costs you" before reading any result off it.

## Files

Shipped:

| file | rows | purpose |
|---|---:|---|
| `train.parquet` | 3000 | source trajectories; GRPO training pool |
| `val_200.parquet` | 200 | source trajectories for the SFT validation split |
| `val_60.parquet` | 60 | GRPO validation, questions only, includes AIME |

Derived, rebuilt by the prep step below:

| file | rows | purpose |
|---|---:|---|
| `folded/folded_train.parquet` | 9457 | SFT training |
| `folded/folded_val.parquet` | 628 | SFT validation |
| `prefix_demos.parquet` | 3000 | Prefix-RFT demonstration store |

The `folded/` directory also holds matching `.jsonl` copies for eyeballing rows.

## Preparing the data

Only the three shipped parquets are guaranteed to reach you. The derived files may not
have travelled, and they are not interchangeable across tokenizers or `max_length`
settings, so rebuild them. From the repo root, in the training environment:

```bash
# 1. SFT training split
python scripts/build_sft_parquet.py \
    --from-parquet data/traj_qwen32b_3k_v1/train.parquet \
    --format folded --max-length 16384 \
    --output-dir data/traj_qwen32b_3k_v1/folded \
    --output-name folded_train.parquet

# 2. SFT validation split. It has to be a second call: _refold_parquet looks for a
#    sibling val split by stripping a `_train` suffix off the input stem, and this file
#    is named `train.parquet`, so the check never fires and val_200 is silently skipped.
python scripts/build_sft_parquet.py \
    --from-parquet data/traj_qwen32b_3k_v1/val_200.parquet \
    --format folded --max-length 16384 \
    --output-dir data/traj_qwen32b_3k_v1/folded \
    --output-name folded_val_train.parquet
mv data/traj_qwen32b_3k_v1/folded/folded_val_train.parquet \
   data/traj_qwen32b_3k_v1/folded/folded_val.parquet
mv data/traj_qwen32b_3k_v1/folded/folded_val_train.jsonl \
   data/traj_qwen32b_3k_v1/folded/folded_val.jsonl

# 3. Prefix-RFT demonstration store
python scripts/build_prefix_demos.py \
    --from-parquet data/traj_qwen32b_3k_v1/train.parquet \
    --output data/traj_qwen32b_3k_v1/prefix_demos.parquet
```

Pass `--max-length` the same value as `data.max_length` in the training job, and
`--tokenizer` too if you are not on Qwen3-8B. Step 2 prints
`Dropped 1 folded row(s) over --max-length 16384 (longest 18871 tokens); 0 question(s)
lost entirely`, which is expected and explained under SFT below.

Then gate all three. This is not optional. The folded gate is what catches the format bug
behind the `007` regression, and it is cheap next to a training run:

```bash
python scripts/check_sft_folded_format.py \
    --folded data/traj_qwen32b_3k_v1/folded/folded_train.parquet \
    --native data/traj_qwen32b_3k_v1/train.parquet
python scripts/check_sft_folded_format.py \
    --folded data/traj_qwen32b_3k_v1/folded/folded_val.parquet \
    --native data/traj_qwen32b_3k_v1/val_200.parquet
python scripts/check_prefix_demos.py \
    --demos data/traj_qwen32b_3k_v1/prefix_demos.parquet
```

Expected on a clean rebuild (reproduced 2026-08-24):

| gate | result |
|---|---|
| `folded_train` | 9457 rows, max 10,070 tokens, all six gates passed |
| `folded_val` | 628 rows, max 10,369 tokens, all six gates passed |
| `prefix_demos` | 3000 demonstrations, 9457 decisions, PASSED |

Anything else means stop and read the output rather than launching.

## Verification status (2026-08-24)

Re-checked against the shipped files rather than against this README. What holds:
schemas and row counts, zero nulls, zero duplicate questions, every split-disjointness
claim (train x `val_200`, train x `val_60`, `val_60` x `val_combined`, `val_60` x
`val_aime` all zero overlap), message structure (every trajectory is `[system, user, ...]`
ending on an assistant turn, every `tool` turn preceded by an assistant turn), complete
`<think>` removal, and the composition and token statistics below. The stored system
prompts are byte-identical to what the current `PromptBuilder` renders in sub-agent mode
(`deepmath` to the math template, `hotpotqa` and `nq` to the gaia template), so there is
no drift between training context and inference context, and the collection-time suffix
is confirmed stripped.

`train.parquet` passes every gate untouched. The one defect found was in `val_200`, where
a single over-length trajectory failed the folded-SFT gate; `--max-length` now handles it
during prep.

## Schema

`train.parquet` and `val_200.parquet`: `data_source, question, result, extra_info,
messages`. Same columns as `combined_train.parquet` plus `messages`, so one file serves
both the SFT and GRPO consumers.

| column | used by |
|---|---|
| `messages` | SFT and Prefix-RFT, the full conversation |
| `data_source`, `question`, `result`, `extra_info` | GRPO, question and ground truth |

`val_60.parquet` omits `messages`, which GRPO never reads.

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
| uses >=1 tool | 72.8% | 73.0% |
| solved only after a failed attempt | 15.3% | 15.5% |

Composition matches the source pool (deepmath 50.0%, hotpotqa 42.5%, nq 7.5%).
`val_200` is stratified on source x tool-use x difficulty jointly, so it mirrors train
on all three axes rather than source alone. Token counts are post-`<think>`-strip.

Reading the last three rows: token counts include the system prompt (about 700 tokens)
and reproduce under the Qwen3-8B tokenizer (median 1470 measured against 1471 stated; p95
comes out slightly higher at 2239, depending on how the turns are joined). "Turns" counts
orchestrator turns after the planning turn, so the median trajectory holds 3 assistant
decisions: a plan, one tool call, and an answer. Median tool calls per trajectory is 1.

`val_60.parquet`: 20 deepmath, 17 hotpotqa, 3 nq, 20 AIME, the same proportions as
`val_combined.parquet`. Its 40 non-AIME questions were drawn from the 22,775 pool
questions the teacher **never attempted**, so it carries no teacher-solvability selection
bias. Disjoint from train, `val_200`, `val_combined` and `val_aime`.

## How to use

### SFT

Train on `folded/`, never on `train.parquet` directly. `messages` is the stored multi-turn
transcript, which the orchestrator never sees at inference: every turn is rebuilt as a
fresh `[system, user]` memory prompt by `AgenticOrchestrator._build_memory_prompt`.
Training on the raw transcript optimises a context that never occurs at evaluation, which
is why `007` was superseded by `008`, where loss fell while task performance regressed.

```yaml
data.train_files: 'data/traj_qwen32b_3k_v1/folded/folded_train.parquet'
data.val_files: 'data/traj_qwen32b_3k_v1/folded/folded_val.parquet'
data.max_length: 16384
```

Folding yields about 3 rows per question (3000 trajectories to 9457 rows), so the 50:50
math:search balance holds at the *question* level but not the *row* level, because search
trajectories take more turns.

**One folded val row is dropped by `--max-length`.** A single deepmath trajectory folds to
a row of 18,871 tokens whose user turn alone is 17,648, so right-truncation at 16,384
would consume the whole 464-token target and leave the row with no supervised tokens.
Dropping a row is not dropping a question: that trajectory keeps its other 6 decisions, so
all 200 questions stay represented and `val_200` stays i.i.d. with train at the question
level. `train` has no such row (its longest folded row is 10,070 tokens).

### Prefix-RFT

Prefix-RFT does not read `train.parquet`. `verl_ext.prefix_rft.demos.DemoStore` reads a
separate one-row-per-question store with its own schema (`question_key, question_id,
data_source, question, n_steps, steps[{response, tool_name, tool_result}]`), which is what
`prefix_demos.parquet` is. Point training at it:

```bash
export PREFIX_DEMOS_PATH=data/traj_qwen32b_3k_v1/prefix_demos.parquet
```

`scripts/build_prefix_demos.py` normally builds that store from a `collected_*.jsonl`,
filtering on `correct` and carrying `question_id`. Neither field survives into a built
trajectory parquet, so the script takes a `--from-parquet` path as well. The adapter sets
`correct=True`, sound here because only verified-correct trajectories were written to this
file (see Provenance), and uses the row position as `question_id`, which is carried for
diagnostics and ordering only. **It does not use `extra_info.idx`**: 170 of that column's
values collide across data sources in this file, which is exactly the failure `demos.py`
warns about, attaching a maths demonstration to a search question with nothing downstream
to notice. The lookup key is `question_key`, the SHA-1 of the question text.

The conversion reuses `_classify_turns` and `_strip_thinking` from `build_sft_parquet.py`,
the same helpers the folded-SFT path uses, so the two views of a trajectory cannot drift
apart. Its 9457 decisions match the 9457 folded SFT rows exactly.

**Coverage is 100% here, and that is the point of using one pool.** The store is matched
to the RL training questions by question text, so what matters is the overlap with
`data.train_files`:

| RL train set + store | questions | with a demonstration |
|---|---:|---:|
| this pool + `prefix_demos.parquet` | 3000 | 3000 (100%) |
| `combined_train.parquet` + `prefix_demos.parquet` | 1800 | 180 (10%) |
| `combined_train.parquet` + the older `data/training/prefix_rft/prefix_demos.parquet` | 1800 | 1358 (75%) |

Only 180 of these 3000 questions also appear in `combined_train.parquet`, because the two
collections were run on different pools. That is why the stores are not interchangeable
and why merging them is pointless: 161 of those 180 are already in the older store, so a
merge would buy 19 questions. Keeping the RL pool and the store on the same collection is
what gets coverage to 100%.

634 of the 3000 trajectories (21%) hold a single decision and can never carry a prefix, so
2366 questions are actually prefixable. A miss is not an error, it means that question
trains as ordinary GRPO, which is what `demo_ratio` does.

### GRPO

```yaml
data.train_files: 'data/traj_qwen32b_3k_v1/train.parquet'
data.val_files: 'data/traj_qwen32b_3k_v1/val_60.parquet'
```

`messages` is ignored by the GRPO data path. **Never** pass the folded files to GRPO,
because about 3 duplicate rows per prompt would corrupt the group-relative advantage
estimates.

One file on each side means no dataset concatenation, which sidesteps a real dtype trap:
`extra_info.difficulty` is `double` in this directory and `int64` in `data/training/`, and
`RLHFDataset` builds one dataset per file and calls `datasets.concatenate_datasets`, which
aligns features strictly. Listing a file from here alongside one from there raises:

```
ValueError: The features can't be aligned because the key difficulty ...
has unexpected type - Value('int64') (expected either Value('float64') or Value("null"))
```

If you want `val_combined.parquet` alongside `val_60.parquet` for continuity with earlier
runs, cast the older file up to `double` so it matches this directory. The values are
small integers, which `float64` represents exactly, so nothing is lost:

```python
import pyarrow as pa
import pyarrow.parquet as pq

table = pq.read_table("data/training/val/val_combined.parquet")
i = table.schema.get_field_index("extra_info")
new = pa.struct([f.with_type(pa.float64()) if f.name == "difficulty" else f
                 for f in table.schema.field(i).type])
table = table.set_column(i, pa.field("extra_info", new),
                         table.column("extra_info").cast(new))
pq.write_table(table, "data/training/val/val_combined_f64.parquet")
```

Cast at the Arrow level, not through pandas. A pandas roundtrip promotes `year` and
`aime_id` to `float64` as well, because their non-AIME rows are null, and you trade one
alignment error for the next. Verified 2026-08-24: after this cast the two files
concatenate to 110 rows with `year` and `aime_id` still `int64`. The `extra_info` field
order still differs between the two files, which `datasets` does not mind.

### What using one pool for everything costs you

Worth stating explicitly in any write-up, because it is a property of the design rather
than a bug:

- These 3000 questions were all solved by the teacher, so they are a selected-easy subset.
  Measured on earlier rollouts, that is about 42% dead groups against about 49.5% for
  unselected questions. Fewer dead groups sounds good, but it means the GRPO signal comes
  from an easier distribution than the benchmarks you report on.
- SFT and GRPO now see the same questions, so GRPO gets almost no unseen problems and its
  headroom over the SFT checkpoint is compressed.
- Prefix-RFT then replays, on those same questions, the very teacher trajectory the SFT
  checkpoint was trained to imitate. Full coverage and full overlap are the same fact.

None of this invalidates a result, but it does mean improvements measured on this pool
will overstate what transfers. Held-out benchmark numbers carry the argument, and the
gap between the two is worth reporting rather than smoothing over.

### The validation sets are not interchangeable

| set | n | metric | AIME |
|---|---:|---|---|
| `folded/folded_val.parquet` | 628 rows / 200 questions | SFT token loss | no |
| `val_60.parquet` | 60 | GRPO task reward | yes (20) |
| `val_combined.parquet` | 50 | GRPO task reward | yes (20) |

`val_200` must stay i.i.d. with `train.parquet`, which is the only condition under which
val loss detects overfitting. AIME is deliberately excluded: token loss on AIME traces
measures teacher-imitation, not problem-solving, and mixing distributions makes a rising
curve uninterpretable. `val_60` is the one validation set with no teacher-solvability
selection bias, so it is the honest read on a run trained entirely on this pool.

## Provenance

- Teacher Qwen3-32B, thinking enabled, T=0.6 / top_p=0.95 / top_k=20; sub-agents Qwen3-1.7B greedy.
- `<think>` blocks and the collection-time `system_prompt_suffix` are stripped; the student trains with thinking disabled.
- One trajectory per question; tool-using traces preferred over tool-free.
- Every answer verified: about 90% exact match after normalisation, the rest individually reviewed (97.4% reviewer agreement on re-judged records). 158 trajectories with incorrect-but-passing answers excluded.
- All three files verified byte-identical to `build_sft_parquet.py` output.
- Source collection log: `data/training_30k/sft_collection_raw.jsonl` (12,195 records incl. failures, with `collection_run` provenance).
- Exclusion list `data/results/sft_exclude_final.json`; verdicts `data/results/sft_review_final.jsonl`.
- Held-out evaluation uses `data/training_30k/test/`, disjoint from everything here.

**None of the four paths named above currently exist in this repo** (`data/training_30k/`,
`sft_collection_raw.jsonl`, `data/results/sft_exclude_final.json`,
`data/results/sft_review_final.jsonl`), checked 2026-08-24. The parquets themselves are
intact and independently verified, but the audit trail behind the review, exclusion and
byte-identity claims cannot be re-run from here, and neither can `build_prefix_demos.py`'s
documented entry point, which is why the prep step above converts from the parquet
instead. Restore the collection log, or treat those four bullets as a record of work done
elsewhere rather than something reproducible.

## Known limitations

- 27.2% of train trajectories use no tool (the 32B answered from memory, mostly deepmath). An 8B cannot reproduce this, so these examples risk teaching unsupported guessing.
- 221 `code_generator` calls returned empty output (5.4% of trajectories). Genuine at collection time, not a processing artefact.
- The whole pool is teacher-solved, which biases SFT, GRPO and Prefix-RFT together. See "What using one pool for everything costs you".
- `extra_info.difficulty` is stored as `double` here against `int64` in `data/training/` (nulls force the promotion). No information is lost, but the dtype is not cosmetic: it makes these files un-concatenable with the older ones. Cast before mixing.
- `extra_info` field order also differs from `combined_train.parquet` (`golden_answers` and `difficulty` are swapped), so strict `pyarrow.concat_tables` fails on order alone. `promote_options="permissive"` and `pandas.concat` both handle it.
- One `val_200` trajectory is 18,717 tokens and exceeds a 16,384 `data.max_length` after folding. `--max-length` drops the offending row during prep.
- `extra_info.idx` is not unique (2830 distinct values over 3000 rows) and collides across data sources 170 times. Key on the question text, never on `idx`.
- 634 of 3000 trajectories hold a single decision and can never carry a Prefix-RFT prefix.
