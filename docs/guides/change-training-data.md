# Change the training data (RL / GEPA)

This is not one seam - it is two, with different amounts of work behind them.
Read the first section before the second: most changes people want ("more
math, less search", "drop NQ", "harder DeepMath only") are the cheap one.

Applies to both data-prep scripts, which share the same download/normalise
helpers:

| Script | Feeds | Sources |
|---|---|---|
| `src/fine_tuning/data/prepare.py` | RL (GRPO) train/val/test parquets | Search-R1 (HotpotQA + NQ), DeepMath-103K, local AIME jsonl (val-only) |
| `src/gepa_integration/data/prepare.py` | GEPA `D_feedback`/`D_pareto`/test splits | same two HF sources, via `_PRESETS` |

---

## 1. Reweight or filter the existing sources - a CLI flag, no code

Both scripts already parameterise the composition. Nothing below touches
`agent_engine`, the orchestrator, or the reward/scorer path - it only changes
which raw rows get sampled before normalisation.

**RL data** (`src/fine_tuning/data/prepare.py`):

```bash
python src/fine_tuning/data/prepare.py \
    --n-search 900 --n-math 900 \
    --search-source hotpotqa \        # hotpotqa | nq | both
    --hotpot-ratio 0.85 \             # only used when --search-source=both
    --deepmath-min-difficulty 5 \     # 1-9; higher = harder-only
    --n-val-search 20 --n-val-math 10 --n-val-aime 20 \
    --n-test-search 100 --n-test-math 100 \
    --output-dir data/training --seed 42
```

Every one of those is read at `main()` time and passed straight into
`_download_search_r1` / `_download_deepmath` - see `src/fine_tuning/data/prepare.py:652-786`
for the full flag list, including the val/test counts. Changing the search/math
*ratio* is `--n-search` vs `--n-math`; changing DeepMath's difficulty floor is
`--deepmath-min-difficulty`; dropping NQ entirely is `--search-source hotpotqa`.

**GEPA data** (`src/gepa_integration/data/prepare.py`) goes one step further:
the mix is a named preset, not loose flags, because GEPA ties the mix to which
benchmark's failure modes it's targeting:

```python
_PRESETS: dict[str, dict] = {
    "gaia": {
        "search": {"feedback": 112, "pareto": 37, "test": 75},
        "math":   {"feedback": 38,  "pareto": 13, "test": 25},
        "deepmath_min_difficulty": 1,
        "hotpot_ratio": 0.85,
        "description": "75% Search-R1 (85/15 HotpotQA/NQ) + 25% DeepMath (no difficulty filter)",
    },
    "math": { ... },
}
```

A third preset - say, a GPQA-flavoured mix - is a new dict entry with its own
`search`/`math` split-size triples and `deepmath_min_difficulty`/`hotpot_ratio`,
picked up automatically by `--preset` (`argparse` reads `choices=list(_PRESETS)`).
No other code changes: `prepare()` (`src/gepa_integration/data/prepare.py:194`)
is generic over whatever preset it's handed.

In both cases, re-running the script overwrites the parquet/JSON outputs in
place - re-point `--output-dir` if you want to keep the old mix around for
comparison.

---

## 2. Add a genuinely different source - this requires code, not config

If what you want isn't a different weighting of Search-R1/DeepMath but a
**different dataset entirely** (a different retrieval corpus, a different math
set, a different domain altogether), there is no flag for that. Both scripts
are hardcoded to two HF datasets plus a local AIME jsonl. Wiring in a new one
means writing the two pieces every existing source has:

1. **A row normaliser** - raw HF/jsonl schema → the shared VERL row shape
   (`data_source`, `question`, `result`, `extra_info`). Follow
   `normalise_deepmath_row` (`src/fine_tuning/data/prepare.py:140`) as the
   template - note it tries several fallback field names (`final_answer` /
   `answer` / `problem`), which is defensive against schema drift on the Hub,
   not required boilerplate.
2. **A download/split function** - streams or loads the source, applies
   test-then-val-then-train ordering (**this order matters**: it's what
   guarantees the three splits never overlap regardless of how large
   `n_train` is - see `_download_search_r1`'s docstring), and raises loudly if
   it can't fill a quota rather than silently returning fewer rows.

Then wire it into `main()` (RL) or into a new `_PRESETS` entry plus a branch in
`prepare()` (GEPA) alongside the existing `_download_search_r1` /
`_download_deepmath` calls.

`REQUIRED_COLS = {"data_source", "question", "result", "extra_info"}`
(`src/fine_tuning/data/prepare.py:300`) is checked by `validate_parquet_schema`
on every write - get the normaliser's output dict right and the rest of the
pipeline (VERL loading, the reward function, GEPA's example builder) doesn't
care where the row came from.

**This is real work, not a declarative extension point** - unlike adding a
benchmark (`DATASET_SPECS` row) or a tool (`@register_tool`), there is no
table to add a line to. Budget for: a normaliser, a downloader, a `main()`/
preset wire-up, and rerunning both data-prep and the RL/GEPA smoke jobs
(`jobs/fine_tuning/004_smoke_8b.job`, `jobs/gepa/003_smoke_gepa_gpu.job`) before a
full run, since a malformed `extra_info` or empty `result` fails far into
training rather than at prep time.

## See also

- [pipelines/rl.md](../pipelines/rl.md) - what the RL data feeds into
- [pipelines/gepa.md](../pipelines/gepa.md) - what the GEPA data feeds into
- [add-an-adaptation-method.md](add-an-adaptation-method.md) - swapping the
  *method* (GEPA/SFT/RL) rather than the data underneath one
