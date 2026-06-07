# Design: Failure Mode Analyzer Script

**Date:** 2026-05-12
**Target file:** `scripts/failure_modes/analyze_failure_modes.py`

---

## Purpose

Re-classify every failed MAS question-run from the 20 hard-coded run inventory entries into one of six mutually exclusive failure modes, then output a breakdown matching the thesis Table (failure mode × benchmark counts, share %, unique question count).

---

## Input

### Run inventory (hard-coded)

20 MAS runs (5 benchmarks × 4 thinking conditions) from `experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/`. Paths are resolved relative to a `--root` CLI argument (default: repo root, auto-detected from the script's own `__file__` location).

Each entry: `(benchmark_label, variant_label, relative_path_to_raw_results_json)`.

### Raw results record schema

Each record in a `raw_results.json` is a dict with:
- `question_id` (int)
- `question` (str) — full question text; used by Signal C for visual-keyword detection
- `correct` (bool)
- `prediction` (str, may be empty)
- `turns` (int)
- `action_history` (list of `{tool_name, sub_goal, command, result}`)
- `tool_counts` (dict: tool_name → int) — pre-computed; always agrees with action_history

Only records where `correct == False` are classified.

---

## Classification cascade

Rules are applied in priority order; the first matching rule wins.

| Priority | Mode key | Signal |
|---|---|---|
| 1 | `modality_tool_gap` | Any step has `tool_name` in `{video_analysis, image_inspector}` **OR** ≥ 2 `text_inspector` steps with ≥ 1 empty result AND ≥ 1 subgoal containing an image/visual keyword (`image`, `photo`, `picture`, `figure`, `diagram`, `video`, `screenshot`, `visual`, `illustration`) |
| 2 | `tool_loop_or_empty_final` | `prediction.strip() == ""` OR `turns >= 15` OR any single tool appears ≥ 3 times in `action_history` |
| 3 | `direct_reasoning_no_action` | `len(action_history) == 0` |
| 4 | `computational_subgoal_error` | `code_generator` appears **≥ 2 times** in `action_history` (multiple wrong computational sub-goals; a single code call in a multi-step trace falls through to single-shot) |
| 5 | `retrieval_evidence_failure` | `web_search` appears **≥ 2 times** in `action_history` (multiple failed searches; a single web_search blindly trusted is single-shot) |
| 6 | `single_shot_tool_trust` | catch-all |

---

## Output

### Console

Pretty-printed table identical in structure to the thesis table: rows = failure modes, columns = AIME / GAIA / GPQA / HLE / MuSiQue / Total / Share% / Unique questions. Printed to stdout.

### `breakdown.json`

```json
{
  "failure_modes": {
    "<mode_key>": {
      "label": "<human label>",
      "benchmarks": {
        "<benchmark>": {
          "count": 42,
          "question_ids": [1, 5, 7, ...]
        }
      },
      "total": 42,
      "share_pct": 16.6,
      "unique_questions": 30
    }
  },
  "totals": { "<benchmark>": <int>, "overall": 2534 }
}
```

### `breakdown.csv`

One row per failure mode, columns: `failure_mode`, `aime`, `gaia`, `gpqa`, `hle`, `musique`, `total`, `share_pct`, `unique_questions`.

Output files are written to `--output-dir` (default: `scripts/failure_modes/`).

---

## CLI

```
python scripts/failure_modes/analyze_failure_modes.py [--root REPO_ROOT] [--output-dir DIR]
```

---

## Error handling

- Missing `raw_results.json`: warn and skip that run entry.
- Malformed record (missing keys): skip with a warning, counted under a `"parse_error"` key in the JSON.
- All records correct in a file: skip silently (no failures to classify).

---

## Module structure

All logic in named functions; no global state. Functions:

- `resolve_root(root_arg)` → `Path`
- `load_run(path)` → `list[dict]`
- `classify_failure(record)` → `str` (mode key)
- `run_inventory(root)` → `list[tuple[str, str, Path]]`
- `analyze(root, output_dir)` → writes files, prints table
- `main()` → argparse entry point
