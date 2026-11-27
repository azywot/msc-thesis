# Tool-call intervention (RSI phase 2) - runbook

Run this on Snellius from the repository root (`$HOME/azywot/msc-thesis`), in the
order below. The pipeline collects Qwen3-8B trajectories, lets a reflector change one
orchestrator tool call in the failed ones (rewrite a call, add a follow-up call, or
insert a call where none was made), checks that the fixes help, then trains two LoRA
adapters (arm C and arm I) and compares them.

- **Arm C**: SFT on the model's own correct trajectories only.
- **Arm I**: arm C plus new rows from rescued trajectories, starting at the fixed call.

Design: `docs/superpowers/specs/2026-11-17-tool-call-intervention-design.md`.

Every job needs `SERPER_API_KEY` or `TAVILY_API_KEY` in `.env`, and
`mkdir -p out/rsi` before the first `sbatch`. Use the `agent_engine` conda env for
every `python` command below.

## The three fix kinds

The reflector fixes exactly one orchestrator step in a failed trajectory. The fix is
one of:

- **rewrite**: replace a tool call the orchestrator made.
- **follow-up**: add a call after the last one the orchestrator made.
- **insert**: add a call when the orchestrator made none and went straight to an answer.

## 1. Collect trajectories (job 004)

```bash
mkdir -p out/rsi && sbatch jobs/rsi/004_collect_8b.job
```

One H100, up to 24 h; the full 1800 questions take about 5 h (about 8.5 min per
batch of 100). Three passes, each with its own seed (`config.seed + k`: 0, 1, 2).
Pass 1 runs every question; passes 2 and 3 rerun only the questions still unsolved,
so a question has three records only if it was wrong every time. Each question gets
its own domain's seed prompt (`deepmath` for math, `gaia` for `hotpotqa` and `nq`);
the job logs this at the start.

Output: `data/training/rsi_phase2/collect_8b/collected_<ts>.jsonl`, one record per
run, right or wrong, with the question, gold answer and aliases, seed, prediction,
plan, `action_history`, sub-agent calls and full messages. Progress (pass, batch and
each question's result) goes to `out/rsi/collect_8b_<jobid>.err`; check it with
`grep batch out/rsi/collect_8b_<jobid>.err | tail -1`.

## 2. Build arm C

```bash
python scripts/build_sft_parquet.py data/training/rsi_phase2/collect_8b/collected_*.jsonl \
    --output-dir data/training/rsi_phase2/arm_c \
    --output-name arm_c_folded_train.parquet \
    --format folded --collection-config '' \
    --reference-parquet data/training/train/combined_train.parquet \
    --max-length 16384
```

Output in `data/training/rsi_phase2/arm_c/`: `arm_c_folded_train.parquet` and
`arm_c_folded_val.parquet`.

## 3. Rescue with the 8B reflector (job 005)

```bash
sbatch jobs/rsi/005_rescue_8b.job
```

One H100, up to 24 h. It reads every `collected_*.jsonl`, takes the questions that
were wrong in all three passes, and asks the 8B reflector (thinking on) for one fixed
call: the step to fix, a sub-goal and the corrected `web_search` query or
`code_generator` task. The fix may use the question's own entities and numbers, but
not the gold answer. Each fix is injected and the orchestrator continues to an answer.
Alongside it, a no-fix control resumes the same question from the same step with the
same seed and nothing injected (two control draws per question). At most two attempts per question; the second is shown
the first fix and its wrong answer. Progress goes to `out/rsi/rescue_8b_<jobid>.err`.
Output in `experiments/results/rsi/phase2/rescue_8b/`: `candidates.json`,
`rescue.jsonl`, `gate.json`, `gate.md`.

Rows are appended to `rescue.jsonl` after every chunk of 100 candidates
(`rescue.chunk_size` in the config), so if the job is killed, resubmit it: it skips
the (question, attempt) pairs already on disk and loses at most one chunk. If the
reflector fails on every prompt of a chunk (for example the engine ran out of
memory, or the 32B server is down), the job stops with a `RuntimeError` before
writing that chunk's rows; fix the cause and resubmit.

Each row has a `status`: `ok` (the fix was injected and its continuation graded in
`correct`), `parse_error` (the reply had no usable fix), `leakage` (the fix was
rejected by the leakage guard) or `rescued_earlier` (the question was already
rescued at an earlier attempt, so this row holds only that attempt's no-fix control
draw). Every row carries a control draw in `control_correct`.

To redo only the report from an existing `rescue.jsonl`, run
`scripts/run_call_intervention.py` with `--report-only` and the same `--config`,
`--collected` and `--out-dir` the job uses. `--subset N` limits the run to the first N 0-of-3
questions in collected order for a quick trial (section 14).

## 4. The gate

Read `experiments/results/rsi/phase2/rescue_8b/gate.md` (the same numbers are in
`gate.json`; the job also prints `gate.md` at the end of its log).

- **Denominator**: every rate the gate uses is over the domain's 0-of-3 questions
  (`zero_of_n` in `gate.json`), not over the candidates. Questions skipped before
  the rescue (`skipped_error`, `skipped_planning_answer`) count as not rescued. The
  same rates over the candidates are reported as `rescue_rate_candidates`,
  `control_rate_candidates` and `net_points_candidates`; they do not decide the gate.
- **Rescue rate**: rescued questions over 0-of-3 questions, per domain (math and
  search), after leakage rejection.
- **No-fix control rate**: the same questions resumed from the same step with no
  injected call, once per attempt (two draws per question), whether or not the fix
  rescued the question. A question counts as control-solved if any of its control
  draws is correct. Some questions get solved by resampling alone.
- **Net rescue rate** = rescue rate minus no-fix control rate (`net_points` in
  `gate.json`).
- **Pass**: net rescue of at least 10 points in at least one domain, over its 0-of-3
  questions. `gate.md` starts with `# Gate: PASS` or `# Gate: FAIL` and lists the
  passing domains.

The per-domain table also reports attempts used, leakage rejections, parse errors
and the kind distribution (rewrite, follow-up, insert).

Check the `skipped_planning_answer` count in each domain. Runs whose planning turn
already gave the final answer have no tool-call step to fix, so they are skipped by
design. If that count is large relative to the 0-of-3 questions, raise it before
training, because it shrinks what arm I can learn from.

### If the gate fails

Run the 32B reflector before giving up (three H100s: the 8B orchestrator in process,
Qwen3-32B served on port 8001):

```bash
sbatch jobs/rsi/006_rescue_32b.job
```

Output in `experiments/results/rsi/phase2/rescue_32b/` (same four files). If the
32B gate also fails, stop and write up why the fixes fail. Do not build arm I.

From here on, use the `rescue_8b` or `rescue_32b` directory whose gate passed
(the commands below show `rescue_8b`).

## 5. Blind check

Rerun the reflector on a sample of rescued questions without the gold answer, to see
whether the fixes lean on hindsight. Submit the same job with `BLIND=1`:

```bash
sbatch --export=ALL,BLIND=1 jobs/rsi/005_rescue_8b.job
# or, if the 32B reflector passed the gate:
sbatch --export=ALL,BLIND=1 jobs/rsi/006_rescue_32b.job
```

This adds a `blind` section to `gate.json` and `gate.md` ("N of M previously rescued
questions rescued again; no-fix control: K of M") and writes `blind.jsonl`. The
control figure (`control_rescued`, `control_rate`) is the chance baseline: compare
the blind rate with it, not with zero. A large gap between the blind rate and the
rescue rate means the fixes depend on seeing the answer. It is reported, not a
pass/fail condition.

## 6. Build arm I

```bash
python scripts/build_arm_i_parquet.py \
    --arm-c-dir data/training/rsi_phase2/arm_c \
    --rescue experiments/results/rsi/phase2/rescue_8b/rescue.jsonl \
    --reference-parquet data/training/train/combined_train.parquet \
    --output-dir data/training/rsi_phase2/arm_i \
    --max-length 16384 --tokenizer Qwen/Qwen3-8B
```

Output in `data/training/rsi_phase2/arm_i/`: `arm_i_folded_train.parquet` (arm C
rows first, then the new rows), `arm_i_folded_val.parquet` (a byte copy of arm C's
validation file) and `arm_stats.json` (rows per arm, rescued questions, rows by
kind and domain, rows dropped for length). All rescued trajectories go in, including
ones the no-fix control would also have solved; the control only decides the gate.

## 7. Format check on both arms

The training job runs the check itself as a pre-flight, but running it first fails
fast, before a queue wait. `--native` is omitted because the arm files were not
folded from the native SFT parquet.

```bash
for arm in c i; do
  for split in train val; do
    python scripts/check_sft_folded_format.py \
        --folded data/training/rsi_phase2/arm_$arm/arm_${arm}_folded_$split.parquet \
        --max-length 16384
  done
done
```

## 8. Train both arms

```bash
sbatch --export=ALL,TRAIN_FILE=data/training/rsi_phase2/arm_c/arm_c_folded_train.parquet,VAL_FILE=data/training/rsi_phase2/arm_c/arm_c_folded_val.parquet,EXPERIMENT_NAME=qwen3-8b-rsi-arm-c,NATIVE_FILE=,NATIVE_VAL_FILE= jobs/fine_tuning/007_train_sft_folded.job

sbatch --export=ALL,TRAIN_FILE=data/training/rsi_phase2/arm_i/arm_i_folded_train.parquet,VAL_FILE=data/training/rsi_phase2/arm_i/arm_i_folded_val.parquet,EXPERIMENT_NAME=qwen3-8b-rsi-arm-i,NATIVE_FILE=,NATIVE_VAL_FILE= jobs/fine_tuning/007_train_sft_folded.job
```

**Confirm in the first arm's stderr file** that the pre-flight format check ran
without `--native`. Slurm must pass the empty `NATIVE_FILE=` and `NATIVE_VAL_FILE=`
through to the job. The job never prints the native path, and the check script logs
to stderr, so open `out/fine_tuning/sft_train/sft_folded_<jobid>.err` (not the
`.log`) and look at the pre-flight section. With `--native` the check logs
`Every native question present: ...` and `Trajectories N → rows M (...x)`. If
neither line appears, the empty values arrived and all is well. If either appears,
the default native files were used: cancel both jobs, because the check ran against
the wrong data.

### Where the adapter lands, and the run tag

Each job prints `Run tag: <dd-mm-YYYY_HH-MM>-<jobid>` near the start of its log. The
job trains under `/scratch-shared/$USER/fine_tuning/lora_adapters/<EXPERIMENT_NAME>/<run-tag>/`
and, if training finishes cleanly, copies `best_adapter` (and `last_adapter`,
`selection.json`) to durable storage:

```
data/adapters/<EXPERIMENT_NAME>/<run-tag>/best_adapter
```

That is `ARCHIVE_DIR` in the job. Use the archive path, not the scratch path:
scratch is purged. Get the run tag from the log, or from
`ls data/adapters/qwen3-8b-rsi-arm-c/` and `ls data/adapters/qwen3-8b-rsi-arm-i/`.

## 9. Generate the evaluation configs with the run tags

The adapter paths in `scripts/generate_configs.py` contain a literal `<run-tag>`
(`RSI_ARM_C_ADAPTER_PLACEHOLDER` and `RSI_ARM_I_ADAPTER_PLACEHOLDER`). Replace it in
each with the real run tag of that arm, then regenerate:

```bash
python scripts/generate_configs.py --suite rsi_arm_c_inference
python scripts/generate_configs.py --suite rsi_arm_i_inference
```

Commit the edited constants and the regenerated configs together, or
`tests/unit/test_wiring_invariants.py::test_the_generator_reproduces_the_committed_configs`
fails.

Configs land in `experiments/configs/qwen3/rsi_inference/arm_c/<dataset>/` and
`.../arm_i/<dataset>/`, one per benchmark (GAIA, HLE, GPQA, AIME, MuSiQue), named
`qwen8B_sub8B_none.yaml`. Check that `lora_adapter_path` in each contains the real tag
and no `<run-tag>`:

```bash
grep -rn "<run-tag>" experiments/configs/qwen3/rsi_inference/ || echo "no placeholders left"
```

What the configs do:

- The adapter is loaded on the **orchestrator only**. Both sub-agents stay base
  Qwen3-8B, because only orchestrator rows were trained.
- GAIA and HLE configs include the `text_inspector` sub-agent, like the base
  reference run, so all three columns (base, arm C, arm I) see the same tools. The
  adapter was trained without that tool.
- Thinking is off (`thinking_mode: "NO"`), seed prompts, `subset_num: -1`.

## 10. One-question GPU smoke run

Before the full evaluation, check that the adapter loads and one question runs end
to end. Use a copy of one arm C config with `subset_num: 1` and a scratch output
directory, so the smoke run never lands in the `rsi_inference` results tree
(`compare_arms.py` picks the newest run directory there).

```bash
mkdir -p out/rsi_smoke
sed 's/subset_num: -1/subset_num: 1/' \
    experiments/configs/qwen3/rsi_inference/arm_c/aime/qwen8B_sub8B_none.yaml \
    > out/rsi_smoke/arm_c_aime.yaml
srun --partition=gpu_h100 --gpus=2 --cpus-per-task=8 --time=01:00:00 \
    python scripts/run_experiment.py --config out/rsi_smoke/arm_c_aime.yaml \
    --output-dir out/rsi_smoke/arm_c_aime_run
```

(`srun` needs the conda env active and the module loads from the job scripts; if you
prefer, wrap the same `python` line in a short `sbatch` script.) Success is one
completed question and a `raw_results.json` in `out/rsi_smoke/arm_c_aime_run/<split>_<date>_<jobid>/` (the runner creates that run subdirectory under `--output-dir`). A
missing-adapter or `<run-tag>` error here means step 9 was not done.

## 11. Evaluate

```bash
./experiments/scripts/run_all_in_folder.sh experiments/configs/qwen3/rsi_inference/arm_c
./experiments/scripts/run_all_in_folder.sh experiments/configs/qwen3/rsi_inference/arm_i
```

Each config is one SLURM job (2 GPUs). Results land in
`experiments/results/rsi_inference/arm_c/<benchmark>/qwen8B_sub8B_none/<run>/raw_results.json`
and the same under `arm_i`.

The base reference is the existing AgentFlow run with base Qwen3-8B in every role:
`experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/<benchmark>/qwen8B_subagent_tools_none/`.
Compare with care on MuSiQue: that base run
(`musique/qwen8B_subagent_tools_none/validation_2026-03-15-22-09-22_*`) predates the
`validation_subset_200` split name, so its question set may differ from the arm runs.
The paired McNemar test is therefore arm I vs arm C only, never against base.

## 12. Compare

```bash
python scripts/compare_arms.py \
    --base-root experiments/results/1_milestone_no_img_no_mindmap_AgentFlow \
    --arm-c-root experiments/results/rsi_inference/arm_c \
    --arm-i-root experiments/results/rsi_inference/arm_i \
    --base-variant qwen8B_subagent_tools_none \
    --arm-variant qwen8B_sub8B_none \
    --out-dir experiments/results/rsi_inference/comparison
```

Writes `comparison.json` and `comparison.md` to `--out-dir`. The table has per
benchmark accuracy for base, arm C and arm I, the McNemar p (arm I vs arm C), and
behaviour shares (AIME: questions where `code_generator` was called; MuSiQue and
GAIA: questions with two or more `web_search` calls), then the five-benchmark mean.

**Success** is arm I's mean accuracy over GAIA, HLE, GPQA, AIME and MuSiQue being
above arm C's. The last line of `comparison.md` gives the verdict
(`arm I mean > arm C mean: True` or `False`). Per-benchmark p values are reported;
significance is not required for success. For the gain to be trusted, the behaviour
shares should move in the targeted direction (more code calls on AIME, more repeated
searches on MuSiQue and GAIA for arm I than for arm C).

## 13. Report with figures

After the gate (section 4), make the report from the rescue folder:

```bash
python scripts/report_intervention.py \
    --collected data/training/rsi_phase2/collect_8b/collected_*.jsonl \
    --rescue-dir experiments/results/rsi/phase2/rescue_8b \
    --out-dir experiments/results/rsi/phase2/report_8b
```

It writes `report.md` (figures with captions and the numbers behind them),
`report.json` and `figures/*.png` and `.pdf`: rescue against the no-fix control with
the gate threshold, the ceiling on the training questions, the funnel, fix kinds and
steps, the behaviour of rescued trajectories, the blind check (if `blind.jsonl`
exists) and a table of rescued examples (`--examples N`, default 8). The ceiling is
an upper bound on the training questions, not a benchmark gain. Single rates carry
95% Wilson intervals; the net rescue interval uses Newcombe's paired method, because
the fix and the control are measured on the same questions. The report refuses to
run if its counts disagree with `gate.json`.

After `compare_arms.py` (section 12), run it again with
`--comparison experiments/results/rsi_inference/comparison/comparison.json` to add the
benchmark accuracy and behaviour figures.

## 14. Trial run on a subset

To check the pipeline and get a first look before the full run, collect on the first
200 training questions, then rescue and report on them:

```bash
sbatch --export=ALL,SUBSET=200 jobs/rsi/004_collect_8b.job
sbatch --export=ALL,SUBSET=200 jobs/rsi/005_rescue_8b.job   # after 004 has finished
```

Collection writes to `data/training/rsi_phase2/collect_8b_subset200`, the rescue to
`experiments/results/rsi/phase2/rescue_8b_subset200`. The rescue keeps the first N
0-of-3 questions in collected order and computes `zero_of_n`, the skip counts and the
candidates on the questions up to the last of those, so the gate rates describe the
same questions. Job 006 takes the same `SUBSET`. Without `SUBSET` nothing changes.

```bash
python scripts/report_intervention.py \
    --collected data/training/rsi_phase2/collect_8b_subset200/collected_*.jsonl \
    --rescue-dir experiments/results/rsi/phase2/rescue_8b_subset200 \
    --out-dir experiments/results/rsi/phase2/report_8b_subset200
```

A subset this small gives wide intervals, so the trial checks the pipeline and gives a
first look; it is not the gate decision.

## Running the tests

`tests/rsi/test_run_prompt_search_cli.py` hangs in a checkout that has the 58G
`./cache` directory. Deselect it there (`--ignore` does not exclude it):

```bash
python -m pytest tests/unit tests/rsi --deselect tests/rsi/test_run_prompt_search_cli.py -p no:cacheprovider
```

## Output map

| Step | Output |
|---|---|
| 1 collect | `data/training/rsi_phase2/collect_8b/collected_<ts>.jsonl` |
| 2 arm C | `data/training/rsi_phase2/arm_c/arm_c_folded_{train,val}.parquet` |
| 3-5 rescue, gate, blind | `experiments/results/rsi/phase2/rescue_{8b,32b}/{candidates.json,rescue.jsonl,gate.json,gate.md,blind.jsonl}` |
| 6 arm I | `data/training/rsi_phase2/arm_i/{arm_i_folded_train.parquet,arm_i_folded_val.parquet,arm_stats.json}` |
| 8 adapters | `data/adapters/qwen3-8b-rsi-arm-{c,i}/<run-tag>/best_adapter` |
| 9 configs | `experiments/configs/qwen3/rsi_inference/arm_{c,i}/<dataset>/qwen8B_sub8B_none.yaml` |
| 11 evaluation | `experiments/results/rsi_inference/arm_{c,i}/<benchmark>/qwen8B_sub8B_none/<run>/` |
| 12 comparison | `experiments/results/rsi_inference/comparison/comparison.{json,md}` |
| 13 report | `experiments/results/rsi/phase2/report_8b/{report.md,report.json,figures/}` |

## Logs

Python progress messages go to each job's `.err` file; the `.log` file holds shell
and vLLM output.

| Job | Logs |
|---|---|
| 004 collect | `out/rsi/collect_8b_<jobid>.{log,err}` |
| 005 rescue (8B) | `out/rsi/rescue_8b_<jobid>.{log,err}` |
| 006 rescue (32B) | `out/rsi/rescue_32b_<jobid>.{log,err}` |
| 007 training | `out/fine_tuning/sft_train/sft_folded_<jobid>.{log,err}` |
| evaluation | `experiment.log` in each run folder |
