# RSI pipeline - prompt search (phase 1)

Recursive self-improvement (RSI) is a further fine-tuning direction, compared
with the other fine-tuning directions (SFT, GRPO, LoRA). One cycle has two
steps: a **search** that improves the prompts the agent team runs under, then
an **internalisation** step that trains the model on trajectories produced
under the improved prompts. This page documents the first step only.

**Phase 1 is a MASS-style prompt search.** Qwen3-8B plays every role:
orchestrator, `web_search` and `code_generator` sub-agents, and reflector. No
weights change. The reflector reads trajectories of the current best prompts
and proposes one edit at a time; an edit is kept only if it passes screening
and a strict held-out check.

**Phase 2 (trajectory collection under the improved prompts, then fine-tuning)
is not implemented yet.** There is no code for it in this repository.

Design: `docs/superpowers/specs/2026-11-04-recursive-self-improvement-design.md`.
Plan: `docs/superpowers/plans/2026-11-08-rsi-phase1-prompt-search.md`.
Code: `src/rsi/`, entry point `scripts/run_prompt_search.py`.

---

## Two reflector setups

| Setup | Config | Job | GPUs |
|---|---|---|---|
| **Default**: Qwen3-8B reflects on its own engine. The reflector calls the executor's in-process vLLM engine with its own sampling parameters (per-candidate seed, reflector temperature and token limit), so the executor settings are untouched. | `search_reflector_qwen3_8b.yaml` | `001_prompt_search_8b.job` | 1 H100 |
| **Control**: Qwen3-32B reflects over a separate `vllm serve` (port 8001, tensor parallel 2). The executor is still Qwen3-8B. | `search_reflector_qwen3_32b.yaml` | `002_prompt_search_32b.job` | 3 H100 |

The control tells you whether a weak 8B reflector was the limiting factor. The
`in_process` backend refuses to start unless `search.reflector.model` equals
the orchestrator model.

Executor setup in both configs: tools `web_search` and `code_generator`,
`direct_tool_call: false`, `max_turns: 7`, `thinking_mode: "NO"`, no
per-tool model entries (every tool uses the orchestrator model).

## The prompt bundle

A **bundle** (`src/rsi/bundle.py`) is the plain-data object the reflector
edits. `bundles/v0.json` is the seed bundle and renders byte-identically to
today's prompts.

Editable units (a proposal edits exactly one unit):

| Unit | Editable parts |
|---|---|
| `orchestrator` | `guidance_opening`, `guidance_closing`, `example` (the worked example), `turn0_prompt`, `turn1n_prompt`, `tool_descriptions` (description strings only), `tool_use_note`. A proposal gives a partial dict of just the parts it changes. |
| `web_search` | The analysis template (full replacement string). |
| `code_generator` | The task template (full replacement string). |

Protected, and enforced by validation before any rollout: tool names,
parameter names, types and `required` lists; the `<tools>` header and the
`<sub_goal>`/`<tool_call>` format paragraph; the `\boxed{}` final answer rule;
attachment notes; the memory-turn structure. Sub-agent templates must keep
exactly their required placeholders. Worked examples must call only enabled
tools with valid arguments and end with a `\boxed{` answer. `text_inspector`
stays at the seed.

## Data

Only `data/training/train/combined_train.parquet` is used (1,800 rows:
hotpotqa 765, nq 135, deepmath 900). No benchmark data is touched.

| `--domain` | Rows | Prompt template |
|---|---|---|
| `search` | hotpotqa + nq | `gaia` |
| `math` | deepmath | `deepmath` |

Each domain is split, stratified by data source and seeded by `pool_seed`,
into three disjoint pools:

| Pool | Size | Use |
|---|---|---|
| shown | 300 | Source of the minibatches the reflector sees (with gold answers). |
| screen | 200 | Cheap screening of candidates. |
| held-out | 50 | Strict promotion test. |

The remaining 350 rows per domain are in no pool and are used only by
`--confirm`. Pools are written to `pools.json`; a run directory refuses to
resume with different pools.

## One round

Rounds run `rounds` times (default 10). Round `r`:

1. **Shown minibatch.** The champion runs on 16 shown questions. One seeded
   permutation of the shown pool is cut into disjoint minibatches, so rounds
   within a pass do not repeat questions; when the pool is used up it is
   permuted again. The draw is identical on a resumed run.
2. **Reflector input.** The reflector sees the current and seed prompts, the
   rendered system prompt (read-only), the trajectories with sub-agent inputs
   and outputs, a history of earlier edits (up to `history_max`, halved if the
   input is over budget), and short reasons why last round's proposals were
   rejected. It is told to write general strategies and not copy questions,
   entities, numbers or answers. The input is fitted to
   `input_budget_tokens`; every cut is recorded in `reflector_stats`. If it
   cannot fit even without history, the round is skipped (`prompt_over_budget`).
3. **K proposals.** `candidates_per_round` (3) proposals are sampled with
   per-slot seeds. A slot that returns an error, empty text, a truncated
   output, unparseable JSON or an unknown unit is re-sampled up to
   `max_parse_retries` (2) times. A round with no usable proposal keeps the
   champion.
4. **Validation.** Each proposal is checked against the protected parts above
   and rejected with a named reason before any rollout. Partial
   `tool_descriptions` are completed from the champion.
5. **Leakage guard** (`src/rsi/leakage.py`), on every editable string of the
   candidate. A candidate is rejected if it contains:
   - an 8-token n-gram (`ngram`) copied from any question in the training
     parquet; or
   - a gold answer (or alias) of any question shown so far, matched as a token
     sequence after answer normalisation.

   Two rules keep the guard usable on real data. Answers of 3 characters or
   fewer (ignoring spaces) and the words yes, no, true and false are exempt,
   because many answers are such tokens ("0", "1", "b") and banning them would
   reject almost any candidate. And the check is **champion-relative**: an error
   that the current champion already triggers is ignored, so only newly
   introduced leaks reject a candidate. The question n-gram rule is unaffected
   by the exemption.
6. **Screen.** Surviving candidates and the champion run on the same 16
   questions drawn from the screen pool. The best candidate by screen accuracy
   (ties go to the lower slot) moves on only if it scores at least the
   champion's screen score.
7. **Strict held-out promotion.** That one candidate runs on the 50 held-out
   questions and gets a version label (`v1`, `v2`, ...). It becomes the new
   champion only if its held-out accuracy is **strictly greater** than the
   champion's. Ties keep the old champion.

Every round is written to disk, so a killed job loses at most the round in
progress.

## Running it

Dry run (no GPU, no model; builds tool schemas with a stub, makes the pools
and `v0.json`, prints pool sizes):

```bash
python scripts/run_prompt_search.py \
  --config experiments/configs/rsi/phase1/search_reflector_qwen3_8b.yaml \
  --domain search --dry-run
```

SLURM (the `mkdir -p out/rsi` must come first; the job log directory has to
exist before submission):

```bash
# 001: 8B reflector on one GPU (24 h). DOMAIN=search or DOMAIN=math.
mkdir -p out/rsi && sbatch --export=ALL,DOMAIN=search jobs/rsi/001_prompt_search_8b.job
mkdir -p out/rsi && sbatch --export=ALL,DOMAIN=math   jobs/rsi/001_prompt_search_8b.job

# 002: 32B reflector control on three GPUs (24 h)
mkdir -p out/rsi && sbatch --export=ALL,DOMAIN=search jobs/rsi/002_prompt_search_32b.job

# 003: confirmation of a finished 8B run on 200 unused questions (8 h, one GPU)
mkdir -p out/rsi && sbatch --export=ALL,DOMAIN=search jobs/rsi/003_confirm_8b.job
```

Details:

- **Resume** by resubmitting the same command. The run name defaults to the
  config file stem, so the same config and domain land in the same directory
  and continue from the last completed round. The rollout counter continues.
  A non-default name: `--export=ALL,DOMAIN=search,EXTRA_ARGS="--run-name NAME"`.
- **Config changes between resumes** are refused (`config.json` must match,
  except `rounds` and `output_root`), so extending `rounds` is allowed and
  nothing else is.
- **Report only** (no GPU, regenerates `report.md` from files on disk):
  `python scripts/run_prompt_search.py --config <yaml> --domain search --report-only`
- **Confirm** (needs a finished search): `--confirm N` scores v0 and the
  champion on N questions outside every pool, writes `confirm.json`, and
  regenerates the report. Job 003 uses N = 200; change it with `CONFIRM_N`.
  Confirm the 32B-reflector run on the same single-GPU job by overriding the
  config: `sbatch --export=ALL,DOMAIN=search,CONFIG=experiments/configs/rsi/phase1/search_reflector_qwen3_32b.yaml jobs/rsi/003_confirm_8b.job`.
  Resubmitting reuses `confirm/<label>.json` files written for the same
  questions.
- `--dry-run`, `--report-only` and `--confirm` are mutually exclusive.
- Jobs need `SERPER_API_KEY` or `TAVILY_API_KEY` in `.env` for web search.

## Outputs

Under `experiments/results/rsi/phase1/<run_name>/<domain>/`:

| Path | Contents |
|---|---|
| `config.json`, `pools.json` | Search settings and pool question ids for this run. |
| `state.json` | Resume state: rounds done, champion, seed and champion held-out accuracy, edit history, version table, rollout count, `beats_seed`. |
| `bundles/v0.json`, `bundles/v<N>.json` | The seed bundle and every candidate that reached the held-out check. |
| `heldout/v<N>.json` | Per-question held-out trajectories for each of those versions. |
| `rounds/round_NN.json` | Full round record: minibatch ids, raw reflector outputs, errors, leakage errors, screen scores, chosen candidate, held-out score, accepted. |
| `rounds/round_NN_reflector_input.json` | The exact messages sent to the reflector, the shown trajectories and the history size. Written before sampling, so a job killed mid-sample still leaves its input. |
| `confirm.json`, `confirm/v<N>.json` | Confirmation scores, paired counts and per-question records (after `--confirm`). |
| `report.md` | Human-readable report (see below). |

## Reading `report.md`

The report has a summary (domain, reflector, rounds done and skipped,
rollouts, seed and champion held-out accuracy, `beats_seed`, confirmation),
a search curve per round with links to the record and reflector input, a table
of rejection reasons, the units edited, diffs of seed against champion, the
rendered champion prompt, and paired seed-versus-champion trajectories.

**`beats_seed` is optimistic.** The champion is chosen by its score on the 50
held-out questions, and `beats_seed` is measured on those same questions, so
the winner's curse inflates it. With 50 questions one question is two points.
Treat it as "worth confirming", not as a result. Run `--confirm` and report the
confirm accuracies and the paired counts (both right, seed only, champion only,
neither). The report flags a confirmation as stale if the champion changed
after it ran.

## Config fields

The YAML is a normal `ExperimentConfig` plus a `search:` block
(`SearchConfig` in `src/rsi/search.py`). The script requires
`tools.direct_tool_call: false` and `baseline: false`.

| Field | Default | Meaning |
|---|---|---|
| `data_path` | required | Training parquet. |
| `pools` | `{shown: 300, screen: 200, heldout: 50}` | Pool sizes. |
| `pool_seed` | `0` | Seed for the pool split and the confirm sample. |
| `rounds` | `10` | Search rounds. |
| `candidates_per_round` | `3` | K proposals per round. |
| `shown_minibatch`, `screen_minibatch` | `16`, `16` | Questions per round on the shown and screen pools. |
| `max_parse_retries` | `2` | Re-samples of a failed proposal slot. |
| `history_max` | `8` | Past edits shown to the reflector. |
| `ngram` | `8` | Leakage n-gram length. |
| `seed` | `0` | Seeds minibatches and proposal sampling. |
| `output_root` | `experiments/results/rsi/phase1` | Output root. |
| `reflector` | required | See below. |

`search.reflector`:

| Field | Default | Meaning |
|---|---|---|
| `backend` | `in_process` | `in_process` (executor's engine) or `openai` (needs `base_url`). |
| `model` | required | Model id; also used for the token counter. |
| `base_url` | none | e.g. `http://localhost:8001/v1` for the 32B control. |
| `max_model_len` | `32768` | Context length. |
| `input_budget_tokens` | `14000` | Reflector input budget. |
| `max_tokens` | `16384` | Reflector output limit. |
| `per_call_chars` | `1500` | Cap on each tool-call excerpt in the input. |
| `temperature`, `top_p`, `top_k` | `0.6`, `0.95`, `20` | Reflector sampling. |
| `enable_thinking` | `true` | Reflector thinking. |
| `timeout_s` | `900` | Request timeout (`openai` backend). |

Input budget plus output limit (14,000 + 16,384) must fit within
`max_model_len` (32,768); config validation rejects anything larger. The 32B
server is started with `--max-model-len 40960` by the job body.

## Evaluating a bundle on a benchmark (optional diagnostic)

```yaml
prompt_bundle_path: experiments/results/rsi/phase1/<run>/<domain>/bundles/v<N>.json
```

Any inference config can render its prompts from a bundle, including the
sub-agent templates. Constraints:

- the benchmark's prompt template must match the bundle's domain template
  (`gaia` for `search`, `deepmath` for `math`), otherwise the run refuses to
  start because the benchmark's final instructions would be lost;
- `tools.direct_tool_call` and `baseline` must be false;
- it is mutually exclusive with `gepa_prompt_path`;
- the bundle is copied into the run directory as `prompt_bundle.json`;
- when the key is unset, nothing changes, including config fingerprints of
  existing configs.

This is a diagnostic. The planned comparison evaluates the trained models under
the seed prompts, not under searched prompts.

## Known limitations

- **The GPU path has not been run.** The search, validation, leakage, report
  and confirm logic is covered by tests with fake executors and reflectors;
  nothing has run against real vLLM yet.
- **SIGTERM handling.** The script turns SIGTERM into a clean exit so caches
  are saved, but vLLM may install its own handler and override it. If a job is
  killed hard, the web caches may not be flushed (they are also saved after
  every executor run). Resume still works because every round is on disk.
- **Pre-download `Qwen/Qwen3-32B` before submitting job 002.** A 32B download
  inside the job would eat into the 15 minute startup limit of the reflector server.
- **The 24 h wall-clock fit is untested.** 10 rounds with a thinking reflector
  and 50-question held-out runs may or may not fit; resume covers an overrun.
- **Search and held-out noise.** Screening uses 16 questions and held-out 50,
  so most accepted gains are within noise until confirmed.
