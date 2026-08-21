# Prefix-RFT pipeline - blending demonstration and exploration into Flow GRPO

> **Prefix-RFT runs on Flow GRPO, not on plain single-turn GRPO.** It inherits the whole
> RL pipeline this project already had: a trajectory becomes one batch row per *turn*,
> every turn of a trajectory shares the question's `uid`, and the final sparse reward is
> propagated to all of them (`src/fine_tuning/rollout.py:333`). Prefix-RFT changes what
> one rollout in eight *starts from*; it does not change how credit is assigned across
> turns. Read every metric on this page as a per-turn quantity.
>
> "GRPO" in `algorithm.adv_estimator: grpo` is the advantage estimator sitting on top of
> that layout, which is exactly what the plain GRPO baseline uses too. Flow GRPO decides
> which turns receive the reward; GRPO normalises it (`src/fine_tuning/README.md:275`).
> The one place Prefix-RFT departs is group statistics on prefixed questions - see
> divergence 9 below.

Prefix-RFT is Flow GRPO in which one rollout per prompt does not start from scratch: it
is seeded with a prefix of a Qwen3-32B teacher demonstration, and the policy writes the
continuation. The composite trajectory is scored and trained on like any other rollout,
so the demonstration is reinforced only in proportion to how much it actually helped -
unlike SFT, there is no separate imitation loss. Entropy-based clipping keeps only the
top 20% of prefix tokens in the gradient, because the paper shows unclipped prefix
gradients dominate the update while making up a small fraction of the batch.

**The reference document is [`docs/superpowers/specs/2026-08-17-prefix-rft-design.md`](../superpowers/specs/2026-08-17-prefix-rft-design.md)**
- the full paper mapping, the multi-turn adaptation, every hyperparameter's source, and
the "Implementation record" of what changed while building it. This page is orientation.
For the RL machinery Prefix-RFT reuses (the orchestrator-inside-the-training-loop
pattern, GPU layout, checkpoint handling), see [rl.md](rl.md) - Prefix-RFT does not
duplicate any of it.

> **Status (2026-08-18): the mechanism is verified on GPU at two scales.**
> `011_tiny_prefix_rft.job` passes all five checks (run 25755605) and
> `010_smoke_prefix_rft.job` passes on 8 questions (run 25756950): teacher text replayed
> verbatim in exactly one rollout per question then continued on-policy, validation
> on-policy, prefix tokens in the loss (849 and 816), the entropy clip keeping 20.3% and
> 20.1% against the paper's 20%, replay tokenisation matching the proxy's, and LoRA
> adapters written.
>
> `012_capped_prefix_rft.job` closes the remaining gap (run 25762046): 9 of 10 steps at
> the production batch shape (`rollout.n` 8, batch 32, 4 GPUs) before the 12 h wall, with
> `low_t` decaying 0.928 -> 0.072 along the closed-form cosine, mean `k` falling 2.42 ->
> 1.68 between the run's halves, `off_ratio` at most 0.067, and the entropy clip keeping
> 20.13-20.29% of prefix tokens at every single step. Gradients (0.078-0.154) and entropy
> (0.215-0.258) were flat; `kl_loss` rose 0.007 -> 0.155, small in absolute terms but the
> one metric to watch in `013`. Checkpoint written at `global_step_5`.
>
> **The mechanism and the curriculum are both verified. Nothing blocks `013`.**

---

## Two ways to measure a prefix - the thing to know before reading a metric

The paper's test bed is single-turn math: one prompt, one response, one row in the batch.
Ours is Flow GRPO, where a trajectory is many turns and each turn is its own row. That
difference is what this whole section is about. CoSMAS trajectories are multi-turn - a
planning decision, one or more tool calls, a synthesis decision - so "a prefix of the
demonstration" cannot be a token fraction of one response. It is an integer number of
teacher **decisions**: for a demonstration with `m` decisions, the
first `k` are replayed verbatim and the model takes over from decision `k+1`.
`k = floor(l * m)`, clamped to `[0, m-1]` so there is always at least one on-policy
decision to score, with `l` drawn from the paper's schedule unchanged.

That is **step mode**, `prefix_rft.mode: steps`, and it is the default.

**Two consequences of measuring in decisions:**

1. **`k` is a staircase, not a continuous ratio.** Teacher trajectories average 2.98
   decisions, so the paper's continuous cosine decay is quantised into roughly three
   levels over training.
2. **Single-decision demonstrations never carry a prefix.** 273 of the 1358 demonstrated
   questions have exactly one decision, so `k <= m-1 = 0` always and they train as pure
   GRPO. 1085 of the 1800 training questions are prefixable.

### Token mode: the paper's own measure

`prefix_rft.mode: tokens` measures the prefix in tokens instead, which is what the paper
does. Both modes share `sample_l` unchanged, so both inherit the same curriculum. The fraction is taken over the concatenation of every decision's response, so with
decision token lengths `n_1..n_m` summing to `T`:

```
B = clamp(floor(l * T), 0, T - 1)    token budget for this rollout
j = how many whole decisions fit in B
r = B - (n_1 + ... + n_j)            leftover tokens
```

Decisions `1..j` are replayed whole; if `r > 0`, decision `j+1` is **split** and the
model finishes the turn the teacher started. The `T - 1` clamp is the paper's
`prefix_len <= demo_len - 1` guard applied to the concatenation, so at least one token
is always generated. A split turn's `prefix_mask` row is 1 on its first `r` tokens and 0
after, which is why a turn is no longer wholly teacher or wholly policy.

```
y* = [resp_1 | resp_2 | resp_3]        teacher, m = 3, T = 120 tokens

l = 0.80   B = 96    ####|####|##~~~~     j = 2, r = 16, decision 3 split
l = 0.35   B = 42    ####|##~~|~~~~~~     j = 1, r = 12, decision 2 split
l = 0.00   B = 0     ~~~~|~~~~|~~~~~~     j = 0, r = 0,  plain GRPO
```

The split turn is the one replayed turn that reaches vLLM, through
`continue_final_message`, which leaves the final assistant message open-ended. Its token
IDs are rebuilt locally afterwards, so the vendored daemon is untouched.

**The ceiling differs too.** Step mode is clamped to `m-1` decisions, so on the average
2.98-decision demonstration it can never replay more than about 67% of the teacher,
while token mode reaches the schedule's full 95%. Measured against the real store at
step 0: step mode replays a mean 2.51 decisions, a 0.70 fraction of the demonstration,
against token mode's 0.95. Step mode is therefore systematically *less*
demonstration-heavy early in training, which is the opposite of what "the same
curriculum, quantised" would suggest.

**Read these two things into any comparison of the modes:**

1. **Coverage is not the same by default, and the shipped config removes the
   difference.** Token mode's guard is a token guard, so a single-decision
   demonstration can be split partway through its only response. That would take
   prefixable coverage from 1085 questions to all 1358, giving any steps-vs-tokens
   difference a rival explanation. `config_prefix_rft_tokens.yaml` therefore sets
   `prefix_rft.min_demo_decisions: 2`, which makes token mode skip exactly the
   questions step mode cannot reach, so both prefix the same 1085. This gives up a
   real capability of token mode for the sake of a controlled comparison; a run that
   wants the paper's full coverage sets it to 1 and reports the confound. The trainer
   prints the eligible count at startup, so the two runs can be checked against each
   other.
2. **A tool call can be split mid-JSON.** The model may then complete it into something
   malformed, which will show up as depressed `actor/reward_with_prefix` in the early
   steps when `l` is near 0.95. That is a finding about mid-turn prefixes in an agentic
   setting, not a defect to patch out.

Run it either way:

```bash
# tiny GPU check of the split path, ~10 min on 3 GPUs (do this first)
sbatch jobs/fine_tuning/014_tiny_prefix_rft_tokens.job

# production: 013 takes its config from PREFIX_CONFIG, defaulting to step mode
PREFIX_CONFIG=experiments/configs/fine_tuning/config_prefix_rft_tokens.yaml \
  sbatch jobs/fine_tuning/013_train_prefix_rft.job

# or override the mode on any prefix config, from the driver
python scripts/launch_verl.py --config <config.yaml> --prefix-mode tokens
```

The flag is on `launch_verl.py` alone. The driver owns the schedule and dispatches
`prefix_k` in step mode or `prefix_l` in token mode; the rollout workers read the mode
off whichever key arrives, so there is no second setting that could disagree with the
first. Passing both keys is a hard error.

Replayed decisions are **supervised, not teacher-forced context**: they enter the loss
as response tokens with `prefix_mask = 1`, carrying the trajectory advantage under the
entropy filter. This is load-bearing, not an implementation detail - the paper's Table 8
scores excluding the prefix from the loss at 33.8 against 51.8 for the version that
trains on it.

---

## Before you start

Prefix-RFT does not start from raw data. It needs a **teacher trajectory collection**,
and that collection is not part of this pipeline - it is the same one the SFT pipeline
uses, produced by `006_collect_sft_data.job` (Qwen3-32B running the orchestrator, 4 GPUs,
up to 60 h). `008` reads the newest `data/training/sft/collected_*.jsonl` and will refuse
to run without one:

```
ERROR: no data/training/sft/collected_*.jsonl found.
```

**If you have already run the SFT pipeline, you have this file and can start at 008.**
If not, run `006` first; nothing else here will work.

You also need:

- **Two conda environments.** `agent_engine` (has pytest, no verl) and `cosmas-train`
  (has verl and vLLM, no pytest). `009` uses both, which is why the logic lives in
  verl-free modules - see "How it hooks into the framework".
- **`.env` keys.** `SERPER_API_KEY` or `TAVILY_API_KEY` (the rollout workers call real
  tools and every job refuses to start without one) and `WANDB_API_KEY` for the prefix
  curves. `HF_TOKEN` if the models are not already cached.
- **`data/training/{train,val}`** readable. If a job dies reading a parquet it can see,
  check the directory has its execute bit (`chmod u+x`); read-without-execute makes a
  directory untraversable even by its owner.

What each stage costs:

| Stage | Partition | GPUs | Wall clock |
|---|---|---|---|
| `006` collect teacher trajectories (prerequisite, shared with SFT) | gpu_h100 | 4 | up to 60 h |
| `008` build the demonstration store | genoa | 0 | minutes |
| `009` CPU verification suite | genoa | 0 | ~5 min |
| `011` mechanism on 2 questions | gpu_h100 | 3 | ~10 min |
| `010` smoke on 8 questions | gpu_h100 | 3 | ~25 min |
| `012` capped production, 10 steps | gpu_h100 | 4 | ~12 h (measured 71 min/step) |
| `013` production | gpu_h100 | 4 | ~66 h of a 72 h wall |

Everything from `008` down is cheap until `012`. Run them in order; each one is designed
to fail faster than the one after it.

## Running it

```bash
# 1. Build the demonstration store from the teacher trajectory collection
#    produced by 006_collect_sft_data.job (see "Before you start").
sbatch jobs/fine_tuning/008_build_prefix_demos.job

# 2. Verify on the CPU partition: unit tests, the store gate, replay
#    tokenisation against the real tokenizer, a trip-wire, and the
#    cosmas-train sync/import checks. No GPU cost.
sbatch jobs/fine_tuning/009_run_tests_for_prefix_rft.job

# 3. Two questions, one optimiser step, 3 GPUs, ~10 min. Asks the narrow
#    question "does a teacher prefix actually get replayed, masked,
#    advantage-corrected and entropy-clipped on real hardware?" Run this
#    before 010 - it fails in minutes where 010 fails in hours.
sbatch jobs/fine_tuning/011_tiny_prefix_rft.job

# 4. Smoke test: 8 questions, 8B on 2 GPUs, asserts the prefix machinery was
#    actually active (not just that training completed).
sbatch jobs/fine_tuning/010_smoke_prefix_rft.job

# 5. The production run, stopped after 10 steps (~7 h on 4 GPUs). The only
#    thing that exercises the CURRICULUM - low_t falling 0.95 -> 0.05 and mean
#    prefix length falling with it - and the production batch shape
#    (rollout.n: 8, train_batch_size: 32, TOOL_STEPS: 5).
sbatch jobs/fine_tuning/012_capped_prefix_rft.job

# 6. The production run: 1800 questions x 2 epochs, ~112 steps, 72 h on 4 GPUs.
#    Only after 012 is green - see "After 012" below.
sbatch jobs/fine_tuning/013_train_prefix_rft.job
```

**Steps 3 and 4 answer "is the mechanism correct?". Step 5 answers "is the method
usable?", and they are different questions.** 011 pins the schedule over a single step,
so nothing before 012 has ever moved the cosine decay that is the method's central
dynamic; and 011/010 run `rollout.n` of 4 and 2, where production gives the hybrid
rollout seven on-policy peers. 012's pass says the production run is worth starting. It
says nothing about whether Prefix-RFT beats the GRPO baseline - 10 steps carries no
signal about final quality, and no check in 012 looks at reward level.

012's own checks are `E` (the curriculum moved: `low_t` fell, mean `k` fell with it,
`off_ratio` stayed under the paper's 0.5 concern threshold) and `F` (stability: KL,
grad norm, entropy, clip ratio - recorded as warnings rather than failures, because ten
steps cannot establish a trend, but this project's earlier GRPO-FT runs failed through
KL blow-up rather than through crashing, so a run that trains and checkpoints is not
automatically healthy).

Both passed on run 25762046, though not through the job's own check block: the 12 h wall
landed mid-step-10 and killed the script before it ran, so `E` and `F` were read off the
per-step metrics in `out/fine_tuning/prefix_rft/capped_25762046_verl.log` instead. `E`:
`low_t` 0.928 -> 0.072 tracking the closed form, mean `k` 2.42 -> 1.68 across the halves,
`off_ratio` never above 0.067. `F`: grad norm 0.078-0.154 and entropy 0.215-0.258 both
flat, `kl_loss` 0.007 -> 0.155 - a rise, but at `kl_coef` 0.01 that is 0.0016 of the loss,
and nine steps on 320 questions is early-training drift rather than the blow-up shape.
Watch it in `013`.

The run also settled a question none of the earlier checks could reach: the entropy clip
kept 20.13-20.29% of prefix tokens on every one of the nine steps, against the paper's
0.2, at the production batch shape. And `reward_with_prefix` beat `reward_without_prefix`
on all nine (0.74 against 0.41 on average), which is the mechanism doing the thing it
exists to do.

Step 2 is cheap and catches the pipeline's silent-failure modes: a demonstration attached
to the wrong question, a `prefix_mask` misaligned with the responses it marks, or a copied
verl method left stale by an upgrade. All of these produce a run that trains, checkpoints
and reports success while optimising the wrong thing - run the gate.

It also enforces three **verl runtime contracts**, via
`scripts/check_prefix_rft_runtime_contracts.py`. These are not theoretical: each was
discovered by a failed GPU run, because subclassing verl's classes and copying its methods
is not sufficient - verl's runtime imposes rules that no import error and no unit test
reveals.

| Contract | What breaks it | How it presents |
|---|---|---|
| Ray binds only `@register`-marked methods | Overriding a registered worker method without re-applying the decorator | `AttributeError: 'RayWorkerGroup' object has no attribute 'init_model'`. Note the override *removes* the method from the remote interface. A changed `dispatch_mode` is worse: it binds, runs on the wrong ranks, and never raises. |
| `init_model` converts config subtrees into typed dataclasses that reject undeclared keys | Adding a custom key under `actor_rollout_ref.actor` | `TypeError: FSDPActorConfig.__init__() got an unexpected keyword argument ...`, after Ray has started and the GPUs have loaded the model. Extra keys directly on `actor_rollout_ref` are fine - verl never converts that level, which is why `prefix_entropy_keep_ratio` lives there. |
| A copied body resolves its globals at call time | Copying a method without also copying its module's imports | `NameError` on the first training step, after everything else has succeeded. |

`launch_verl.py --dry-run` is the companion: it builds the exact command a real run would
issue and appends Hydra's `--cfg job`, proving the overrides *compose*. The contracts check
proves verl will *accept* the result. Both are pre-flight gates in 010 and 011, so a config
that cannot launch never consumes an allocation.

The store is keyed on **question text**, not a dataset index (indices collide across
data sources), so the same `prefix_demos.parquet` serves both the smoke split and the
production run - a question is covered if and only if its text appears in the store.

> **The production run and what follows it are covered in "After 012" below.** They are
> worth starting only once the staged path above is green; the spec's "Out of scope"
> section predates those stages.

## Configuration

`experiments/configs/fine_tuning/config_prefix_rft.yaml` starts from `config.yaml` (the
existing GRPO config) and adds one block; everything else - LoRA rank 64, `rollout.n: 8`,
`train_batch_size: 32`, `kl_loss_coef: 0.01`, the 4xH100 layout - is inherited unchanged.

| Key | Value | Source |
|---|---|---|
| `prefix_rft.enable` | `true` | ours |
| `prefix_rft.demos_path` | `data/training/prefix_rft/prefix_demos.parquet` | ours |
| `prefix_rft.n_prefixed_rollouts` | `1` | paper A.2 - one of eight rollouts |
| `prefix_rft.high` | `0.95` | paper A.2 |
| `prefix_rft.low_init` | `0.95` | paper A.2 |
| `prefix_rft.low_target` | `0.05` | paper A.2 |
| `prefix_rft.sampler_alpha` / `sampler_beta` | `1.0` / `1.0` | `BetaSampler` defaults - Beta(1,1) is A.2's uniform draw |
| `prefix_rft.entropy_keep_ratio` | `0.2` | paper A.2, §6 - top 20% by entropy |
| `prefix_rft.singleton_baseline` | `none` | ours |
| `prefix_rft.seed` | `42` | ours |
| `actor_rollout_ref.actor.calculate_entropy` | `true` | required by the clip; the entrypoint forces it on and fails loudly if it isn't set |

These keys live in `src/verl_ext/prefix_rft/config/prefix_rft_trainer.yaml`, a primary
Hydra config with the vendored AgentFlow config's keys inlined (it cannot be composed via
`hydra.searchpath`, which Hydra permits only in a primary config). `scripts/check_prefix_rft_trainer_sync.py`
diffs that inlined block against the vendored source, so a re-vendor cannot silently
change GRPO's setup while leaving Prefix-RFT on the old one.

`config_prefix_rft_smoke8b.yaml` mirrors `config_smoke8b.yaml`'s reductions: 2 GPUs, 8
samples, 2 rollouts, 1 epoch. With `rollout.n: 2` the hybrid rollout is 1 of 2 - it
exercises every code path while distorting the imitation/exploration balance, so the
smoke test asserts *machinery*, not quality.

`config_prefix_rft_tiny8b.yaml` goes further down: two chosen questions, `rollout.n: 4`,
one optimiser step. Both questions have exactly three teacher decisions, one routed
through `web_search` and one through `code_generator`, so the replay path is exercised
both ways. `scripts/build_tiny_prefix_split.py` regenerates the split and re-derives the
choice if the smoke split changes.

It carries **one deliberate deviation**, and you need to know why before copying it. The
cosine controller decays over `total_training_steps`, this run has exactly **one** step,
and **verl's `global_steps` starts at 1, not 0** - so on the only step it ever takes, the
schedule is already fully decayed to `low = 0.05`, making `k = floor(l * m)` a coin flip
that can be 0. A curriculum needs many steps to mean anything; a machinery test needs a
deterministic prefix. So the tiny config sets `low_target: 0.9`, giving `l` in
`[0.9, 0.95]` and `k = 2` for every three-decision demonstration on any step. **The
production and smoke configs keep the paper's 0.05.** The same off-by-one is why any
tooling that predicts `k` must ask the schedule about step 1.

`PREFIX_RFT=true` in a config's `env:` block is what switches `scripts/launch_verl.py`
and `scripts/train_orchestrator.py` from the plain GRPO module/rollout class to the
Prefix-RFT ones; the switch is additive, so an unset (or `false`) `PREFIX_RFT` reproduces
the existing GRPO launch byte-for-byte.

## Metrics to watch

Four print lines trace dispatch through to the loss (see "Debugging" below); these are
the aggregate metrics on top of them.

| Metric | Meaning |
|---|---|
| `actor/n_prefixed_rollouts` | How many rollouts in the step were seeded with a demonstration prefix. Zero here means the machinery never engaged - the run would be indistinguishable from plain GRPO in every other log line. |
| `actor/num_prefix_tokens` | Prefix tokens that entered the loss this step. |
| `actor/prefix_tokens_zeroed` / `actor/prefix_tokens_total` | The entropy clip's keep ratio. Expect close to the paper's 20% (`1 - zeroed/total`); far from it means the clip threshold or the entropy computation drifted. |
| `actor/off_ratio` | Share of the batch that is demonstration tokens. The paper's own operating point is **5-10%** (Table 4), and Table 4 warns of instability above roughly 0.5. Run 012 measured 0.067, i.e. inside the paper's range, so a number in this region is on target rather than suspiciously low. Read it that way: the paper reaches 51.8 against plain RFT's 45.5 at this share of the batch, so a single-digit percentage is not evidence the prefix is too small to matter. The smoke check treats >0.5 as a finding, not a hard fail. |
| `actor/reward_with_prefix` vs `actor/reward_without_prefix` | The paper's Figure 4 signature: prefixed rollouts should out-score unprefixed ones once training has run long enough to matter. On a two-step smoke run this is weak evidence and is recorded rather than gated on. |
| `actor/prefix_low` / `actor/prefix_high` | The schedule's current bounds on `l`, for confirming the cosine decay is moving. |
| `actor/prefix_steps` | Step mode only: mean `k`, the number of whole teacher decisions replayed. |
| `actor/prefix_l` | Token mode only: mean sampled `l`, the paper's own quantity. Deliberately a different series from `actor/prefix_steps`, because mean `k` and mean `l` are not comparable. |
| `actor/prefix_split_fraction` | Share of prefixed turns that were split mid-turn rather than replayed whole. Near 0 in step mode, near 1 in token mode; a low value in a token-mode run means budgets keep landing on decision boundaries and the two modes are converging. |

A misconfigured Prefix-RFT run looks exactly like a GRPO run in every metric *except*
these - checkpoints save, loss falls, reward moves. These are what distinguish the two,
which is why `010_smoke_prefix_rft.job` asserts on them directly rather than trusting
that training "worked."

## Debugging: following one prefix through the run

A Prefix-RFT run that silently behaves like GRPO is the failure this pipeline is most
exposed to, so the chain from dispatch to loss prints at every hop. All four lines use
`print`, deliberately: `logger.info` from these modules does **not** reach the SLURM log,
which is what made an early failure undiagnosable.

Read them in this order, in `*_verl.log` and `*_orchestrator.log`:

| Line | Where | Means |
|---|---|---|
| `Prefix dispatch: N of M rollouts prefixed (is_train=...); ks=[...]` | `*_verl.log` | The daemon computed a `k` per rollout. `ks` all zero means the schedule or the store lookup is the problem, not the replay. |
| `[PrefixOrchestratorRollout] replaying k of m teacher decisions ...` | `*_orchestrator.log` | A replay controller was **constructed**. This does *not* prove replay happened. |
| `[ReplayProvider] served replayed turn i/k (n response tokens)` | `*_orchestrator.log` | Replay actually reached the generation path. Absent while the line above is present means the orchestrator never called `generate` on the wrapped provider. |
| `Prefix mask: N of M rollouts marked, T prefix tokens` | `*_verl.log` | Replayed turns survived into `prefix_mask`. Zero here with a non-zero dispatch means the marking was lost between the triplet metadata and the batch. |

`_make_controller` also prints on each of its downgrade paths - missing `prefix_k` in the
payload (with the keys it did receive), missing store or tokenizer, no demonstration for
the question. Each of those was previously a silent `return None`, i.e. a silent
downgrade to plain GRPO.

## Regression checks for the inference path

Prefix-RFT touches `src/fine_tuning/` and adds two seams to `OrchestratorRollout`. Those
seams return their argument unchanged, so `agent_engine` inference should be unaffected -
but that is a claim, and `jobs/refactor_check/gaia_agentflow_smoke.job` is the evidence.
It runs vanilla AgentFlow on five GAIA questions in a few minutes:

```bash
sbatch jobs/refactor_check/gaia_agentflow_smoke.job                              # thinking off
sbatch --export=ALL,VARIANT=orchestrator jobs/refactor_check/gaia_agentflow_smoke.job
```

Both variants are worth running: `ORCHESTRATOR_ONLY` takes a different path through the
provider (Qwen3's `enable_thinking`) and through response parsing, where a `<think>` block
precedes the tool call, so a refactor can break one and not the other. The job fails if
every prediction is empty or no tool was called; accuracy is printed but not gated on,
because five GAIA questions carry no signal at an 8B model's single-digit accuracy.

## After 012: finishing the implementation

A green 012 is the last verification gate. What remains is running the method and
writing it up, in this order.

### 1. The production run

```bash
sbatch jobs/fine_tuning/013_train_prefix_rft.job
```

1800 questions x 1 epoch at batch 32, 56 steps, about 66 h of a 72 h wall on 4 H100s.

> **One epoch, not two.** The cosine schedule spans `total_training_steps`, so
> `total_epochs` sets how fast `low_t` decays and not merely how long the run lasts. At
> the 71 min/step that `012` measured, a 112-step span would be cut off by the wall near
> step 58 with `low_t` still around 0.47 - the prefix would still be half the
> demonstration and the run would never reach the near-on-policy phase A.2's schedule
> ends in. The GRPO reference run hit the same wall at 82 of 112 steps, which for plain
> GRPO costs only training time. At 56 steps the decay completes inside the wall.
`013` is `005_train.job` with three things changed and nothing else - the config, the job
name, and the log paths - so the GPU layout, crash monitoring, sub-agent placement and
checkpoint handling are the production-tested ones.

Checkpoints land on scratch:

```
/scratch-shared/$USER/fine_tuning/lora_adapters/qwen3-8b-prefix-rft-search-math/<run-tag>/
    global_step_<N>/actor/lora_adapter/adapter_model.safetensors
```

Note the run tag printed at the start; the evaluation configs need it.

**What to watch that a GRPO run would not have.** The four prefix metrics in "Metrics to
watch" plus the two the paper gives predictions for:

| Signal | What the paper says | What to do if it does not hold |
|---|---|---|
| `actor/prefix_low` falls 0.95 -> 0.05 | A.2's cosine decay | Stop the run. The curriculum is not moving and 012 should have caught it. |
| `actor/prefix_steps` (mean `k`) falls with it | follows from the above | As above. |
| `actor/off_ratio` | 5-10% is the paper's own range (Table 4); above ~0.5 it warns of instability | A single-digit percentage is the target, not a shortfall. Record anything above 0.5: the paper attributes instability to demonstration tokens dominating the batch. |
| `reward_with_prefix` above `reward_without_prefix` early, narrowing later | Figure 4 | Record it. Its absence is a real finding about the multi-turn adaptation, not a bug to chase - and it is the honest thing to report either way. |

### 2. Evaluation

Follow the pattern of `experiments/configs/qwen3/lora_inference/` (the GRPO adapter) or
`sft_inference/`: one config per benchmark, `lora_adapter_path` pointed at the chosen
`global_step_<N>/actor/lora_adapter`, five benchmarks (GAIA, MuSiQue, HLE, GPQA, AIME),
thinking on and off. These configs do not exist yet for Prefix-RFT and are the one piece
of scaffolding still to write.

Compare against the GRPO baseline (`qwen3-8b-grpo-search-math-v2`) rather than against
the paper's numbers: the paper's are single-turn mathematics with a verifier, and eight
documented divergences separate that setting from this one.

### 3. Reporting the results

Whatever the results are written up in, the eight divergences below need stating rather
than left for a reader to find. Two deserve prominence because they change what is being
measured:

- **The prefix measure** - in the default step mode `k` is an integer number of teacher decisions, not a
  token fraction, and with a mean of 2.98 decisions per demonstration the paper's
  continuous schedule is quantised into roughly three levels.
- **GRPO rather than Dr.GRPO**, chosen so Prefix-RFT differs from this project's RL
  baseline in exactly one respect.

If the production run does not beat the GRPO baseline, the earlier GRPO-FT experience is
the first place to look: those runs failed through KL blow-up rather than through
crashing, and `012`'s Check F exists to catch the same shape early.

## Alignment with the paper and the reference implementation

Checked against `papers/PrefixRFT_2507.01679v3.md` (A.2, line 345, is the hyperparameter
paragraph) and `repos/prefix_rft/recipe/prefix_rft/` (`core_algos.py`, `dp_actor.py`,
`rl_dataset.py`).

### What matches

| Element | Paper / repo | Ours |
|---|---|---|
| Rollouts per prompt, one prefixed | 8 rollouts, "one of them starts with the sampled prefix" (A.2) | `rollout.n: 8`, `n_prefixed_rollouts: 1` |
| Prefix length draw | `l ~ U[low_t, 0.95]`, prefix = `l x` demonstration length (A.2) | Same draw, via `Beta(1,1)` rescaled onto `[low_t, high]`, which is that uniform |
| Schedule shape | `low_t` cosine-decays 0.95 -> 0.05 (A.2) | `CosineDecayController`, ported from the repo's `global_step.py` |
| Entropy clip ratio | top 20% of prefix tokens by entropy (A.2, §6) | `entropy_keep_ratio: 0.2` |
| Clip scope | "for each mini-batch" (A.2); repo flattens `entropy[prefix_mask]` across the micro-batch before sorting | Global ranking across the micro-batch, same flattening |
| Clip mechanism | advantages of non-selected prefix tokens set to zero (§3) | Same: zero the advantage, do not mask the token |
| Prefix tokens are trained on | Table 8: freezing the prefix scores 45.4 against 45.5 for plain RFT, so this is load-bearing | Prefix tokens enter the loss with `prefix_mask = 1` |
| Prefix advantage | `p_score = score - mean(unprefixed group)`, then `/ num_rollouts_per_prefix`, applied only where `prefix_mask` (`core_algos.py:200-215`) | `apply_prefix_advantage` with `num_rollouts_per_prefix = 1`, reproducing the same expression |
| Singleton group handling | any group of one takes mean 0 and std 1 (`core_algos.py:188-191`) | Same, for the prefixed *and* the unprefixed group |
| Hybrid rollout excluded from the on-policy baseline | Grouped by `(question, prefix_index)`, unprefixed rollouts sharing a sentinel id | Same grouping by `(uid, is_prefix_rollout)` |

The table above is a claim about the reference, so it is checked against the reference
rather than asserted. `test_matches_the_reference_implementation` transcribes
`compute_grpo_prefix_outcome_advantage` into the test file and demands the same
advantages from ours across `rollout.n` in {2, 3, 4, 8} and five reward patterns. It is
worth having: it found that the singleton row above was only half true. We applied the
singleton rule to the prefixed group but not to the unprefixed one, so a question with
exactly one unprefixed rollout was centred on its own score where the reference centres
it on zero. Production `rollout.n` is 8, so this was unreachable there and only ever
affected the `rollout.n: 2` smoke config; it is fixed, and the test fails if it returns.

Two parameterisation differences that are **not** behavioural: the repo configures the
clip as a *mask* ratio (fraction zeroed, `ent_mask_ratio`) where we configure a *keep*
ratio, so our `0.2` is the repo's `0.8`; and the repo drives that ratio through a
controller so it can be scheduled, where the paper fixes it at 20% and so do we. Rounding
at the split point can differ by a single token on small batches.

### Where we diverge, and why

1. **Integer step prefix instead of a token fraction - in step mode only.** In the
   default `prefix_rft.mode: steps` we replay a whole number of teacher *decisions*
   while the paper cuts at an arbitrary token, which is the one divergence that changes
   what a prefix *is*. `prefix_rft.mode: tokens` removes it: the prefix becomes a token
   fraction and a decision can be split. Token mode's only remaining departure is that
   turn boundaries exist at all, since the fraction is taken over the concatenation of
   the decisions rather than over one response. Note that the two modes do not train on
   the same prefixed questions - see "Two ways to measure a prefix" above.
2. **GRPO with std normalisation instead of Dr.GRPO.** The paper uses Dr.GRPO (A.2), and
   the repo ships both (`compute_dr_grpo_outcome_advantage` alongside the GRPO variant).
   We pair with GRPO so Prefix-RFT differs from this project's existing RL baseline in
   exactly one respect, and a difference in results is attributable to the prefix rather
   than to a simultaneous algorithm change.
3. **PPO clip range and KL follow this project's GRPO baseline, not the reference's
   defaults.** We run `clip_ratio_low: 0.2` / `clip_ratio_high: 0.3` (clip-higher) with
   `use_kl_loss: true` and `kl_loss_coef: 0.01`, against the reference config's 0.2/0.2,
   `use_kl_loss: False` and `0.001`. The paper does not state these directly - A.2 says
   the shared hyperparameters follow Yan et al. (2025) - so the reference config is the
   only evidence, and it is a default rather than a published run setting. Same reasoning
   as the Dr.GRPO choice: matching our own RL baseline keeps the comparison controlled.
   Both differences are in the conservative direction for this project's known failure
   mode, since a larger KL coefficient penalises drift from the reference policy harder.
4. **LoRA rank 64 instead of a full fine-tune.** Inherited from the existing RL pipeline
   and its GPU budget.
5. **Multi-turn agentic trajectories with an outcome reward**, against the paper's
   single-turn mathematics with a verifier.
6. **Demonstration coverage.** 1358 of 1800 questions carry demonstrations, 1085 of them
   prefixable, against the 10% and 1% regimes the paper's Table 2 validates.
7. **The schedule spans the actual run length** rather than the paper's fixed 500 steps.
   A hardcoded 500 would truncate the curriculum at 22% of its decay on a ~112-step run.
8. **Four copies are taken from code this project does not own** - verl's `update_policy`,
   the vendored `_train_step` and `_async_set_up`, and the vendored config keys - each
   generated from its source plus marked edits and guarded by
   `check_prefix_rft_trainer_sync.py`, rather than hand-maintained with a comment.
9. **Group statistics are per rollout for prefixed questions, per row for the rest.**
   Flow GRPO emits one row per turn, all sharing a `uid` and a reward, so verl's GRPO
   averages over rows and thereby weights each rollout by its turn count.
   `apply_prefix_advantage` deduplicates by `rollout_id` first, matching the reference's
   one-row-per-rollout layout - it has to, or the hybrid rollout's turns would be centred
   against themselves and the prefix advantage would be exactly zero every step. It runs
   after `compute_advantage` and only touches questions that have a prefixed rollout, so
   within one batch those questions get unweighted group statistics and the rest keep
   verl's turn-count-weighted ones. Making both per rollout would have changed the GRPO
   baseline this method is being compared against, which is the larger distortion. The
   gap is bounded by how much turn counts vary within a question; run 25762046 averaged
   1.28 turns per rollout.
10. **`config_prefix_rft_tiny8b.yaml` sets `low_target: 0.9`**, not the paper's 0.05. Test
   configuration only, for the reason given under "Configuration"; the smoke and
   production configs use the paper's value.

## How it hooks into the framework

Prefix-RFT is a Level 3 adaptation method (see
[guides/add-an-adaptation-method.md](../guides/add-an-adaptation-method.md#level-3--online-adaptation-like-rl))
and reuses everything RL already built rather than duplicating it: the same
`OrchestratorRollout`-drives-the-real-orchestrator pattern, the same frozen sub-agent
server, the same reward function, and the same **Flow GRPO** credit assignment. Two
identity seams were added to `OrchestratorRollout` (`_wrap_provider`, `_wrap_tools`) so
the replay shims have somewhere to attach; both return their argument unchanged in the
base class, so the existing GRPO rollout suite is the regression test for "did this touch plain GRPO."

| Piece | Role |
|---|---|
| `scripts/build_prefix_demos.py` | Builds `prefix_demos.parquet` from a teacher trajectory collection |
| `scripts/check_prefix_demos.py` | Pre-flight gate over the store |
| `scripts/check_prefix_replay_tokenisation.py` | Verifies replayed turns tokenise exactly as the daemon's proxy would - the pipeline's highest-risk assumption, checked on CPU without a GPU run |
| `src/verl_ext/prefix_rft/` | `schedule.py`, `demos.py`, `advantage.py`, `entropy.py`, `masks.py`, `dispatch.py`, `actor.py`, `daemon.py`, `trainer.py`, `entrypoint.py` - the extension, entirely in modules we own |
| `src/fine_tuning/prefix_rollout.py`, `src/fine_tuning/prefix_replay.py` | `PrefixOrchestratorRollout` and the replay controller that stands in for the model on replayed decisions |

Everything under `src/fine_tuning/agentflow/` is untouched, and verl is neither forked
nor patched - see the spec's "Architecture" section for why each piece of logic lives in
a verl-free module rather than folded into the classes that use it.
