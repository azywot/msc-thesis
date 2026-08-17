# Prefix-RFT pipeline - blending demonstration and exploration into GRPO

Prefix-RFT is GRPO in which one rollout per prompt does not start from scratch: it is
seeded with a prefix of a Qwen3-32B teacher demonstration, and the policy writes the
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

> **Status (2026-08-18): the mechanism is verified on GPU.** `011_tiny_prefix_rft.job`
> passes all five checks (run 25755605): exactly one rollout per question replayed the
> teacher verbatim and then continued on-policy, validation stayed on-policy, 849 prefix
> tokens entered the loss, the entropy clip kept 20.3% of them against the paper's 20%,
> `off_ratio` was 0.121, and a LoRA adapter was written. Both prefixed rollouts scored
> reward 1.0.
>
> **This verifies the mechanism, not the method.** 011 pins the schedule over a single
> step and runs `rollout.n: 4`, so the cosine curriculum has never moved and the
> production batch shape is untested. Run `010` and then `012` before spending the full
> run; `012` exists precisely to answer whether the curriculum works.

---

## The step-prefix adaptation - the thing to know before reading a metric

The paper's test bed is single-turn math: one prompt, one response. CoSMAS trajectories
are multi-turn - a planning decision, one or more tool calls, a synthesis decision - so
"a prefix of the demonstration" cannot be a token fraction of one response. It is an
integer number of teacher **decisions**: for a demonstration with `m` decisions, the
first `k` are replayed verbatim and the model takes over from decision `k+1`.
`k = floor(l * m)`, clamped to `[0, m-1]` so there is always at least one on-policy
decision to score, with `l` drawn from the paper's schedule unchanged.

**Two consequences, accepted rather than fixed:**

1. **`k` is a staircase, not a continuous ratio.** Teacher trajectories average 2.98
   decisions, so the paper's continuous cosine decay is quantised into roughly three
   levels over training.
2. **Single-decision demonstrations never carry a prefix.** 273 of the 1358 demonstrated
   questions have exactly one decision, so `k <= m-1 = 0` always and they train as pure
   GRPO. 1085 of the 1800 training questions are prefixable.

Replayed decisions are **supervised, not teacher-forced context**: they enter the loss
as response tokens with `prefix_mask = 1`, carrying the trajectory advantage under the
entropy filter. This is load-bearing, not an implementation detail - the paper's Table 8
scores excluding the prefix from the loss at 33.8 against 51.8 for the version that
trains on it.

---

## Running it

```bash
# 1. Build the demonstration store from a teacher trajectory collection.
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
```

**Steps 3 and 4 answer "is the mechanism correct?". Step 5 answers "is the method
usable?", and they are different questions.** 011 pins the schedule over a single step,
so nothing before 012 has ever moved the cosine decay that is the method's central
dynamic; and 011/010 run `rollout.n` of 4 and 2, where production gives the hybrid
rollout seven on-policy peers. A pass on 012 says the 40-hour run is worth starting. It
says nothing about whether Prefix-RFT beats the GRPO baseline - 10 steps carries no
signal about final quality, and no check in 012 looks at reward level.

012's own checks are `E` (the curriculum moved: `low_t` fell, mean `k` fell with it,
`off_ratio` stayed under the paper's 0.5 concern threshold) and `F` (stability: KL,
grad norm, entropy, clip ratio - recorded as warnings rather than failures, because ten
steps cannot establish a trend, but this project's earlier GRPO-FT runs failed through
KL blow-up rather than through crashing, so a run that trains and checkpoints is not
automatically healthy).

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

> **The production run, the five-benchmark evaluation, and the Chapter 7 write-up are
> out of scope for this pipeline doc.** They follow only if the smoke path is clean and
> the compute is worth spending - see "Out of scope" in the spec.

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
| `actor/off_ratio` | Share of the batch that is demonstration tokens. The paper's Table 4 warns this destabilises training above roughly 0.5; the smoke check treats that as a finding, not a hard fail. |
| `actor/reward_with_prefix` vs `actor/reward_without_prefix` | The paper's Figure 4 signature: prefixed rollouts should out-score unprefixed ones once training has run long enough to matter. On a two-step smoke run this is weak evidence and is recorded rather than gated on. |
| `actor/prefix_low` / `actor/prefix_high` | The schedule's current bounds on `l`, for confirming the cosine decay is moving. |

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
| Singleton group handling | mean 0, std 1 (`core_algos.py:189-191`) | Same, including dividing by `std + epsilon` when std is 1 |
| Hybrid rollout excluded from the on-policy baseline | Grouped by `(question, prefix_index)`, unprefixed rollouts sharing a sentinel id | Same grouping by `(uid, is_prefix_rollout)` |

Two parameterisation differences that are **not** behavioural: the repo configures the
clip as a *mask* ratio (fraction zeroed, `ent_mask_ratio`) where we configure a *keep*
ratio, so our `0.2` is the repo's `0.8`; and the repo drives that ratio through a
controller so it can be scheduled, where the paper fixes it at 20% and so do we. Rounding
at the split point can differ by a single token on small batches.

### Where we diverge, and why

1. **Integer step prefix instead of a token fraction.** We replay a whole number of
   teacher *decisions*; the paper cuts at an arbitrary token. See "The step-prefix
   adaptation" above. This is the one divergence that changes what a prefix *is*.
2. **GRPO with std normalisation instead of Dr.GRPO.** The paper uses Dr.GRPO (A.2), and
   the repo ships both (`compute_dr_grpo_outcome_advantage` alongside the GRPO variant).
   We pair with GRPO so Prefix-RFT differs from this project's existing RL baseline in
   exactly one respect, and a difference in results is attributable to the prefix rather
   than to a simultaneous algorithm change.
3. **LoRA rank 64 instead of a full fine-tune.** Inherited from the existing RL pipeline
   and its GPU budget.
4. **Multi-turn agentic trajectories with an outcome reward**, against the paper's
   single-turn mathematics with a verifier.
5. **Demonstration coverage.** 1358 of 1800 questions carry demonstrations, 1085 of them
   prefixable, against the 10% and 1% regimes the paper's Table 2 validates.
6. **The schedule spans the actual run length** rather than the paper's fixed 500 steps.
   A hardcoded 500 would truncate the curriculum at 22% of its decay on a ~112-step run.
7. **Four copies are taken from code this project does not own** - verl's `update_policy`,
   the vendored `_train_step` and `_async_set_up`, and the vendored config keys - each
   generated from its source plus marked edits and guarded by
   `check_prefix_rft_trainer_sync.py`, rather than hand-maintained with a comment.
8. **`config_prefix_rft_tiny8b.yaml` sets `low_target: 0.9`**, not the paper's 0.05. Test
   configuration only, for the reason given under "Configuration"; the smoke and
   production configs use the paper's value.

## How it hooks into the framework

Prefix-RFT is a Level 3 adaptation method (see
[guides/add-an-adaptation-method.md](../guides/add-an-adaptation-method.md#level-3--online-adaptation-like-rl))
and reuses everything RL already built rather than duplicating it: the same
`OrchestratorRollout`-drives-the-real-orchestrator pattern, the same frozen sub-agent
server, the same reward function. Two identity seams were added to
`OrchestratorRollout` (`_wrap_provider`, `_wrap_tools`) so the replay shims have
somewhere to attach; both return their argument unchanged in the base class, so the
existing GRPO rollout suite is the regression test for "did this touch plain GRPO."

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
