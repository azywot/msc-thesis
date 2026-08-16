# Prefix-RFT: blending demonstration and exploration in the RL pipeline

**Date:** 2026-08-17
**Branch:** `feat/add-prefix-rft`
**Status:** design approved in chat; awaiting spec review
**Paper:** `papers/PrefixRFT_2507.01679v3.md` (Huang et al., ICML 2026, arXiv:2507.01679v3)
**Reference implementation:** `repos/prefix_rft`, recipe at `recipe/prefix_rft/`

## Goal

Add Prefix-RFT as a fourth adaptation method alongside GEPA, SFT and GRPO. It is a
Level 3 method in the taxonomy of `docs/guides/add-an-adaptation-method.md`: it trains
weights from the agent's own behaviour and therefore runs the real orchestrator inside
the training loop.

Prefix-RFT is GRPO in which one rollout per prompt does not start from scratch. It
starts from a prefix of a teacher demonstration and the model writes the continuation.
The composite trajectory is scored and trained on like any other rollout, so the
demonstration is reinforced in proportion to how much it helped.

Success means: the pipeline runs end to end on the 8B smoke path, the prefix machinery
is observably active in the metrics, and every hyperparameter traces to the paper or the
reference repository. Whether to spend the production run is a separate decision.

## What the method is

Three components, all from the paper.

**Prefix sampling** (§3). Of `N` rollouts for a prompt, `N-1` are ordinary on-policy
rollouts. The last one is seeded with `y*_<L`, a prefix of a demonstration, and the
policy generates the continuation. Rollout budget is unchanged: a prefixed rollout
replaces an on-policy one rather than being added to them.

**Advantage-driven weighting** (§3, §5.2). Prefix tokens carry the advantage estimated
from the whole hybrid trajectory, so a prefix that leads to a good outcome is
reinforced and one that does not is not. Table 8 measures the alternative: a static
0.001 weight in the UFT style scores 43.8 against 51.8 for dynamic weighting.

**Entropy-based clipping** (§3, §6, A.4). Only the top 20% highest-entropy prefix
tokens keep their advantage; the rest are zeroed. Without it, off-policy prefix
gradients dominate, since Table 4 shows prefix tokens roughly double the gradient norm
while making up 5-10% of the batch. The two failure modes Table 8 records are freezing
the prefix entirely, which reverts to plain RFT (45.4 against RFT's 45.5), and updating
all prefix tokens, which yields 45.7 with response-length explosion.

**Prefix length schedule** (§3, A.2). `l ~ U(low_t, 0.95)` with `low_t` cosine-decaying
from 0.95 to 0.05 across training, giving a curriculum from demonstration-heavy to
exploration-heavy. The uniform-schedule ablation (§6) is measurably worse.

## Adaptation to a multi-turn agentic setting

The paper's test bed is single-turn math: one prompt, one response, one verifier.
CoSMAS trajectories are multi-turn: a planning decision, one or more tool-call
decisions, and a synthesis decision, with the orchestrator's prompt rebuilt from folded
memory at every step.

**Decision.** The prefix is an integer number of teacher *decisions*, not a token
fraction. For a demonstration with `m` decisions, the first `k` are replayed verbatim
and the model takes over from decision `k+1`.

```
y* = [resp_1 | resp_2 | resp_3 | resp_4]        teacher, m = 4

step 20    k = 3     ####|####|####|~~~~~~~~~
step 250   k = 2     ####|####|~~~~~|~~~~~~~
step 480   k = 0     ~~~~~|~~~~~|~~~~~|~~~~~

####  teacher tokens, prefix_mask = 1, in the loss
~~~~  on-policy continuation, prefix_mask = 0
```

`k = floor(l * m)` clamped to `[0, m-1]`, where `l` comes from the paper's schedule
unchanged. The clamp is the step-level analogue of the reference implementation's
`prefix_len >= demo_len -> prefix_len = demo_len - 1` guard
(`recipe/prefix_rft/rl_dataset.py:300-301`) and guarantees at least one on-policy
decision, so there is always a continuation to score.

**Consequences, accepted.** Two follow from choosing steps over tokens.

1. `k` is a staircase, not a continuous ratio. Teacher trajectories have a median of 4
   decisions, so a 500-step cosine decay is quantised into roughly 4 levels.
2. 71 of the 700 demonstrated questions have a single decision. For them `k <= m-1 = 0`
   always, so they never carry a prefix and behave as pure GRPO.

Both are recorded rather than fixed. The mitigation, if it later matters, is a
token-level prefix on the boundary decision, which restores continuous control.

**Replayed decisions are supervised.** They enter the loss as response tokens with
`prefix_mask = 1`, carrying the trajectory advantage under the entropy filter. They are
not teacher-forced context. This is load-bearing: excluding the prefix from the loss is
the DR-PO variant that Table 8 scores at 33.8 against 51.8, and freezing it scores 45.4,
which is plain RFT.

## Architecture

Nothing under `src/fine_tuning/agentflow/` is modified. That directory is vendored and
`VENDORED.md` forbids patching it; all extensions are subclasses living in our own
trees.

```
src/verl_ext/prefix_rft/
  schedule.py      CosineDecayController, BetaSampler, step discretisation
  demos.py         demonstration store keyed by question idx
  advantage.py     prefix-aware GRPO advantage correction
  actor.py         PrefixRFTActor: entropy clipping on prefix tokens
  worker.py        PrefixRFTWorker: installs the actor
  daemon.py        PrefixRFTDaemon: per-rollout k dispatch, prefix_mask tensor
  trainer.py       PrefixRFTTrainer: _train_step with the prefix hooks
  entrypoint.py    mirrors the agentflow entrypoint with our classes substituted

src/fine_tuning/
  prefix_rollout.py   PrefixOrchestratorRollout(OrchestratorRollout)

scripts/
  build_prefix_demos.py    demonstration store builder
  check_prefix_demos.py    preflight gate

experiments/configs/fine_tuning/
  config_prefix_rft.yaml, config_prefix_rft_smoke8b.yaml

jobs/fine_tuning/
  008_build_prefix_demos.job, 009_run_tests_for_prefix_rft.job, 010_smoke_prefix_rft.job

docs/pipelines/prefix-rft.md
```

### Data flow

```
prefix_demos.parquet
        |
        v
PrefixRFTTrainer  ---- computes l, k per rollout at dispatch (has global_step)
        |
        v
PrefixRFTDaemon._async_set_up  ---- puts prefix_k in each task payload
        |
        v  HTTP
PrefixOrchestratorRollout
   |-- _ReplayProvider        first k generate() calls return teacher responses
   |-- _ReplayToolRegistry    those steps return stored teacher tool results
   |-- AgenticOrchestrator    unmodified; builds every prompt itself
   |-- Triplet.metadata["prefix"] = True on replayed turns
        |
        v  HTTP
PrefixRFTDaemon.get_train_data_batch  ---- emits prefix_mask alongside responses
        |
        v
PrefixRFTTrainer._train_step
   |-- verl compute_advantage (grpo)
   |-- prefix advantage correction on prefix_mask tokens
        |
        v
PrefixRFTActor.update_policy  ---- entropy clip, then verl policy loss
```

## Component specifications

### 1. Demonstration store

Source is `data/training/sft/collected_20260605_214650.jsonl`, the Qwen3-32B teacher
collection already used for SFT. The builder reuses `scripts/build_sft_parquet.py`'s
`_strip_thinking` (line 106) and its trajectory splitter (line 189), which already
returns `(plan, [(action_content, tool_name, tool_result)], answer)`. That is exactly
the decomposition a step-level prefix needs, including the stored tool results that make
replay possible.

Output `data/training/prefix_rft/prefix_demos.parquet`, one row per question:

| Column | Content |
|---|---|
| `idx` | question index, joins to `combined_train.parquet` |
| `data_source` | `hotpotqa`, `nq` or `deepmath` |
| `n_steps` | number of teacher decisions, `m` |
| `steps` | ordered list of `{response, tool_name, tool_result}` |

Only correct teacher trajectories are kept, matching the SFT builder's filter, so the
reference repository's `demos_corr` flag is uniformly true and needs no representation.
Thinking is stripped, matching `THINKING_MODE: NO` at RL time.

Coverage is 700 of the 1800 RL training questions. In the paper's terms that is a
demonstration ratio of 0.39, well inside the range Table 2 validates, where 10% and even
1% coverage still beat both SFT and RFT. The remaining 1100 questions train as ordinary
GRPO, which is what the reference implementation's `demo_ratio` mechanism does
(`rl_dataset.py:194-203`).

`scripts/check_prefix_demos.py` is a CPU preflight gate, run the way
`007_run_tests_for_sft_folded.job` runs the SFT gate. It asserts, on every row:

- no empty step responses,
- no surviving `<think>` blocks,
- every non-final step has a parseable tool call and a stored tool result,
- replaying the first `k` steps reproduces the prompt `AgenticOrchestrator._build_memory_prompt`
  builds, for every `k` in `[0, m-1]`.

The last check is the same class of defect that made an early SFT run score below its own
base model, as recorded in `docs/pipelines/sft.md`. The training job refuses to start if
the gate fails.

### 2. Schedule

`src/verl_ext/prefix_rft/schedule.py` ports `CosineDecayController` and `BetaSampler`
from `recipe/prefix_rft/scheduler/global_step.py:148-186` and `256-275`.

| Parameter | Value | Source |
|---|---|---|
| `prefix_high` | 0.95, constant | paper A.2 |
| `prefix_low` init | 0.95 | paper A.2 |
| `prefix_low` target | 0.05 | paper A.2 |
| decay | cosine over `total_training_steps` | paper A.2, §3 |
| sampler | Beta(1, 1), which is uniform on `[low_t, high]` | `BetaSampler` defaults, matching `l ~ U(low, high)` |
| warmup_ratio | 0.0 | `CosineDecayController` default |

The paper decays over 500 steps. Our run is `total_epochs: 2` over 1800 questions at
`train_batch_size: 32`, so roughly 112 steps. The schedule is expressed against
`trainer.total_training_steps` rather than a hardcoded 500, so the curriculum spans the
run as the paper intends rather than truncating at 22% of the decay.

Discretisation, ours: `k = clamp(floor(l * m), 0, m - 1)`.

### 3. Rollout

`PrefixOrchestratorRollout` subclasses `OrchestratorRollout` and changes only how the
first `k` decisions are produced.

- `_ReplayProvider` wraps `_CapturingProvider`. Its first `k` `generate()` calls return
  the teacher's stored response for that decision; subsequent calls delegate to the real
  provider. Both kinds of turn are captured, so both become triplets.
- A replaying tool registry returns the teacher's stored tool result for the replayed
  decisions, then hands over to the real registry. This is where the efficiency gain
  comes from: no generation, no Serper call and no sub-agent call for replayed steps,
  against a baseline where generation was 1091s of a 1216s step
  (`src/fine_tuning/README.md:222`).
- `AgenticOrchestrator` is untouched, so replayed prompts are built by the same
  `_build_memory_prompt` used at inference. This follows the guide's rule: drive the real
  orchestrator, never a simplified loop.
- Replayed turns carry `Triplet.metadata = {"prefix": True}`. That field already exists
  in the vendored `types.py:30`, so no vendored file changes.

Rollout 0 of the 8 is the hybrid one; rollouts 1 to 7 are unchanged. Validation rollouts
always get `k = 0`, so checkpoint selection measures unaided policy quality.

`k` is computed at the driver, where `global_step` lives, and shipped per rollout in the
task payload. `PrefixRFTDaemon` overrides `_async_set_up` (vendored `daemon.py:295-315`)
to write a per-rollout copy of the sample carrying `prefix_k`, rather than the shared
dict the vendored loop reuses.

### 4. prefix_mask

`PrefixRFTDaemon.get_train_data_batch` extends the vendored method to emit a
`prefix_mask` tensor shaped like `responses`, set to 1 across every token of a triplet
whose metadata marks it as prefix. It follows the same right-padding, truncation and
drop-mask paths as the existing response tensors, so a prefix token that is truncated
away is also dropped from the mask.

### 5. Advantage

`algorithm.adv_estimator` stays `grpo`, matching the existing RL baseline, so
Prefix-RFT differs from it in exactly one respect. After verl's `compute_advantage`
returns, `src/verl_ext/prefix_rft/advantage.py` overwrites the advantage on prefix
tokens, porting `compute_grpo_prefix_outcome_advantage`
(`recipe/prefix_rft/core_algos.py:162-217`):

- groups are `(question uid, prefix uid)`; the 7 unprefixed rollouts share one group and
  the hybrid rollout is alone in its own,
- a singleton group takes mean 0 and std 1, so the hybrid rollout's score passes through
  uncentred,
- prefix tokens then get `score_hybrid - mean(scores of the 7 unprefixed rollouts)`,
  divided by the rollouts-per-prefix count, which is 1 here,
- non-prefix tokens keep the advantage verl computed.

This is the quantity Figure 4 plots as the gap between reward-with-prefix and overall
training reward, and it is what makes the prefix's influence fade as the policy improves.

**Why not the `_v2` variant.** `recipe/prefix_rft/core_algos.py:310-353` computes the
group mean even for singleton groups. With one prefixed rollout that yields a centred
score of exactly 0 for the hybrid, and a prefix advantage of `-mean(unprefixed)`, which
is non-positive regardless of whether the prefix helped. `_v2` therefore presupposes
several rollouts sharing one prefix. The paper states one of eight (A.2), and the
reference repository ships no configuration file fixing these knobs, since they arrived
through an unversioned `$TRAIN_CONFIG`. The paper is the authority and the non-`_v2`
function is the one consistent with it.

**Known risk, carried from the reference.** With a singleton prefix group the hybrid
rollout's *continuation* tokens get an uncentred advantage equal to the raw reward, so
they are pushed non-negatively regardless of how the unprefixed rollouts did. The
reference authors flag exactly this in a comment at `core_algos.py:294-296`. Mitigation
is monitoring, not prevention: log the hybrid rollout's mean advantage separately, and
expose `prefix_rft.singleton_baseline: none | group`, defaulting to `none` for fidelity,
where `group` centres the hybrid continuation against the unprefixed rollouts. Flip it
only if training destabilises, and record the flip.

### 6. Entropy clipping

`PrefixRFTActor` subclasses `verl.workers.actor.DataParallelPPOActor` and overrides
`update_policy` to:

- add `prefix_mask` to verl's fixed `select_keys` list (`dp_actor.py:516`), which
  otherwise drops it,
- per micro-batch, sort prefix tokens by current-policy entropy and zero the advantage
  of the bottom 80%, retaining the top 20%.

This reproduces `dp_actor.reshape_func`'s `entropy` branch
(`recipe/prefix_rft/dp_actor.py:132-158`), including that the ratio comes from a
controller so it can later be scheduled; the default is `ConstController(0.8)`, which
masks the bottom 80% and so keeps the top 20% the paper specifies.

Entropy is the current policy's, recomputed per micro-batch, as in the reference. Doing
the clip at the driver on old-policy entropy would be simpler but is not equivalent
here: `ppo_mini_batch_size` is 8 against a `train_batch_size` of 32, so the policy moves
between mini-batches. Requires `actor.calculate_entropy: true`.

The actor is installed by `PrefixRFTWorker`, which subclasses verl's
`AsyncActorRolloutRefWorker` and reassigns `self.actor.__class__` after
`super().init_model()`. verl is neither forked nor patched, and the worker stays a
handful of lines.

### 7. Trainer

`PrefixRFTTrainer` subclasses the vendored `AgentFlowTrainer` and overrides `_train_step`
to insert the advantage correction between `compute_advantage` and `update_actor`, and
to instantiate our daemon.

`_train_step` is a single ~200-line method in vendored code with no smaller seam, so the
override is a copy with the hooks added. Its docstring records the vendored revision it
was copied from, so a future re-vendor knows to re-sync it. The alternative, rebinding
the vendored module's `compute_advantage` attribute, was rejected: it is invisible at the
call site and `VENDORED.md` exists to prevent exactly that kind of hidden patch.

### 8. Configuration

`experiments/configs/fine_tuning/config_prefix_rft.yaml` starts from `config.yaml` and
adds one block. Everything not listed is inherited unchanged, including LoRA rank 64,
`rollout.n: 8`, `train_batch_size: 32`, `kl_loss_coef: 0.01` and the 4xH100 layout.

| Key | Value | Source |
|---|---|---|
| `prefix_rft.enable` | `true` | ours |
| `prefix_rft.demos_path` | `data/training/prefix_rft/prefix_demos.parquet` | ours |
| `prefix_rft.n_prefixed_rollouts` | `1` | paper A.2, one of eight |
| `prefix_rft.high` | `0.95` | paper A.2 |
| `prefix_rft.low_init` | `0.95` | paper A.2 |
| `prefix_rft.low_target` | `0.05` | paper A.2 |
| `prefix_rft.low_ctrl_type` | `cosine_decay` | paper A.2 |
| `prefix_rft.sampler` | `beta`, alpha 1, beta 1 | `BetaSampler` defaults |
| `prefix_rft.entropy_keep_ratio` | `0.2` | paper A.2, §6 |
| `prefix_rft.entropy_ctrl_type` | `const` | reference default |
| `prefix_rft.singleton_baseline` | `none` | ours, see risk above |
| `actor.calculate_entropy` | `true` | required by the clip |

`config_prefix_rft_smoke8b.yaml` mirrors `config_smoke8b.yaml`'s reductions: 2 GPUs, 8
samples, 2 rollouts, 1 epoch, `save_freq` and `test_freq` 1. With `rollout.n: 2` the
hybrid rollout is 1 of 2, which exercises every code path while distorting the
imitation/exploration balance; the smoke test asserts machinery, not quality.

## Divergences from the paper

Recorded here so the write-up can state them rather than discover them.

1. **Integer step prefix instead of token fraction.** Deliberate. Consequences in
   "Adaptation" above.
2. **GRPO with std normalisation instead of Dr.GRPO.** The paper uses Dr.GRPO (A.2).
   Keeping GRPO means Prefix-RFT differs from the existing RL baseline in one respect
   only, so a difference in results is attributable to the prefix rather than to a
   simultaneous algorithm change.
3. **LoRA rank 64 instead of full fine-tune.** Inherited from the existing RL pipeline
   and its GPU budget.
4. **Multi-turn agentic trajectories with an outcome reward** against the paper's
   single-turn math with a verifier.
5. **700 of 1800 questions carry demonstrations**, inside the range Table 2 validates.
6. **Schedule spans the actual run length** rather than the paper's fixed 500 steps.
7. **One vendored method is copied** into a subclass, documented at the copy site.

## Testing

**Unit** (`tests/unit/test_prefix_rft_*.py`, CPU, no GPU):

| Target | Assertion |
|---|---|
| `schedule.py` | `low_t` is 0.95 at step 0, decays monotonically, reaches 0.05 at the final step; `l` always lies in `[low_t, 0.95]` |
| step discretisation | `k` in `[0, m-1]` for all `m >= 1`; `k = 0` whenever `m = 1`; `k` decreases in expectation as training advances |
| `demos.py` | store loads, indexes by `idx`, and misses return `k = 0` rather than raising |
| daemon `prefix_mask` | mask is 1 exactly on replayed-triplet response tokens, 0 on padding, and survives truncation consistently with `responses` |
| `advantage.py` | on a hand-built group of 8 with known scores, prefix tokens equal `score_hybrid - mean(unprefixed)` and non-prefix tokens are untouched |
| entropy clip | on a synthetic entropy tensor, exactly the top 20% of prefix tokens retain non-zero advantage and no non-prefix token is touched |
| replay provider | first `k` calls return teacher text, call `k+1` delegates, and triplet metadata is marked correctly |

**Gate.** `check_prefix_demos.py` over the full store, wired into
`009_run_tests_for_prefix_rft.job` alongside the unit tests, on the CPU partition.

**Smoke.** `010_smoke_prefix_rft.job`, 8B on 2 GPUs, asserting that
`actor/num_prefix_tokens` and `actor/off_ratio` are non-zero, that the run reaches a
checkpoint, and that reward-with-prefix exceeds overall training reward in the early
steps, which is the Figure 4 signature and the cheapest evidence the prefix is doing
anything.

**Metrics**, ported from `recipe/prefix_rft/ray_trainer.py:1139-1145` and `1183-1184`:
`actor/prefix_steps` (our step-level analogue of `prefix_ratio`),
`actor/num_prefix_tokens`, `actor/off_ratio`, `actor/prefix_low`, `actor/prefix_high`,
and reward split by prefixed against unprefixed rollouts.

## Out of scope

The production 4xH100 run, the five-benchmark evaluation suite, and the Chapter 7
write-up. Those follow only if the smoke path is clean and the compute is worth
spending.
