# Prefix-RFT: blending demonstration and exploration in the RL pipeline

**Date:** 2026-08-17
**Branch:** `feat/add-prefix-rft`
**Status:** approved; implemented through Task 8 of the plan. Revised 2026-08-17 to match
what was built. Every change made during implementation is recorded under
"Implementation record" at the end.
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

1. `k` is a staircase, not a continuous ratio. Teacher trajectories average 2.98
   decisions, so the cosine decay is quantised into roughly three levels.
2. 273 of the 1358 demonstrated questions have a single decision. For them `k <= m-1 = 0`
   always, so they never carry a prefix and behave as pure GRPO. 1085 questions are
   prefixable.

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

A split runs through the whole tree: **every piece of real logic lives in a module
that imports no verl**, and the verl-touching classes are thin wrappers over it. This
is not stylistic. verl is absent from the `agent_engine` env and pytest is absent from
`cosmas-train`, so a module importing verl cannot be exercised by a test in either
environment. The split is what keeps the logic under test.

```
src/verl_ext/prefix_rft/            verl-free logic          verl-touching wrapper
  schedule.py      CosineDecayController, BetaSampler, step discretisation
  demos.py         DemoStore, keyed on the question text
  dispatch.py      prefix_k_for(): how many decisions this rollout replays
  masks.py         build_prefix_mask(): marks replayed tokens in the batch
  advantage.py     apply_prefix_advantage(): the prefix-aware GRPO advantage
  entropy.py       clip_prefix_advantage_by_entropy(): the top-20% filter
  actor_edits.py   the three edits to verl's update_policy, as data
  trainer_edits.py the two edits to the vendored _train_step, as data
                                            actor.py     PrefixRFTActor
                                            worker.py    PrefixRFTWorker
                                            daemon.py    PrefixRFTDaemon
                                            trainer.py   PrefixRFTTrainer
                                            entrypoint.py, __main__.py
  config/prefix_rft_trainer.yaml   primary Hydra config

src/fine_tuning/
  prefix_replay.py    ReplayController, ReplayToolRegistry, ReplayProvider
                      (imports only agent_engine, so it is CPU-testable)
  prefix_rollout.py   PrefixOrchestratorRollout(OrchestratorRollout)

scripts/
  build_prefix_demos.py             demonstration store builder
  check_prefix_demos.py             preflight gate on the store
  check_prefix_rft_trainer_sync.py  guards all three copied methods
  check_prefix_rft_trainer_sync.py  guards all three copies

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
| `question_key` | SHA-1 of the stripped question text. **The lookup key.** |
| `question_id` | row position in the shuffled parquet, diagnostics only |
| `data_source` | `hotpotqa`, `nq` or `deepmath` |
| `question` | the question text, diagnostics only |
| `n_steps` | number of teacher decisions, `m` |
| `steps` | ordered list of `{response, tool_name, tool_result}` |

**The key is the question text, not `extra_info.idx`.** `prepare.py` assigns `idx` per
data source, so it collides across them: in `sft_train.parquet`, `idx` takes 700 distinct
values across 968 rows, and `idx = 669` is both a deepmath question and a hotpotqa one.
Keying on it would replay a maths demonstration into a search question. Training would
run, rewards would be computed, and nothing downstream would flag it. `question_id` is
also unusable: it is the row position in the shuffled parquet, which the rollout worker
never sees. The question text is unique across all 968 rows and both sides hold it
verbatim. `question_key` is defined once, in `demos.py`, and imported by the builder so
the build-time and lookup-time hashes cannot drift.

Only correct teacher trajectories are kept, matching the SFT builder's filter, so the
reference repository's `demos_corr` flag is uniformly true and needs no representation.
Thinking is stripped, matching `THINKING_MODE: NO` at RL time. Trajectories with a
surviving `<think>` opening tag are dropped: the strip matches `<think>...</think>`, so an
unclosed tag means the teacher hit its token limit mid-thought. Both such trajectories in
the 2026-06-05 collection are 26k-plus character repetition loops, and replaying one would
teach the policy to loop.

**Coverage, as built: 1358 of the 1800 RL training questions, 4047 decisions, mean 2.98
per question, of which 1085 are prefixable** (the other 273 have a single decision). In
the paper's terms that is a demonstration ratio of 0.75, comfortably above the 10% and 1%
regimes Table 2 validates. The remaining ~440 questions train as ordinary GRPO, which is
what the reference implementation's `demo_ratio` mechanism does
(`rl_dataset.py:194-203`).

An earlier estimate of 700 questions came from counting `sft_train.parquet`, which had
been thinned by a train/val split and a math:search rebalance that Prefix-RFT does not
need. Building directly from the collection recovers the rest.

`scripts/check_prefix_demos.py` is a CPU preflight gate, run the way
`007_run_tests_for_sft_folded.job` runs the SFT gate. It asserts, on every row:

- no empty step responses,
- no surviving `<think>` blocks,
- every step that *is* a tool call has a parseable call and a stored tool result,
- no step in the middle is anything other than a tool call,
- no trajectory ends on a tool call, which would mean it has no answer,
- no duplicate `question_key`, which would shadow a demonstration.

The middle-step rule is what the data actually looks like, and it took a correction to
get right: the first decision is the *planning turn*, which is legitimately not a tool
call, so an initial "every non-final step is a tool call" rule rejected 1087 of 1360 rows.
Measured across the store, the shape is uniform: step 0 is the plan, the middle steps are
all tool calls with stored results, the last is the answer, and no tool step is missing a
result.

The gate is the same class of guard that would have caught the defect that made an early
SFT run score below its own base model, as recorded in `docs/pipelines/sft.md`. The
training job refuses to start if it fails.

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

- `ReplayProvider` wraps `_CapturingProvider`. Its first `k` `generate()` calls return
  the teacher's stored response for that decision; subsequent calls delegate to the real
  provider. Both kinds of turn are captured, so both become triplets.
- `ReplayToolRegistry` returns the teacher's stored tool result for the replayed
  decisions, then hands over to the real registry. This is where the efficiency gain
  comes from: no generation, no Serper call and no sub-agent call for replayed steps,
  against a baseline where generation was 1091s of a 1216s step
  (`src/fine_tuning/README.md:222`). **The stored result is single-use.** Serving it on
  every lookup means the policy's first genuine tool call after replay ends is handed the
  teacher's stale result instead of executing, and the trajectory silently stops
  corresponding to anything the policy did.
- `AgenticOrchestrator` is untouched, so replayed prompts are built by the same
  `_build_memory_prompt` used at inference. This follows the guide's rule: drive the real
  orchestrator, never a simplified loop.
- Replayed turns carry `Triplet.metadata = {"prefix": True}`. That field already exists
  in the vendored `types.py:30`, so no vendored file changes.

Two identity seams, `_wrap_provider` and `_wrap_tools`, are added to `OrchestratorRollout`
so the shims have somewhere to attach. The base implementations return their argument
unchanged, so ordinary GRPO is unaffected, and the existing rollout suite gates that.

The shims live in `src/fine_tuning/prefix_replay.py`, importing only `agent_engine`.
`fine_tuning.rollout` pulls in agentflow, which needs `agentops`, absent from the CPU test
env; keeping the shims separate is what makes them testable.

Rollout 0 of the 8 is the hybrid one; rollouts 1 to 7 are unchanged. Validation rollouts
always get `k = 0`, so checkpoint selection measures unaided policy quality.

`k` is computed at the driver, where `global_step` lives, and shipped per rollout in the
task payload. `PrefixRFTDaemon` overrides `_async_set_up` (vendored `daemon.py:295-315`)
to write a per-rollout copy of the sample carrying `prefix_k`, rather than the shared
dict the vendored loop reuses.

### 4. prefix_mask

`PrefixRFTDaemon.get_train_data_batch` calls the vendored method, then attaches a
`prefix_mask` tensor shaped like `responses`, set to 1 across every token of a triplet
whose metadata marks it as prefix, plus an `is_prefix_rollout_list` the advantage needs.
`build_prefix_mask` applies the same truncation and skip rules as the base method (a turn
is dropped only when prompt *and* response are empty), and a hard row-count assertion
fails the step if the two ever disagree. Silent misalignment here would mark the wrong
tokens as demonstrations.

### 5. Advantage

`algorithm.adv_estimator` stays `grpo`, matching the existing RL baseline, so
Prefix-RFT differs from it in exactly one respect. After verl's `compute_advantage`
returns, `src/verl_ext/prefix_rft/advantage.py` overwrites the advantage on prefix
tokens, porting `compute_grpo_prefix_outcome_advantage`
(`recipe/prefix_rft/core_algos.py:162-217`):

- groups are `(question uid, prefix uid)`; the 7 unprefixed rollouts share one group and
  the hybrid rollout is alone in its own,
- **grouping is per rollout, not per row.** Flow GRPO emits one row per turn, all
  carrying the same `uid` and the same reward, so scores are deduplicated by
  `rollout_id` before any group statistic is taken. A row-level port would place the
  hybrid rollout's several turns in a group of their own, centre them against
  themselves, and yield a prefix advantage of exactly zero on every step,
- a singleton group takes mean 0 and std 1, so the hybrid rollout's score passes through
  uncentred,
- prefix tokens then get `score_hybrid - mean(scores of the 7 unprefixed rollouts)`,
  divided by the rollouts-per-prefix count, which is 1 here.

For a question with no prefixed rollout, which covers the ~440 undemonstrated questions,
the 273 single-decision ones and every `k = 0` draw, the port reduces to plain GRPO and verl's output stands
unchanged. For a question that does have one, the port replaces verl's advantage on all
of that question's rows, because the reference excludes the hybrid rollout from the
on-policy baseline and verl cannot. That exclusion matters: at 1 of 8 with a hybrid
reward near 1.0, including it would lift the group mean and put a systematic negative
bias on every on-policy rollout.

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

The filter itself is `entropy.py`'s `clip_prefix_advantage_by_entropy`, which sorts
prefix tokens by entropy and zeroes the advantage of all but the top `keep_ratio`. Ranking
is global across the micro-batch, not per row, as the reference does
(`recipe/prefix_rft/dp_actor.py:138-139`): a row of uniformly low-entropy prefix tokens
can be dropped entirely while another row keeps several.

`PrefixRFTActor` subclasses `verl.workers.actor.DataParallelPPOActor` and carries three
marked edits to `update_policy`: add `prefix_mask` to verl's fixed `select_keys` list
(`dp_actor.py:516`) which otherwise drops it, force `calculate_entropy`, and apply the
filter immediately before the policy loss.

Entropy is the current policy's, recomputed per micro-batch, as in the reference. Doing
the clip at the driver on old-policy entropy would need no copied code but is not
equivalent here: `ppo_mini_batch_size` is 8 against a batch of several hundred rows, so
the policy takes many optimizer steps per training step and old-policy entropy is stale
for most mini-batches.

The actor is installed by `PrefixRFTWorker`, which subclasses verl's
`AsyncActorRolloutRefWorker` and reassigns `self.actor.__class__` after
`super().init_model()`. verl is neither forked nor patched, and the worker stays a
handful of lines.

### 7. Trainer

`PrefixRFTTrainer` subclasses the vendored `AgentFlowTrainer` and overrides `_train_step`
to insert the advantage correction between `compute_advantage` and `update_actor`, and
to instantiate our daemon.

`_train_step` is a single ~200-line method in vendored code with no smaller seam, so the
override is a copy with two marked edits. The alternative, rebinding the vendored module's
`compute_advantage` attribute, was rejected: it is invisible at the call site and
`VENDORED.md` exists to prevent exactly that kind of hidden patch.

The daemon is *not* constructed here. The vendored `fit()` builds a plain
`AgentModeDaemon` at `trainer.py:427`, so `PrefixRFTTrainer._ensure_prefix_daemon`
promotes that instance in place by reassigning `__class__` and setting the added
attributes explicitly, the same trick the worker uses for the actor. Copying `fit()` as
well would have doubled the copied surface for no gain.

### 7a. Guarding the copies

Three things are copied from sources this project does not own: verl's `update_policy`,
the vendored `_train_step`, and the vendored Hydra config's keys. A docstring is not
enough, because a stale copy keeps running: training would proceed on the old loss body
and report success.

The edits are therefore stored as *data*, in `actor_edits.py` and `trainer_edits.py`, and
`scripts/check_prefix_rft_trainer_sync.py` re-derives each copy from the current source
and diffs it against the file. A verl upgrade or a re-vendor surfaces as a failed check.
The copies themselves were generated by applying those edits at asserted-unique anchors,
not hand-transcribed.

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
| `prefix_rft.sampler_alpha` / `sampler_beta` | `1.0` / `1.0` | `BetaSampler` defaults; Beta(1,1) is A.2's uniform draw |
| `prefix_rft.entropy_keep_ratio` | `0.2` | paper A.2, §6 |
| `prefix_rft.singleton_baseline` | `none` | ours, see risk above |
| `prefix_rft.seed` | `42` | ours |
| `actor.calculate_entropy` | `true` | required by the clip; the entrypoint forces it on |

These live in `src/verl_ext/prefix_rft/config/prefix_rft_trainer.yaml`, which is the
**primary** Hydra config. It cannot compose the vendored AgentFlow config as a default:
that file declares `hydra.searchpath`, which Hydra permits only in a primary config. Its
keys are inlined under an `AGENTFLOW BASE` marker instead, and the sync check diffs that
block against the vendored file so a re-vendor cannot change GRPO's setup while leaving
Prefix-RFT on the old one.

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
5. **1358 of 1800 questions carry demonstrations**, of which 1085 are prefixable. Above
   the 10% and 1% regimes Table 2 validates.
6. **Schedule spans the actual run length** rather than the paper's fixed 500 steps.
7. **Three copies** are taken from code this project does not own (verl's `update_policy`,
   the vendored `_train_step`, the vendored config keys), each guarded by a sync check
   rather than only a comment.

## Testing

**Unit** (`tests/unit/test_prefix_rft_*.py`, CPU, no GPU):

| Target | Assertion |
|---|---|
| `schedule.py` | `low_t` is 0.95 at step 0, decays monotonically, reaches 0.05 at the final step; `l` always lies in `[low_t, 0.95]` |
| step discretisation | `k` in `[0, m-1]` for all `m >= 1`; `k = 0` whenever `m = 1`; `k` decreases in expectation as training advances |
| `build_prefix_demos.py` | one row per decision in order, thinking stripped, tool results attached to the calling step, incorrect trajectories dropped, and questions sharing an `idx` kept distinct |
| `check_prefix_demos.py` | accepts the real shape (plan, tool calls, answer) and a single-decision row; rejects a missing tool result, an unparseable call, a non-tool middle step, a trajectory ending on a tool call, and surviving thinking |
| `demos.py` | store loads, looks up by question text, is whitespace-insensitive, and misses return `0` rather than raising |
| `dispatch.py` | only the first rollout is prefixed; validation never is; `m <= 1` and unknown questions give `k = 0`; the schedule is asked for the current step; lookup passes the question, not an index |
| `masks.py` | mask is 1 exactly on replayed-triplet response tokens, truncates with the response, and drops exactly the rows the base daemon drops |
| `advantage.py` | on a hand-built group of 8, prefix tokens equal `score_hybrid - mean(unprefixed)`; grouping is per rollout not per row; the hybrid is excluded from the on-policy baseline; a failing prefix gets a negative advantage |
| `entropy.py` | exactly the top 20% survive, selection is global across the micro-batch, the highest-entropy tokens are the survivors, no non-prefix token is touched, input is not mutated |
| `prefix_replay.py` | first `k` calls return teacher text and call `k+1` delegates; tokenisation matches the proxy's two calls; the `tool`→`user` remap is applied; a live tool call after replay is not served a stale result |
| copy guards | the copied `update_policy` and `_train_step` still match their sources; edit anchors fail loudly if not unique |

Tests needing verl are skipped on CPU via `importorskip`; the same checks run as scripts
under `cosmas-train`, which has verl but no pytest.

**Gate.** `check_prefix_demos.py` over the full store, plus
`check_prefix_rft_trainer_sync.py` over the three copies, wired into
`009_run_tests_for_prefix_rft.job` on the CPU partition.

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

The production 4xH100 run, the five-benchmark evaluation suite, and reporting the
results. Those follow only if the staged verification path is clean and the compute is
worth spending.

## Implementation record

Eight of the plan's ten tasks are built and committed. What follows is every place the
implementation departed from the design above, and why. Each was found by running
something, not by reading.

### Corrections to the design

1. **The lookup key was wrong.** The design keyed demonstrations on `extra_info.idx`.
   That field is assigned per data source and collides across them, so a maths
   demonstration would have been replayed into a search question, with no downstream
   signal. Now keyed on the question text. Covered by
   `test_questions_colliding_on_idx_stay_distinct`.

2. **Coverage was understated at 700.** That figure came from `sft_train.parquet`, already
   thinned by a train/val split and a math:search rebalance Prefix-RFT does not need.
   Building from the collection gives 1358 questions and 4047 decisions.

3. **The gate encoded the wrong rule.** "Every non-final step is a tool call" rejected
   1087 of 1360 rows, because the first decision is the planning turn. Corrected, and the
   real shape measured and documented.

4. **Advantage grouping had to move from rows to rollouts.** Flow GRPO emits one row per
   turn; a row-level port would have centred the hybrid rollout's turns against themselves
   and produced exactly zero prefix advantage. Recorded in section 5 and tested.

5. **A stale tool result could reach a live tool call.** The replay controller originally
   returned the last stored result on every lookup, so the policy's first genuine tool call
   after replay ended would have been served the teacher's result instead of executing.
   Now single-use, with a regression test.

6. **The vendored Hydra config cannot be composed.** It declares `hydra.searchpath`, which
   Hydra permits only in a primary config. Its keys are inlined and drift-checked instead.

7. **`torch.std` is NaN for a one-element group.** Falls back to 1.0, matching the
   reference's singleton branch.

### Structural decisions taken during implementation

8. **Logic is separated from verl.** verl is absent from `agent_engine` and pytest is
   absent from `cosmas-train`, so any module importing verl is untestable in both. Every
   piece of real logic therefore lives in a verl-free module, with thin wrappers over it.
   This is why `dispatch.py`, `masks.py`, `entropy.py` and `prefix_replay.py` exist and are
   not folded into the classes that use them.

9. **The copies are guarded, not just documented.** The edits live as data and are
   re-derived from the current sources by `check_prefix_rft_trainer_sync.py`. The copied
   bodies were generated by applying those edits at asserted-unique anchors rather than
   hand-transcribed.

10. **Two identity seams were added to `OrchestratorRollout`.** `_wrap_provider` and
    `_wrap_tools` give the replay shims somewhere to attach. Both return their argument
    unchanged in the base class, and the existing GRPO rollout suite gates that.

11. **The entrypoint fails loudly on three configuration mistakes** rather than
    misbehaving quietly: a non-FSDP strategy, `rollout.mode != async`, and an empty
    demonstration store (which would make every rollout plain GRPO while the run still
    reported success).

### Verification of the highest-risk assumption

12. **Replay tokenisation is verified.** Replayed turns never reach vLLM, so their token
    IDs are produced locally by `ReplayController`; the daemon's proxy produces them for
    generated turns. A disagreement would misalign every prefix triplet, and training
    would proceed normally and report success, so nothing in a run would reveal it.

    `scripts/check_prefix_replay_tokenisation.py` compares the two directly on real
    demonstrations with the real Qwen3-8B tokenizer, including the provider's
    `tool`→`user` remap. It passes on 10 demonstrations, runs on CPU in seconds, and is
    stage 2b of `009_run_tests_for_prefix_rft.job`. This did not need the GPU smoke run:
    the check is against the proxy's code path (`daemon.py:216-225`), not against a live
    server.

    What remains outside its scope is whether the proxy's own tokenisation matches what
    vLLM used to generate. That assumption is the vendored proxy's, is shared by every
    on-policy triplet in the existing GRPO pipeline, and is not something Prefix-RFT
    introduces.
