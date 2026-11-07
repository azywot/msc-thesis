# Recursive self-improvement: prompt search, distillation, repeat

**Date:** 2026-11-04
**Status:** draft — not yet approved. Decisions still open are listed under
"Open decisions" and marked **(open)** where they appear.
**Papers:**
- MASS — `docs/papers/2610.12176v1_recursive-self-improvement-multi-agent-self-supervision.md`
  (Lee et al., arXiv:2610.12176v1). Code: github.com/SakanaAI/mass
- MATPO — `docs/papers/2510.04678v1_multi-agent-tool-integrated-policy-optimization.md`
  (Mo et al., arXiv:2510.04678v1). Code: github.com/mzf666/MATPO

## Goal

Add a recursive self-improvement (RSI) loop to CoSMAS. A cycle has two steps:

1. **Search.** The current model improves the prompt its own agent team runs under.
2. **Internalize.** The current model generates trajectories under that improved prompt,
   the prompt is removed, and the model is trained on the trajectories, so the improved
   behaviour no longer needs the prompt.

The trained model then plays every role again in the next cycle. This is MASS adapted to
CoSMAS. MATPO is a second, RL-based way of doing step 2 that also trains the sub-agent
role (see "Internalization variant B").

The thesis question this serves: **can a small multi-agent system improve itself using
only its own outputs and a verifier, and what behaviour does it internalize when it does?**
Whether this becomes a headline method or an extension of the existing "does
collaboration help small models" story is **(open)**, decision D1.

Success means:
- two complete cycles run end to end on the 8B path;
- each cycle's model is evaluated on the five standard benchmarks under the **seed**
  prompt (no optimized prompt at inference);
- the result, positive or negative, is attributable: we can say whether gains come from
  the prompt search, from internalization, or from neither.

A flat or negative result is an acceptable outcome. A result we cannot attribute is not.

## What MASS does, and how it maps onto CoSMAS

MASS (§3, Algorithms 1–3) alternates:
- **Workflow optimization.** Per task, the model proposes a workflow prompt. The prompt
  covers sub-agent roles, instructions, output contracts and call order. The model
  executes it, compares the result pairwise against the best so far, and revises using
  the evaluator's feedback history. This is an evolutionary search with an elite of one.
- **Internalization by SFT.** The best workflow is executed M times, ranked by a
  Bradley–Terry tournament judged by the model itself, and the top K−1 trajectories become
  SFT data **with the workflow removed from the prompt**. Orchestrator and sub-agent
  conversations are both trained, on the same weights.

| MASS component | CoSMAS equivalent | Status |
|---|---|---|
| Workflow optimizer + feedback history | GEPA (`scripts/run_gepa.py`, `src/gepa_integration/`); the reflector reads traces and failure labels | exists; reflector is Qwen3-32B by default, configurable via `reflector.path_or_id` |
| Workflow prompt `w` | GEPA candidate: `system_prompt` + `planning_suffix` | exists |
| Self-evaluator (pairwise, Bradley–Terry) | Ground-truth scoring via `evaluate_answer()` | exists — **not replaced by a self-judge**, see "Divergences" |
| Execute best workflow M times, keep top K | `scripts/collect_sft_data.py --n-samples N`, which keeps correct trajectories | exists; needs to load a GEPA candidate (see Step 3) |
| Remove `w*` before SFT | `scripts/build_sft_parquet.py` folds each trajectory into rows, re-rendering the user turn and copying the system message | needs a prompt-swap option (see Step 4) |
| SFT on shared weights | Memory-folded SFT pipeline (`docs/pipelines/sft.md`) | exists |
| Train sub-agent conversations too | Not captured today; sub-agent calls are single `generate()` calls inside the tools | new (see variant B and stretch S2) |

## Divergences from MASS (decided)

1. **Verifier instead of self-judge.** MASS uses self-evaluation because its tasks
   (open-ended research) have no ground truth. Ours do. Selection uses `evaluate_answer()`
   both in the GEPA search and when filtering trajectories. So the "self" in this design
   is the **optimizer** (reflector) and the **data generator**, not the judge. This is
   stated in the thesis as a scope choice, not hidden.
2. **Per-domain prompts, not per-task workflows.** MASS searches a workflow for each of 8
   long research tasks. We have thousands of short questions; a per-question search would
   overfit and cost too much. GEPA already optimizes one prompt per preset (`gaia`,
   `math`), and that is the unit here.
3. **Workflow = existing prompt components.** We do not introduce a new workflow DSL with
   roles and contracts. The orchestrator's sub-agents are fixed tools (web_search,
   code_generator, …). Their "output contracts" are the tools' sub-agent prompts, which
   stretch S1 makes searchable.
4. **Reflector identity is a variable.** MASS §5.1 finds the optimizer, not the evaluator,
   is the bottleneck. The main arm uses the cycle-g model as its own reflector (faithful).
   A control arm keeps Qwen3-32B as reflector (heterogeneous). See "Arms".

## The cycle

Notation: `M_g` is the orchestrator model after g cycles. `M_0` is base Qwen3-8B.
P_seed is the `PromptBuilder` prompt (system prompt + default planning suffix), which
is what every existing run uses.

**Step 1 — Serve `M_g`.** Merge the previous cycle's LoRA (`scripts/merge_lora.py`) and
serve it with vLLM. For g = 0 this is base Qwen3-8B.

**Step 2 — Prompt search (GEPA).** Run `scripts/run_gepa.py` for both presets (`gaia`,
`math`), seeded from P_seed. `M_g` is the executor. The reflector is `M_g` itself (main
arm) or Qwen3-32B (control arm), set through `reflector.path_or_id/host/port`. Budget and
data splits are unchanged from `experiments/configs/gepa/*.yaml` (150 `D_feedback` / 50
`D_pareto`, `rollout_budget: 750`). Output: P*_g,gaia and P*_g,math.

The GEPA seed for cycle g is always P_seed, **not** P*_{g−1}. Reason: the comparison of
interest is "how much does the search add on top of `M_g`", and seeding from the
previous prompt would mix the two cycles' searches. *(Builder's call to revisit only if
GEPA clearly stalls from P_seed at g ≥ 1.)*

**Step 3 — Collect trajectories under P\*.** Run `scripts/collect_sft_data.py` with `M_g`
as orchestrator over `data/training/train/combined_train.parquet` (1,800 questions,
`data_source` = `hotpotqa` 765 / `nq` 135 / `deepmath` 900). `hotpotqa` and `nq` rows use
P*_g,gaia; `deepmath` rows use P*_g,math. Keep correct
trajectories; use `--n-samples 3`, which spends extra passes only on still-unsolved
questions. Required change: `collect_sft_data.py` must accept a GEPA candidate per data
source (system prompt **and** planning suffix), in the same way `gepa_prompt_path` works
for `run_experiment.py`. Today it only supports a `system_prompt_suffix`.

**Step 4 — Build folded SFT rows with P\* swapped back to P_seed.** This is the MASS
"remove w\*" step, and it is where the known SFT < base bug lives, so it is specified
exactly:
- Each row's system message is replaced by the P_seed system prompt for that row's data
  source. It is the same prompt `PromptBuilder` renders at inference, tools section
  included.
- `_fold_trajectory` is called with P_seed's planning suffix (`_DEFAULT_PLANNING_SUFFIX_TOOLS`),
  not P*'s.
- The memory-prompt user turns need no change. `_memory_user_content` re-renders them
  from question, plan and action history, none of which contain prompt text.
- A check script (in the style of `scripts/check_sft_folded_format.py`) asserts that no
  substring unique to P\* appears in any row, and that row k's
  `[system, user]` equals what `AgenticOrchestrator` would build under P_seed.

**Step 5 — SFT `M_g` → `M_{g+1}`.** Use the existing SFT pipeline unchanged: LoRA rank
64, folded rows, with the hyperparameters of the current SFT config. Training starts from
`M_g`, so cycles stack. Whether to merge the LoRA into the weights or train a fresh LoRA
on `M_0` with the union of data is left to the builder, as long as it is recorded.

**Step 6 — Evaluate `M_{g+1}`.** See "Evaluation". Then g ← g + 1 and repeat from Step 1.
Two cycles (g = 0, 1) are in scope.

## Arms

| Arm | Reflector | Internalization | Purpose |
|---|---|---|---|
| **A — self-RSI (main)** | `M_g` | SFT (Steps 3–5) | The MASS analogue |
| **B — 32B-reflector RSI** | Qwen3-32B every cycle | SFT | Is the small model's own search the bottleneck? (MASS §5.1) |
| **C — no-search control** | none (P\* := P_seed) | SFT on `M_g`'s own correct trajectories under P_seed, same two cycles, same collection settings | Rejection-sampling self-training (STaR / expert iteration). Isolates what the search contributes beyond self-filtering |
| **D — MATPO** (variant B below) | none or `M_g` | RL, orchestrator + sub-agent roles | RL-based internalization that also trains sub-agents |

Already-existing references to compare against: base `M_0`; `M_0` + GEPA (32B reflector)
prompt; the 32B-teacher SFT model; the Flow GRPO model.

Arms A and C are required. B is required if budget allows, since it is the cheapest way
to make a negative result in A interpretable. Whether D runs at all is **(open)**, D3.

## Internalization variant B: MATPO

MATPO trains planner and worker roles in **one** model with RL. The worker's tokens enter
the same GRPO objective as the planner's, and the planner rollout's advantage is broadcast
to all its worker rollouts.

**It fits the existing RL pipeline almost directly:**
- `OrchestratorRollout` (`src/fine_tuning/rollout.py`) already turns every orchestrator
  `generate()` call into a `Triplet` with the trajectory's final reward. This is Flow GRPO,
  and it is MATPO's broadcast for the planner side.
- `_CapturingProvider` records prompt/response token IDs from any `OpenAIProvider` pointed
  at the VERL endpoint, so sub-agent calls can be captured the same way.
- `_wrap_tools(registry, task)` is a per-episode seam, already used by Prefix-RFT.
- CoSMAS sub-agent calls are **single-turn**: one `generate()` per web_search analysis or
  code_generator call. MATPO's worker is multi-turn, so ours is the simpler case.

**Changes:**
1. In MATPO mode, build the sub-agent provider per episode, pointed at the **VERL
   endpoint** (the policy) instead of `SUBAGENT_ENDPOINT`, and wrapped in a
   `_CapturingProvider`. The tool registry is currently built once and shared across
   concurrent episodes. MATPO mode therefore needs per-episode tool instances (sharing the
   `CacheManager`), or captured turns from different episodes will mix.
2. Sub-agent sampling uses the training temperature (0.7), not the current greedy 0.0.
   Otherwise the worker role has no exploration.
3. Sub-agent turns become extra `Triplet`s with the same `reward_value` and
   `metadata={"role": "subagent"}`. Orchestrator triplets get `"role": "orchestrator"`.
4. Log per-role metrics: token counts, response length, and the share of the loss from
   sub-agent tokens.
5. Retire the frozen 1.7B sub-agent server for this mode. This frees GPU 0's ~7.5 GB, but
   all generation now runs on the VERL engine, so expect slower steps.

**Credit assignment.** Decided: keep the existing GRPO estimator and grouping (`uid` per
question) and simply add the sub-agent triplets. This keeps arm D directly comparable to
the existing Flow GRPO run. Known caveat: normalization is per triplet, so trajectories
with more calls weigh more. MATPO normalizes per full rollout and then broadcasts. A
per-rollout normalization switch is a stretch item, not v1.

**Hard dependency:** MATPO-in-one-model requires the orchestrator and sub-agents to be the
**same model**. Every current CoSMAS configuration uses an 8B orchestrator with **1.7B**
sub-agents (GEPA, SFT collection, RL, all `*_inference` configs). Arm D is therefore only
possible under decision D2 = homogeneous 8B. The alternative, training a separate 1.7B
policy alongside the 8B one ("multi-agent-multi-model" in the MATPO paper), is out of
scope.

**Two MATPO findings worth adopting as ablations, not defaults:**
- *Query recap.* Our web_search sub-agent sees only the search query
  (`WebSearchTool.build_analysis_prompt(query, results)`), not the user's question. MATPO
  finds that adding the original question to the worker prompt helps substantially.
- *Worker final summary.* Our web_search sub-agent already returns a `**Final
  Information**` block, which plays this role. No change needed.

## Evaluation

**Benchmarks.** The five used for the SFT and LoRA inference configs: AIME, GAIA, GPQA,
HLE, MuSiQue, with the same splits, `thinking_mode` and sub-agent setup as
`experiments/configs/qwen3/sft_inference/`. New configs go under
`experiments/configs/qwen3/rsi_inference/` and are generated by
`scripts/generate_configs.py`, not hand-written.

**For every model `M_g` in every arm, evaluate two conditions:**
- under P_seed — the headline. This is "has it internalized it";
- under that cycle's P\* — the ceiling the search found. The gap between the two measures
  how much was internalized.

**Metrics:**
- accuracy per benchmark (`metrics.json`);
- **failure-mode distribution** from `classify_failure` (`scripts/failure_modes/`) on
  every run. The key number is the share of "direct reasoning without action". This is
  our analogue of MASS Table 2, where delegation under bare prompts rose from 1/110 to
  87/110;
- tool-call rate and turns per question;
- tokens per question, since MASS reports gains per output token;
- per-cycle collection yield: correct trajectories per question at Step 3, for P\* vs P_seed;
- GEPA search curve per cycle: best `D_pareto` score against rollouts spent. If `M_1` is
  a faster self-optimizer than `M_0`, this is where it shows (MASS Fig. 3).

**Seeds/variance.** At least the repetition count the existing inference configs use.
Report differences with the same statistical treatment as the existing thesis tables.

## Constraints

- **Format rule.** Training rows must equal inference prompts byte for byte
  (`docs/pipelines/sft.md`). Step 4's check script is mandatory, not optional.
- **No test leakage.** GEPA, collection and RL data come only from Search-R1, DeepMath
  and the GEPA presets. None of the five evaluation benchmarks is used anywhere in the
  loop, including as GEPA `D_pareto`.
- **Cluster.** Snellius. Each step is a separate SLURM job under `jobs/`; a cycle is a
  chain of jobs, not one long job. Outputs per cycle go under
  `experiments/results/rsi/<arm>/cycle_<g>/` with the P\* candidates, collected
  trajectories, SFT parquet, adapter path and eval runs, so any cycle can be re-run from
  the previous cycle's adapter.
- **Code style.** black (100), isort. New CLI flags over new scripts where an existing
  script already does 90% of the job (`collect_sft_data.py`, `build_sft_parquet.py`,
  `rollout.py`).

## Non-goals

- A self-judge or Bradley–Terry ranking. We have ground truth (divergence 1).
- Per-question workflow search.
- New tools or new sub-agent types.
- Training the 1.7B sub-agent as a separate policy.
- More than two cycles.
- Prefix-RFT as the internalization step. It could be added later by using P\*
  trajectories as prefixes, but not here.

## Anti-goals (would make this a failure even if it runs)

- Any training row whose prompt differs from what inference builds. This is the SFT < base
  bug again.
- Reporting `M_g` only under P\*: that measures prompting, not self-improvement.
- An evaluation benchmark leaking into GEPA or collection data.
- Results that cannot say which step produced the change. Arm C exists for this reason.

## Stretch items (only after A and C have run)

- **S1 — search the sub-agent prompts.** Add the web_search and code_generator sub-agent
  prompts as extra GEPA components. This is MASS's "communication structure" emphasis. It
  needs `AgentGEPAAdapter` to inject per-tool prompt overrides. A query-recap variant is
  the obvious first mutation.
- **S2 — joint SFT on sub-agent rows.** Under homogeneous 8B, log sub-agent prompt/response
  pairs per trajectory, which is not stored today, and add them as SFT rows. Run at a
  matched supervised-token budget against orchestrator-only SFT. This is MASS Hypothesis 2
  (their 68.3% vs 64.0% at 1.4× tokens).
- **S3 — per-rollout advantage normalization** for arm D.

## Left to the builder

- Job-chaining mechanics and the exact directory layout under `experiments/results/rsi/`.
- LoRA stacking strategy across cycles (Step 5), provided it is recorded.
- How P\* candidates are passed into `collect_sft_data.py` (CLI flag vs config field).
- How the reflector is served when it is `M_g`: a second vLLM instance on another GPU, or
  the same server. The executor and reflector use the same weights, so the CLAUDE.md note
  on sharing `path_or_id` applies. Whether the self-reflector thinks (the 32B reflector is
  served with `--enable-thinking`) is also the builder's call; record it.
- Test coverage. At minimum: a unit test that a P\* trajectory folds into rows identical
  to a P_seed trajectory with the same actions; a smoke test of MATPO-mode triplet capture
  (sub-agent triplets present, roles tagged, no cross-episode mixing).

## Open decisions

- **D1 — Framing.** Is RSI a headline contribution or an extension chapter of the existing
  collaboration study? This affects how much of arms B/D and the stretch items is worth
  the compute.
- **D2 — Sub-agent model.** Keep the 8B orchestrator with 1.7B sub-agents (comparable to
  every existing result; RSI then improves the orchestrator only), or move to homogeneous
  8B everywhere (faithful to MASS and MATPO; needed for arm D and S2; needs a new
  homogeneous baseline run).
- **D3 — Whether arm D (MATPO) runs.** One RL run is ~40 h on 4×H100, plus a cycle-2 run
  if it is done recursively.
- **D4 — Thinking mode for the loop.** GEPA and RL use `NO`; SFT collection used
  `ORCHESTRATOR_ONLY` (32B teacher). Proposed default: `NO` throughout, matching GEPA, RL
  and the 8B `sft_inference` configs.
