# Recursive self-improvement: prompt search, distillation, repeat

**Date:** 2026-11-04, revised 2026-11-08
**Status:** draft, revised after a design review with the author. Not yet approved.
Items marked *(builder's call)* are left to the implementer; items marked *(Claude's call)*
were proposed during the review and should be checked by the author.
**Papers:**
- MASS: `docs/papers/2610.12176v1_recursive-self-improvement-multi-agent-self-supervision.md`
  (Lee et al., arXiv:2610.12176v1). Code: github.com/SakanaAI/mass, local copy at
  `/gpfs/home3/xchen1/azywot/repos/mass`.

## Goal

Add a recursive self-improvement (RSI) loop to CoSMAS. A cycle has two steps:

1. **Search.** The current model, acting as reflector, improves the prompts its own agent
   team runs under: the orchestrator's prompts and the sub-agents' prompts.
2. **Internalize.** The current model generates trajectories under those improved prompts,
   the prompts are swapped back to the seed prompts, and the model is trained on the
   trajectories, so the improved behaviour no longer needs the improved prompts.

The trained model then plays **every** role in the next cycle: orchestrator, every
sub-agent, and reflector. The multi-agent system at cycle g is one fine-tuned 8B model in
all seats. This is MASS adapted to CoSMAS.

**Framing (decided).** RSI is a further fine-tuning direction, compared side by side with
the existing fine-tuning directions (32B-teacher SFT, Flow GRPO, LoRA variants). It is not
a separate headline contribution. The question it answers: **can a small multi-agent
system improve itself using only its own outputs and a verifier, how does that compare
with the other fine-tuning directions, and what behaviour does it internalize?**

Success means:
- two complete cycles run end to end on the homogeneous 8B setup;
- each cycle's model is evaluated on the five standard benchmarks under the **seed**
  prompts (no optimized prompt at inference);
- the result, positive or negative, is attributable: we can say whether gains come from
  the prompt search, from internalization, or from neither, and whether a weak 8B
  reflector was the limiting factor.

A flat or negative result is an acceptable outcome. A result we cannot attribute is not.

## What MASS does, and how it maps onto CoSMAS

What the MASS code actually does (from `repos/mass`, not only the paper):

- **Workflow = a team definition.** A JSON list of agents (orchestrator plus specialists),
  each with a `role` and an `instruction`, embedded in the task prompt
  (`tasks/initial/query<ID>_ourTeam.txt`). The optimizer returns a full replacement of
  this block every round.
- **Optimizer = the current model.** `harness_improvement/iterate_multi_agent_prompt.py`,
  run with Qwen3.6-27B, thinking on, T=0.6, top_p 0.95, top_k 20, 49K output tokens,
  262K context. Its input: the task; the current, best-so-far and original (v0) designs;
  the run's deliverables (report, `results/*.json`, up to 4 code files, all truncated to
  character caps); the run status; and a history of pairwise verdicts against a fixed
  bare-task run, with the design diff behind each verdict and a "recurring issues"
  checklist. It does **not** read the raw conversation or tool transcripts.
- **Selection.** A proposal is compared with the fixed bare-task reference by the
  model's own pairwise judge; winners are arbitrated against the champion. Elite of one,
  11 rounds per task. A task whose best workflow never beats the bare run is dropped
  from collection (`beats_reference`).
- **Internalization.** The retained workflow is executed 18 times, ranked by a
  Bradley–Terry tournament judged by the model itself, ranks 1–15 train and rank 16
  validates. The workflow is removed only from the orchestrator's first prompt; worker
  conversations are trained as well (about 2/3 of sampled windows). Each cycle trains a
  **fresh** LoRA (r 64, alpha 128) on the merged parent model, then merges it.

| MASS component | CoSMAS equivalent | Status |
|---|---|---|
| Optimizer, run by the current model | New reflector loop, run by `M_g` (main arm) or Qwen3-32B (control arm) | new; replaces GEPA |
| Workflow `w` | A **prompt bundle**: orchestrator guidance, worked example, turn-0 prompt, turn-1..N prompt, sub-agent prompts | new representation; see "The prompt bundle" |
| Pairwise self-judge + Bradley–Terry | Ground-truth scoring via `evaluate_answer()` | exists; not replaced by a self-judge |
| Execute best workflow M times, keep top K | `scripts/collect_sft_data.py --n-samples N`, keeps correct trajectories | exists; must load a prompt bundle |
| Remove `w*` before SFT | `scripts/build_sft_parquet.py` folds trajectories into rows | needs bundle swap-back, including sub-agent rows |
| SFT on shared weights, workers included | Memory-folded SFT pipeline (`docs/pipelines/sft.md`) | exists for the orchestrator; sub-agent rows are new |

## Divergences from MASS (decided)

1. **Verifier instead of self-judge.** Our tasks have ground truth; MASS's open-ended
   research tasks do not. `evaluate_answer()` scores both the search and the trajectory
   filter. The "self" in this design is the **optimizer** and the **data generator**, not
   the judge. Stated in the thesis as a scope choice.
2. **One prompt bundle per domain, not per task.** We have 1,800 short questions, not 8
   long tasks. A per-question search would cost far more than the rest of the cycle and,
   with gold answers in view, could write a question's answer into its own prompt. There
   are two domains: **search** (HotpotQA + NQ rows) and **math** (DeepMath rows). This
   matches the existing seed-prompt split (`collect_sft_data.py:_data_source_to_prompt_dataset`
   maps `hotpotqa`/`nq` to `gaia` and `deepmath` to `deepmath`), so each optimized bundle
   has exactly one seed bundle to swap back to. A single general bundle is a possible
   ablation, not the default.
3. **Workflow = existing CoSMAS prompt components.** No new workflow DSL. The sub-agents
   are the fixed tools (web_search, code_generator, ...); their "instructions and output
   contracts" are the tools' sub-agent prompts, which are editable here.
4. **The reflector sees the gold answer; leakage is blocked by checks instead.** MASS has
   no gold answers. Our reflector is shown correct/incorrect **and** the gold answer,
   because an 8B reflector needs the clue (format mismatch versus wrong fact). Leakage is
   prevented by the guards in "Leakage guards".
5. **The reflector reads sub-agent inputs and outputs.** MASS reads deliverables only.
   To improve sub-agent prompts our reflector must see what each sub-agent was given and
   what it returned. See "Reflector input".
6. **A smaller optimizer gets a smaller job.** MASS's optimizer is 27B with a 262K window.
   Ours is 8B with a 32K native window. It edits one unit per round, sees pre-computed
   failure labels, and several candidates are sampled per round. See "The search loop".

## The prompt bundle

The bundle is everything the reflector can change. Everything else is protected
scaffolding that `PromptBuilder` and the tools insert, and the reflector sees it read-only.

**Editable units:**

| Unit | Parts | Where it appears | Seed value |
|---|---|---|---|
| **Orchestrator** | (a) system guidance text | system message, every turn | today's `base_instruction_tools` + `final_instructions_tools` (or the `_subagent` / `_direct` variant actually rendered) from the domain's template |
| | (b) worked example, as structured fields: question, steps (reasoning, sub_goal, tool_call, tool_response), final answer | system message, every turn | the example variant this run's tool set renders (`_select_and_format_example`) |
| | (c) turn-0 prompt | appended to the user turn of the planning turn | `_DEFAULT_PLANNING_SUFFIX_TOOLS` |
| | (d) turn-1..N prompt (new "acting suffix") | appended to the end of each memory user turn | empty string |
| | (e) tool descriptions: the `description` strings of each enabled tool and of each of its parameters | `<tools>` block of the system message, every turn | today's schema descriptions (`web_search.py:104,112`, `code_generator.py` sub-agent-mode schema) |
| | (f) tool-use note: the "Important: You may call tools zero, one, or multiple times..." paragraph after the `<tools>` block | system message, every turn | today's text (`prompts/builder.py:207`) |
| **web_search sub-agent** | the analysis prompt template | `WebSearchTool.build_analysis_prompt` | today's text |
| **code_generator sub-agent** | the task prompt template | `CodeGeneratorTool.build_task_prompt` | today's text (the `return_code: false` branch) |

The six orchestrator parts are coupled (the plan shapes the actions, the guidance and tool
descriptions shape both), so they are one unit and are edited together; the reflector may leave any part
unchanged.

**Editable text is free, as in MASS** (decided). The reflector may rewrite any editable
part entirely, including the opening role line ("You are a reasoning assistant with
tools..."). There is no length cap and no required structure beyond the validation
below. The prompt length of each version is logged, and the reflector input budget
limits growth in practice.

**Protected (never editable)** (decided). The rule: protect what code parses or what
makes up the AgentFlow architecture; everything that is only advice is editable.
- Tool schema structure: tool names, parameter names, types and `required` fields (the
  registry dispatches on them). Only the `description` strings are editable, part (e).
- The tool-call syntax and its format instructions (`<tools>` header, the
  `<sub_goal>`/`<tool_call>` format paragraph); the tool-call parser depends on them.
- The `<sub_goal>` tag convention: `_extract_sub_goal` copies it into the structured
  memory's action history.
- The `\boxed{}` answer rule (`final_instructions`): `orchestrator.py:625` uses it to
  detect a final answer and `utils/parsing.py:150` extracts it for scoring.
- The memory-turn structure (question, `**Query Analysis:**`, `**Previous Steps:**`,
  the `Action Step N` blocks). This is the architecture, not a prompt; editing it would
  confound RSI with the structured-memory ablation. Part (d) is the lever on acting turns.
- The attachment notes: never triggered in training (no attachments), so there is no
  evidence to edit them from.

**Sub-agents in scope.** Only sub-agents actually called in training trajectories are
editable. The training data has no attachments, so `text_inspector` is never called and
its prompt stays at seed. `mind_map` and `image_inspector` are not enabled in the
evaluation configs. *(Claude's call.)*

**Validation of every candidate, before it is run:**
- Sub-agent templates keep all their placeholders (e.g. `{query}`, `{formatted_results}`,
  `{task}`, `{context_block}`), each at least once (the seed web_search template uses
  `{query}` twice), and introduce no unknown `{name}` placeholders.
- The worked example parses: every `tool_call` is valid JSON, names a tool enabled in the
  run, and uses that tool's argument names for the current mode (`task` in sub-agent
  mode, `code` in direct mode). It is rendered by `PromptBuilder`, so the call syntax is
  still produced by code.
- Tool descriptions: the edited schemas are identical to the seed schemas except for
  `description` strings (same tools, names, parameters, types, `required`); every
  description is non-empty.
- The leakage guards below pass.
A failing candidate is logged with the reason and the reflector is re-prompted
(bounded retries; *builder's call* on the number).

**Byte-identity of the seed.** Rendering the seed bundle must produce exactly the prompts
every existing run uses: same system prompt, same planning user turn, same memory user
turns (empty acting suffix adds nothing, not even a newline), same sub-agent prompts.
This is a unit test, and it is what keeps all existing results comparable.

## Leakage guards

The concern: a prompt that contains a training question, its answer, or a worked example
built from one produces "correct" trajectories that SFT then teaches without the
reasoning. Three guards:

1. **Instruction.** The reflector is told to write general strategies only: no question
   text, named entities, numbers or answers from the shown trajectories, and no worked
   example built from a shown question.
2. **Mechanical check on every candidate** (all editable parts, the worked example
   especially):
   - reject if it shares a word n-gram of length ≥ 8 *(builder may tune; record it)* with
     **any** question in `combined_train.parquet`;
   - reject if it contains the gold answer (or any alias) of any question **shown to the
     reflector in this search**. Answers shorter than 4 characters or purely numeric are
     matched only as standalone tokens, to avoid false rejections on "yes" or "2".
   Every rejection is logged with the matching span.
3. **Selection on unseen questions.** The held-out batch that decides whether a version is
   kept is never shown to the reflector (see "Data"). Copying a shown example cannot raise
   that score, so the search does not reward leaking.

## Data

**The training data does not change.** Everything comes from
`data/training/train/combined_train.parquet`: 1,800 questions, `hotpotqa` 765 / `nq` 135 /
`deepmath` 900, the same file the existing SFT and GRPO runs use. No new sources, filters
or re-mixing. The GEPA preset files (`data/gepa/*`) are not used.

For the search, each domain's rows are split once (fixed seed, recorded) into three
disjoint pools:
- **shown pool**: questions whose trajectories the reflector may see;
- **screen pool**: questions used to screen candidates each round; never shown;
- **held-out batch**: fixed set (default 50 per domain) that decides whether a version is
  kept; never shown.
Pool sizes are *builder's call*; record them. The search domain's pools keep the HotpotQA/NQ
ratio of the file.

Trajectory collection (Step 3) still runs over **all 1,800** questions, as the current SFT
pipeline does, including questions the reflector saw; the leakage guards cover that.

**No test leakage.** None of the five evaluation benchmarks is used anywhere in the loop.

## The search loop

Run once per domain per cycle, with `M_g` as executor of every role and as reflector
(main arm) or Qwen3-32B as reflector (control arm). Elite of one, as in MASS.

Each round:
1. Run the champion bundle on a fresh minibatch from the shown pool (default 16).
2. Build the reflector input (below) from those trajectories.
3. Sample K candidates from the reflector (default K = 3). Each candidate names **one
   unit** (orchestrator, web_search, or code_generator) and returns its full replacement,
   plus a short rationale and expected effect. Output is JSON; parse failures are retried.
4. Validate each candidate (placeholders, example, leakage). Drop failures.
5. Screen the surviving candidates on a minibatch from the screen pool (default 16); the
   champion's score on the same minibatch is the bar.
6. Run the best screened candidate on the held-out batch. It becomes the champion if its
   held-out accuracy is strictly higher than the champion's *(tie handling: builder's
   call; record it)*.

Defaults: 10 rounds per domain per cycle. All counts are config fields; the total rollout
count per search is logged. *(Defaults are Claude's call, sized to stay in the same order
of magnitude as the earlier GEPA budget of 750 rollouts; the builder may adjust.)*

**Gate.** If no version beats the seed bundle on the held-out batch, that domain's P\* is
the seed bundle for that cycle and this is recorded (the analogue of MASS
`beats_reference`). Collection for that domain then uses P_seed.

**Seed of each cycle's search.** Always P_seed, not P\*_{g−1}, so each cycle measures what
the search adds on top of `M_g`. *(Revisit only if the search clearly stalls from P_seed
at g ≥ 1.)*

### Reflector input

A fixed input budget that fits Qwen3-8B's 32K native window with room for thinking and
a full JSON unit replacement (default about 16K input tokens, leaving about 16K for
thinking and output; *builder's call*, logged). No YaRN: the same
server runs the executor, and context extension would change the behaviour being
measured.

Contents, in order:
1. Role and task: improve one unit of this bundle for this domain; general strategies
   only (leakage instruction).
2. The current champion bundle, every unit in full; the protected scaffolding (tool
   schemas, format rules) shown read-only.
3. The seed bundle (v0), and the best bundle if it differs from the current one, as in
   MASS (`--include-v0-design`, `--include-best-design`).
4. History: each earlier version's changed unit, a diff, its screen and held-out scores,
   and the rationale it was proposed with. A short "recurring problems" summary across
   rounds (MASS's delta checklist).
5. Per trajectory from this round's minibatch:
   - question, gold answer, prediction, correct/incorrect;
   - the `classify_failure` label (`scripts/failure_modes/analyze_failure_modes.py`);
   - the plan (turn-0 output) and every orchestrator action (reasoning, sub_goal, call);
   - per sub-agent call: the input the orchestrator wrote (query or task), a head-and-tail
     excerpt of the raw tool output under a per-call character cap, and the sub-agent's
     full response.
6. Output contract (JSON schema).

If the budget is exceeded, show **fewer trajectories** before cutting each one harder;
keep a mix of correct and incorrect ones. Reuse the token counting and trimming helpers in
`src/gepa_integration/reflection.py`. Log per round how many trajectories were shown and
how much was truncated.

Reflector sampling and thinking are config fields (see "Thinking mode").

## Phases (decided)

The author runs and inspects prompt optimization on its own before committing to full
RSI. Phase 2 is launched by hand after Phase 1 has been checked; nothing chains from
Phase 1 into collection or training automatically.

### Phase 1: prompt optimization only

- Cycle 0 only: base Qwen3-8B in every role, **8B reflector**, both domains. No
  collection, no training.
- Its own entry point (e.g. `scripts/run_prompt_search.py --config <yaml> --domain
  search|math`; name is the builder's call), one SLURM job per domain, resumable round by
  round. The reflector model, round count and budget are config fields, so a reduced
  "pilot" run and a 32B-reflector run use the same entry point.
- **What it leaves to inspect**, under `experiments/results/rsi/phase1/<run>/<domain>/`:
  - per round: the exact reflector input and raw output, every candidate, each
    validation or leakage rejection with its reason and matching span, screen and
    held-out scores, and which unit was edited;
  - the champion bundle after every round;
  - a human-readable report (markdown) with the search curve (held-out score against
    rollouts), a diff of the final P\* against P_seed per unit, the fully rendered
    prompts under P\* (system prompt, planning turn, one acting turn, each sub-agent
    prompt), rejection counts by reason, and a few trajectories on the same question
    under P_seed and P\*.
- **Check runs:**
  - base 8B under P\* versus P_seed on the held-out batch (computed by the search);
  - optional diagnostic: base 8B on the benchmarks **under P\*** through
    `prompt_bundle_path`, the analogue of the existing GEPA inference runs. The search
    bundle applies to GAIA, HLE and MuSiQue, the math bundle to AIME; GPQA is skipped
    (its own template, covered by neither domain). This is not part of the RSI headline.

**Deciding what comes next** (guidance for the author, who decides):
1. The 8B reflector beats the seed bundle on the held-out batch: run Phase 2 with arms A
   and C.
2. It does not: run Phase 1 again with the **32B reflector** at the same budget.
   - 32B passes: run arms B and C, and arm A's cycle 0. If arm A's search fails the gate
     in both domains, arm A stops there (it has become identical to C) and this is
     reported: "8B cannot yet improve its own team's prompts, 32B can".
   - 32B also fails: revisit the search design before any collection or training.

### Phase 2: full RSI

The cycle below, for the arms chosen. If Phase 1 ran at full budget with the same
settings, cycle 0's Step 2 reuses its P\* instead of searching again.

## The cycle

Notation: `M_g` is the model after g cycles; `M_0` is base Qwen3-8B. P_seed is the seed
bundle for a domain. P\*\_g is cycle g's retained bundle.

**Step 1: Serve `M_g`.** One vLLM instance of `M_g` serves every role: orchestrator,
web_search, code_generator, text_inspector, and the reflector in the main arm. Serve
**full weights** (merged for `lora`, the checkpoint for `full`; see Step 5) via
`path_or_id`, so every role hits the same weights. (Current LoRA `sft_inference` configs
set `lora_adapter_path` on the orchestrator only; the RSI configs must not do that.)

**Step 2: Prompt search.** The search loop above, for both domains. Output P\*\_g,search
and P\*\_g,math.

**Step 3: Collect trajectories under P\*.** `scripts/collect_sft_data.py` with `M_g` in
every role over all of `combined_train.parquet`; search rows use P\*\_g,search, DeepMath
rows P\*\_g,math. Keep correct trajectories, `--n-samples 3`. **Row selection uses the
existing pipeline's rules unchanged, in all three arms** (decided): one trajectory per
question, preferring tool-using ones; rebalanced to the reference 50/50 search:math ratio;
10% held out for SFT validation; planning-turn answers kept. Report the share of no-tool
rows per cycle, which is where drift towards "direct reasoning without action" would show.
Required changes:
- accept a prompt bundle per domain (replaces the current `system_prompt_suffix` path);
- **log every sub-agent call** per trajectory: tool name, the rendered sub-agent prompt,
  the raw tool output, and the response. Not stored today; needed for reflector input
  and for sub-agent SFT rows.

**Step 4: Build SFT rows with P\* swapped back to P_seed.** This is MASS's "remove w\*"
step and where the earlier SFT < base bug lived, so it is exact:
- Orchestrator rows: the system message is the P_seed system prompt for that row's
  domain, as `PromptBuilder` renders it at inference (guidance and worked example back
  to seed). `_fold_trajectory` uses the seed turn-0 prompt. Memory user turns are
  re-rendered by `_memory_user_content` with the seed acting suffix (empty), so it must
  learn to render the acting suffix.
- Sub-agent rows (new): one `[user, assistant]` row per logged sub-agent call, with the
  prompt re-rendered from the **seed** sub-agent template using the logged inputs, and the
  logged response as the target. Only from correct trajectories.
- Mixing: orchestrator and sub-agent rows in one training set. Record the token share
  per role. *(Claude's call, following MASS, which trains workers too. An orchestrator-only
  ablation at matched token budget is stretch S1.)*
- A check script (style of `scripts/check_sft_folded_format.py`) asserts: no substring
  unique to P\* appears in any row; orchestrator row k's `[system, user]` equals what
  `AgenticOrchestrator` builds under P_seed; each sub-agent row's prompt equals what the
  tool builds under the seed template. Mandatory.

**Step 5: SFT `M_g` → `M_{g+1}`.** Existing SFT pipeline (`docs/pipelines/sft.md`).
Training always starts from the full weights of `M_g`, so cycles stack. The training
method is a **config field** (decided), with two values, both already supported:
- `lora`: rank 64 with the existing hyperparameters (`jobs/fine_tuning/007_train_sft_folded.job`,
  2×H100). As in MASS, a **fresh** LoRA each cycle on `M_g`, then merged into full
  weights to give `M_{g+1}`. The merge needs a PEFT merge of the adapter that
  `finalize_sft_run.py` extracts; **`scripts/merge_lora.py` must not be used** on SFT
  checkpoints (`docs/pipelines/sft.md:135`).
- `full`: full-parameter training (`jobs/fine_tuning/007_train_sft_full.job`, 4×H100). The
  archived `best_checkpoint/` is `M_{g+1}` directly; no merge.

The method must be the same in every arm and every cycle of a comparison, and is recorded.
The comparison with the existing SFT direction is like-for-like only against the existing
run of the same method.

**Step 6: Evaluate `M_{g+1}`.** See "Evaluation". Then g ← g + 1. Two cycles (g = 0, 1)
are in scope.

## Arms

| Arm | Reflector | Internalization | Purpose |
|---|---|---|---|
| **A: self-RSI (main)** | `M_g` | SFT (Steps 3–5) | The MASS analogue |
| **B: 32B-reflector RSI** | Qwen3-32B every cycle; executor still `M_g` | SFT | Is the small model's own search the bottleneck? (MASS §5.1) |
| **C: no-search control** | none (P\* := P_seed) | SFT on `M_g`'s own correct trajectories under P_seed, same settings, same two cycles, sub-agent rows included | Rejection-sampling self-training (STaR / expert iteration). Isolates what the search adds beyond self-filtering |

**Arms A and C are required. Arm B** runs whenever arm A's result is flat or negative,
since it is what makes that interpretable, and otherwise only if budget allows.

Which arms run is decided after Phase 1 (see "Phases").

**Reference: homogeneous base 8B.** Base Qwen3-8B in every role (the `M_0` of all arms),
on the five benchmarks under P_seed. All RSI comparisons are against this. These runs
already exist: `experiments/configs/qwen3/agentflow/<benchmark>/qwen8B_subagent_tools_none.yaml`
lists no sub-agent models, and the runner falls back to the orchestrator model
(`src/agent_engine/runner/experiment.py:151`); results under
`experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/`. Reuse them after checking
that splits, tool set and thinking mode match the `rsi_inference` configs; rerun only
what differs.

**Existing references** (use 1.7B sub-agents, so not like-for-like; report them with that
caveat): base 8B orchestrator + 1.7B sub-agents; 32B-teacher SFT; Flow GRPO; GEPA prompts.

## Evaluation

**Benchmarks.** AIME, GAIA, GPQA, HLE, MuSiQue, with the same splits and tool setup as
`experiments/configs/qwen3/sft_inference/`, but with `M_g` in every role. New configs go
under `experiments/configs/qwen3/rsi_inference/` and are generated by
`scripts/generate_configs.py`, not hand-written.

**Benchmarks run under the seed prompts only.** Every model `M_g` in every arm runs each
benchmark with that benchmark's own seed template, exactly as all existing runs do. No
optimized bundle is loaded at benchmark evaluation. This is the headline ("has it
internalized it").

**Internalization gap, measured on the held-out batches.** Per domain and cycle:
- *ceiling*: `M_g` under P\*\_g on the held-out batch (already computed by the search);
- *seed*: `M_g` under P_seed on the held-out batch;
- *recovered*: `M_{g+1}` under P_seed on the same held-out batch.
"Recovered minus seed" against "ceiling minus seed" is how much of the search's gain was
internalized. The held-out batch is never shown to the reflector, but it is part of
`combined_train.parquet` and collection runs over all 1,800 questions, so `M_{g+1}` may
have trained on it: *recovered* is an in-distribution measure, and the benchmarks are the
generalization measure. Bundles are loaded through a new config field (e.g.
`prompt_bundle_path`) that goes **through** `PromptBuilder`, unlike `gepa_prompt_path`,
so protected scaffolding is still generated by code.

**Tool set** *(Claude's call)*. Search and collection use the existing collection tool set
(`web_search`, `code_generator`, sub-agent mode, `max_turns: 7`, as in
`experiments/configs/fine_tuning/collect_sft_qwen32b.yaml`). This matches the AIME, GPQA
and MuSiQue evaluation configs exactly. GAIA and HLE also enable `text_inspector`, whose
schema appears in their system prompt, so their prompts differ from the training rows by
that schema. This is the same known mismatch the existing SFT direction has, kept so the
comparison with it stays like-for-like; the byte-identity checks compare against the
collection tool set. `max_turns` does not enter any prompt.

**Metrics:**
- accuracy per benchmark (`metrics.json`);
- failure-mode distribution from `classify_failure` on every run; the key number is the
  share of "direct reasoning without action" (our analogue of MASS Table 2);
- tool-call rate, turns per question, sub-agent calls per question;
- tokens per question (MASS reports gains per output token);
- collection yield per cycle: correct trajectories per question under P\* vs P_seed;
- search curve per cycle: best held-out score against rollouts spent, and the share of
  candidates rejected by validation or leakage checks. If `M_1` is a better
  self-optimizer than `M_0`, this is where it shows (MASS Fig. 3);
- which units the reflector chose to edit, per round.

**Seeds/variance** (decided, for now): single-seed runs on the full benchmarks, as the
existing inference configs do; plus 5-seed pass@k runs on the text splits (GAIA-text 155,
HLE-text 200; `scripts/pass_at_k.py`) for variance; several seeds on AIME, which is small
and cheap. Same statistical treatment as the existing thesis tables. No compute or Serper
budget is fixed yet; the search defaults may shrink once one is.

## Thinking mode

A config field: separately for the executor roles (the existing `thinking_mode`:
`NO` / `ORCHESTRATOR_ONLY` / `SUBAGENTS_ONLY` / `ALL`) and for the reflector. The value
used is recorded per run and must be the same across arms for a comparison to count. The
executor setting used in collection must equal the one used at the headline evaluation.

**Defaults** *(Claude's call)*:
- **Executor: `NO`.** The folded SFT pipeline only supports thinking off
  (`build_sft_parquet.py` rejects `--keep-thinking` with `--format folded`; the format
  gate fails rows that supervise `<think>`), and every fine-tuning direction RSI is
  compared with was evaluated at `NO`. Other executor modes stay selectable in the config
  but need a new folded format that supervises thinking; that is out of scope for v1.
- **Reflector: thinking on**, as in MASS (`--local-thinking`). This is why the reflector
  input budget defaults to about 16K tokens.

## Constraints

- **Format rule.** Training rows equal inference prompts byte for byte
  (`docs/pipelines/sft.md`), for orchestrator and sub-agent rows. Step 4's check script is
  mandatory.
- **Seed byte-identity.** The seed bundle renders today's prompts exactly (unit test).
- **No test leakage.** As in "Data".
- **Cluster.** Snellius. Each step is a separate SLURM job under `jobs/`; a cycle is a
  chain of jobs. Outputs per cycle under `experiments/results/rsi/<arm>/cycle_<g>/`:
  the pools, every search round (candidates, validation results, scores, reflector
  input/output), P\* bundles, collected trajectories, SFT parquet, adapter and merged-model
  paths, eval runs. Any cycle can be re-run from the previous cycle's merged model.
- **Code style.** black (100), isort. New CLI flags over new scripts where an existing
  script already does most of the job (`collect_sft_data.py`, `build_sft_parquet.py`).

## Code changes (summary)

- Prompt bundle type: load/save, render through `PromptBuilder`, validate.
- `PromptBuilder`: accept overrides for guidance text, worked-example fields, tool
  `description` strings and the tool-use note.
- `AgenticOrchestrator`: optional `acting_suffix`, appended to memory user turns.
- `WebSearchTool`, `CodeGeneratorTool`: sub-agent prompt templates overridable, seed
  rendering unchanged.
- Sub-agent call logging in trajectories.
- New reflector loop (module name and location *builder's call*), reusing
  `evaluate_answer`, `classify_failure`, and `src/gepa_integration/reflection.py` trimming,
  with a standalone entry point and a markdown report for Phase 1.
- `collect_sft_data.py`: per-domain bundles, sub-agent logging.
- `build_sft_parquet.py`: bundle swap-back, acting suffix, sub-agent rows.
- Check script, configs in `generate_configs.py`, SLURM jobs.

## Non-goals

- GEPA as the optimizer.
- MATPO or any RL-based internalization (may return later).
- A self-judge or Bradley–Terry ranking (we have ground truth).
- Per-question workflow search.
- New tools, new sub-agent types, or changes to tool schemas and output format.
- New training data or changes to `combined_train.parquet`.
- Mixed-size teams (1.7B sub-agents) inside the RSI loop.
- More than two cycles.
- Prefix-RFT as the internalization step.

## Anti-goals (would make this a failure even if it runs)

- Any training row whose prompt differs from what inference builds under P_seed.
- A seed bundle that does not render today's prompts exactly.
- Reporting `M_g` only under P\*: that measures prompting, not self-improvement.
- An optimized prompt that contains training questions, their answers, or a worked
  example built from one.
- An evaluation benchmark leaking into search or collection data.
- Results that cannot say which step produced the change (arms B and C exist for this).

## Stretch items (only after A and C have run)

- **S1: orchestrator-only SFT** at a matched supervised-token budget, against the joint
  orchestrator + sub-agent SFT (MASS Hypothesis 2).
- **S2: single general bundle** across both domains, against the search/math split.
- **S3: query recap.** Let the web_search sub-agent template reference the user's
  original question (a new placeholder), as MATPO found helpful. Requires widening the
  protected placeholder set, so it is a separate experiment.

## Left to the builder

- Job-chaining mechanics and the exact layout under `experiments/results/rsi/`.
- Pool sizes, round count, K, minibatch sizes, reflector input budget, per-call caps,
  n-gram length, retry counts, tie handling (defaults above; all recorded).
- Whether the 32B reflector in arm B runs on a second GPU or a separate job.
- Module names and where the reflector loop lives.
- Test coverage. At minimum:
  - seed bundle renders today's system prompt, planning turn, memory turns and sub-agent
    prompts byte for byte;
  - a P\* trajectory folds into rows identical to a P_seed trajectory with the same
    actions, for orchestrator and sub-agent rows;
  - validation rejects: missing placeholder, malformed example tool call, a candidate
    containing a training question n-gram, a candidate containing a shown gold answer;
  - reflector input stays within budget on a long trajectory.
