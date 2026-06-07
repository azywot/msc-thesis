# Fine-Tuning Action Plan for the Orchestrator

## Purpose and Scope

This document translates the empirical failure analysis in `docs/failure_mode.md` into concrete fine-tuning actions for the centralized MAS orchestrator. The target model is the orchestrator only: sub-agents remain stateless execution workers, and the orchestrator remains responsible for query analysis, sub-agent selection, sub-goal formulation, result interpretation, stopping, and final answer synthesis.

The plan is grounded in the selected analysis scope:

| Source | Scope |
|---|---|
| MAS runs | Earliest completed Qwen3-8B `subagent_tools_*` runs across AIME, GAIA, GPQA, HLE, and MuSiQue |
| Baseline runs | Earliest completed direct-tools baseline runs across Qwen3-8B/Qwen3-32B and `none`/`orchestrator` thinking modes |
| Unit of analysis | Question-run case, not unique benchmark question |
| MAS failures analyzed | 2,534 failures out of 3,292 MAS question-run cases |
| MAS-only positives | 105 successful MAS question-run cases, covering 51 unique benchmark-question pairs where all selected direct-tool baselines failed |

The central conclusion is that fine-tuning should not try to teach the orchestrator "more knowledge" in a generic way. The dominant failure modes are policy failures in when to act, how to formulate sub-goals, how to verify tool outputs, how to recover from failed calls, and when to stop.

---

## Executive Action Points

| Priority | Action point | Training method | Evidence from analysis | Expected effect |
|---:|---|---|---|---|
| 1 | Train an action-vs-answer policy before final answer emission | SFT first, then RL preference/reward tuning | Direct reasoning without action: 998 MAS failures, 39.4% of all MAS failures | Reduce premature direct answers, especially on GPQA/HLE and complex AIME tasks |
| 2 | Train evidence-seeking and retrieval verification behavior | SFT with contrastive traces; RL reward for evidence sufficiency | Retrieval/evidence failure: 437 failures; 84 matched direct-baseline successes | Improve web/text tasks by requiring source-specific evidence before stopping |
| 3 | Train verify-before-stop behavior after one tool call | SFT contrast pairs; RL penalty for unsupported single-call finalization | Single-shot tool trust: 361 failures; 87 matched direct-baseline successes | Reduce acceptance of partial, irrelevant, or numerically implausible tool results |
| 4 | Train recovery from empty, repeated, or failed tool calls | RL reward shaping plus SFT recovery examples | Tool loop or empty final answer: 351 failures; 49 same-question MAS positives | Reduce max-turn loops, repeated commands, and empty predictions |
| 5 | Train tool-affordance awareness for unsupported modalities | SFT refusal/rerouting examples; not primarily accuracy RL | Modality/tool mismatch: 341 failures, but image/video tools were intentionally disabled | Prevent futile image/video/text-tool loops and improve graceful failure |
| 6 | Train precise computational sub-goal formulation | SFT from repaired code-task traces; targeted RL checks | Computational sub-goal error: 44 failures; 19 matched direct-baseline successes | Improve code-agent delegation quality on math/data tasks |
| 7 | Train final answer normalization | Synthetic SFT and deterministic reward checks | Answer format/unit error: 2 failures, both clean | Cheaply eliminate preventable formatting/unit misses |

---

## Fine-Tuning Targets by Orchestrator Responsibility

### 1. Initial Query Analysis

The orchestrator must learn to classify the task before deciding whether it can answer directly.

Action points:

- Add a query-analysis label for required evidence type: `internal_reasoning`, `web_evidence`, `file_evidence`, `calculation`, `multi_hop_retrieval`, `unsupported_modality`, or mixed.
- Add a query-analysis label for risk level: `safe_to_answer_directly`, `requires_tool_verification`, `requires_multi_step_plan`, or `unsupported_with_current_tools`.
- Penalize query analyses that identify a tool need but then stop without calling a tool.
- Reward query analyses that state a concrete verification criterion, e.g. "final answer must be a title from the paper text", "code output must print a numeric value", or "answer must be in thousands of hours".

Training data:

| Dataset source | Use |
|---|---|
| 998 direct-reasoning failures | Negative examples for premature finalization |
| 188 same-question MAS positives for direct-reasoning failures | Positive examples for when a related MAS trace successfully handles a similar item |
| 322 matched direct-baseline successes | Contrastive evidence that the question was solvable by a tool-enabled agent |
| 105 MAS-only successes | Positive examples showing when sub-agent decomposition is useful |

Recommended SFT target:

```text
Given the original query and available tools, produce a query analysis that:
1. identifies the evidence/computation required,
2. decides whether direct answering is justified,
3. names the first sub-goal if a tool is needed,
4. states what observation would be sufficient to stop.
```

Do not train the model to always call tools. The positive set must include easy direct-answer cases that were correct, otherwise the orchestrator may overuse tools and lose efficiency.

### 2. Sub-Agent Selection

The orchestrator should learn tool affordances and tool limitations.

Action points:

- Teach `web_search` as a discovery tool, not as a final evidence oracle when the requested answer is inside a linked PDF, table, changelog, paper, or page history.
- Teach `text_inspector` as appropriate for attached text/CSV/PDF-like files, but not for images or video.
- Teach `code_generator` as appropriate for numeric calculation, exact enumeration, parsing structured data, and verification checks.
- Teach explicit unsupported-state handling when the task requires a disabled image/video capability.

Training data:

| Failure mode | Count | Fine-tuning use |
|---|---:|---|
| Modality/tool-coverage mismatch | 341 | Tool-affordance/refusal/rerouting data |
| Retrieval/evidence acquisition failure | 437 | Web vs. document-following selection data |
| Computational sub-goal error | 44 | Code-agent delegation data |

Important caveat:

The modality/tool-coverage failures should not be used to claim that fine-tuning alone will recover accuracy. The experiment disabled image inspection, and some tasks genuinely require an unavailable modality. Fine-tuning can still prevent wasted turns by teaching the orchestrator to say that the task is unsupported under the current tool set or to seek a supported alternative if one exists.

### 3. Sub-Goal Formulation

The most useful fine-tuning target is not merely "call the right tool"; it is "call the right tool with a sub-goal that preserves the original task constraints."

Action points:

- Require each sub-goal to include the exact object being sought, not a vague topic.
- Require code sub-goals to specify inputs, constraints, expected output, and output format.
- Require retrieval sub-goals to distinguish "find the source" from "extract the answer from the source".
- Require multi-hop tasks to carry forward resolved entities explicitly.
- Penalize sub-goals that broaden the question into a generic search and lose the requested relation.

Examples of target transformations:

| Bad sub-goal pattern | Better sub-goal pattern |
|---|---|
| "Search for the paper." | "Find the PDF of the named paper and extract the fish bag volume in m^3 from the calculation section." |
| "Simulate the functions and count intersections." | "Use the periodicity of both composed functions to enumerate all intersections over one fundamental domain, validate numerically at high resolution, and print only the final count." |
| "Query arXiv for hep-lat articles." | "Count January 2020 `hep-lat` articles that have PostScript versions available; exclude records without PS links and print the count." |
| "Analyze the image with text inspector." | "The required evidence is visual and no image inspector is enabled; report unsupported rather than calling text tools." |

Training data:

| Source | Use |
|---|---|
| 44 computational sub-goal failures | Hard negatives for underspecified or wrong computational delegation |
| 19 matched direct-baseline successes for computational failures | Candidate positives for repaired sub-goals |
| Same-question MAS positives | Positive examples for keeping constraints through the action history |

### 4. Result Interpretation

The orchestrator must learn to treat sub-agent outputs as evidence requiring interpretation, not as authoritative final answers.

Action points:

- Add an intermediate "evidence sufficiency check" before every final answer.
- Teach the orchestrator to detect when a tool result is empty, generic, irrelevant, contradictory, or does not contain the requested answer.
- Teach the orchestrator to compare numeric outputs against expected ranges and answer formats.
- Teach the orchestrator to re-query or change tools when a search result says "no helpful information found" or returns only a summary.

Training data:

| Failure mode | Count | Result-interpretation signal |
|---|---:|---|
| Retrieval/evidence acquisition failure | 437 | Search result did not contain the requested evidence |
| Single-shot tool trust | 361 | One tool result was accepted too early |
| Tool loop or empty final answer | 351 | Empty output was not treated as failed evidence |
| Computational sub-goal error | 44 | Computed output was inconsistent, implausible, or answering the wrong question |

Recommended SFT target:

```text
Given the original query, query analysis, and latest tool result, decide:
1. Does the result directly answer the requested question?
2. Is the result complete enough to stop?
3. If not, what is the next corrective action?
```

### 5. Stopping Decision

Stopping is a major training target because several failure modes are premature stops or failure to stop correctly after loops.

Action points:

- Train a binary `STOP` vs. `CONTINUE` decision after each action-history update.
- Add a reason for every stop: `answer_verified`, `unsupported_tool_requirement`, `budget_exhausted_with_best_effort`, or `cannot_resolve_after_failed_recovery`.
- Penalize final answers after exactly one tool call unless the tool result directly satisfies the query and format.
- Penalize repeated identical commands after two failed attempts.
- Reward strategy changes after failed tool outputs.

Stopping policy:

| Situation | Desired policy |
|---|---|
| No tool called and task requires external evidence | Continue with a tool call |
| One search result gives a generic summary but not the requested fact | Continue with refined search or document inspection |
| Code result is empty | Continue by repairing the code/sub-goal |
| Same command failed twice | Change tool, change query, or stop as unsupported if no alternative exists |
| Requested modality is unavailable | Stop with unsupported-tool explanation, not an empty answer |
| Final answer unit/format differs from prompt | Normalize before stopping |

---

## Recommended Training Program

## Stage 0 - Data Construction and Labeling

Before model updates, build a training table from the selected MAS and baseline traces.

Required fields:

| Field | Description |
|---|---|
| `benchmark` | AIME, GAIA, GPQA, HLE, or MuSiQue |
| `question_id` | Original question ID |
| `run_path` | Source `raw_results.json` path |
| `failure_mode` | One of the failure labels in `docs/failure_mode.md` |
| `orchestrator_state` | Original query, query analysis, compact action history |
| `bad_action` | Premature answer, wrong tool, weak sub-goal, repeated call, or bad final synthesis |
| `preferred_action` | Corrected query analysis, tool call, recovery action, or stop decision |
| `training_type` | `sft_positive`, `sft_negative`, `preference_pair`, `rl_reward_event`, or `synthetic` |
| `evidence_source` | MAS success, baseline success, human repair, or synthetic repair |

Data construction action points:

- Extract all 2,534 MAS failures as negative traces with labels.
- Extract all 758 MAS successes as positive traces, but downsample easy direct-answer successes so the model does not overfit to stopping.
- Extract the 105 MAS-only successes as high-value positives for sub-agent decomposition.
- For each failure with at least one matched baseline success, create a contrastive pair: MAS failed trajectory vs. direct baseline successful trajectory.
- For each same-question MAS positive, create a within-system contrastive pair: failed MAS trajectory vs. successful MAS trajectory.
- Human-repair at least one representative trace per failure mode and benchmark before using the data for SFT.

## Stage 1 - SFT on Orchestrator Decision Traces

SFT should teach the desired behavior explicitly before RL is introduced.

Primary SFT tasks:

| SFT task | Input | Target output |
|---|---|---|
| Query-analysis repair | Original query and available tools | Structured query analysis with evidence type, first sub-goal, and stop criterion |
| Next-action prediction | Query, analysis, action history | Correct next tool call, recovery action, or final answer |
| Evidence sufficiency judgment | Query and latest tool result | Continue/stop decision with reason |
| Final-answer normalization | Query, computed result, draft answer | Correct final answer format |
| Unsupported-tool handling | Query and available tools | Explicit unsupported/reroute decision |

Suggested SFT mixture:

| Component | Suggested share | Rationale |
|---|---:|---|
| Action-vs-answer calibration | 30% | Largest observed failure mode |
| Retrieval and evidence discipline | 20% | High count and user-facing impact |
| Verification after tool use | 15% | Prevents single-shot trust |
| Loop recovery | 15% | Teaches robust control flow |
| Tool affordance awareness | 10% | Prevents impossible modality loops |
| Computational sub-goal precision | 7% | Lower count but high-value errors |
| Final answer normalization | 3% | Small count, easy synthetic expansion |

These shares are starting points. They should be adjusted after a held-out evaluation checks whether tool overuse, refusal overuse, or unnecessary verification increases.

## Stage 2 - Preference Tuning

Use preference pairs where the same query has a failed and a successful or repaired trajectory.

Preference pair types:

| Pair type | Preferred | Rejected |
|---|---|---|
| Action calibration | Calls a necessary tool or sub-agent | Direct final answer without evidence |
| Retrieval repair | Refines search or follows document | Accepts incomplete search summary |
| Verification | Cross-checks single tool result | Stops after one weak result |
| Loop recovery | Changes strategy after empty output | Repeats same empty command |
| Tool affordance | Declares unsupported modality or reroutes | Calls unavailable/irrelevant modality tool |
| Sub-goal precision | Delegates fully constrained computation | Delegates vague or wrong computation |
| Final synthesis | Normalizes answer unit/format | Emits raw intermediate value |

Preference labels should be applied at the trajectory level and, where possible, at the local decision level. Local labels are more useful for orchestrator fine-tuning because the same trace may contain both good and bad decisions.

## Stage 3 - RL / Reward Modeling

RL should optimize behavior that cannot be captured well by static SFT alone, especially stopping, verification, and recovery.

Reward events:

| Event | Reward |
|---|---:|
| Correct final answer | Strong positive |
| Correct final answer with fewer unnecessary turns | Additional small positive |
| Correctly calls a needed tool before answering | Positive |
| Correctly follows an evidence source after generic search | Positive |
| Detects empty output and changes strategy | Positive |
| Stops as unsupported when required modality is unavailable | Small positive if unsupported is true |
| Final answer after no action when evidence was needed | Negative |
| Final answer after one weak/irrelevant tool result | Negative |
| Repeats same failed command more than twice | Negative |
| Emits empty prediction | Strong negative |
| Uses unavailable modality tool repeatedly | Strong negative |
| Correct reasoning but wrong final unit/format | Negative, but lower than wrong reasoning |

Reward model features:

- Whether final answer is correct under benchmark evaluator.
- Number of tool calls and whether repeated commands occur.
- Whether tool result contains non-empty evidence.
- Whether final answer can be traced to the latest sufficient evidence.
- Whether required modality is available.
- Whether answer format matches the prompt.

RL caveat:

Do not optimize only for fewer turns. Several observed failures come from stopping too early. Efficiency should be a secondary reward after correctness, evidence sufficiency, and valid stopping.

---

## Failure-Mode-Specific Fine-Tuning Actions

### Action 1 - Reduce Direct Reasoning Without Action

Observed signal: 998 MAS failures, 39.4% of all MAS failures. There are 188 same-question MAS positive cases and 322 matched direct-baseline successes.

Behavioral target:

The orchestrator should not emit a final answer immediately when the query requires external evidence, exact computation, file inspection, or multi-hop entity resolution.

SFT examples to construct:

- Negative: original no-action failure trace.
- Positive: repaired query analysis plus first tool/sub-agent call.
- Positive: matched MAS success when available.
- Preference pair: successful tool-backed trajectory preferred over no-action final answer.

Reward actions:

- Penalize direct final answers for questions whose query analysis states a tool is required.
- Reward first-turn sub-goal creation when the query contains attached files, multi-hop references, exact counts, dates, calculations, or source-specific facts.

Evaluation:

- Measure reduction in no-action failures on held-out AIME, GPQA, HLE, and MuSiQue cases.
- Track tool-call rate to ensure the model does not call tools for trivial direct-answer tasks.

### Action 2 - Improve Retrieval and Evidence Acquisition

Observed signal: 437 MAS failures, 17.2% of all MAS failures. There are 84 matched direct-baseline successes and 56 same-question MAS positives.

Behavioral target:

The orchestrator should treat search as an evidence-gathering process. It should continue when a search result identifies a source but does not contain the requested answer.

SFT examples to construct:

- Search-result insufficiency judgments.
- Refined query generation after irrelevant results.
- Document-following examples where a PDF, changelog, paper, page history, or table must be inspected.
- Multi-hop examples that preserve resolved entities across turns.

Reward actions:

- Penalize final answers based on search summaries that do not include the requested fact.
- Reward follow-up actions that narrow the search or inspect the identified document.
- Penalize entity drift in multi-hop questions.

Evaluation:

- Use GAIA and MuSiQue retrieval-heavy subsets.
- Track whether answer-supporting evidence appears in the last successful tool result before final answer.

### Action 3 - Add Verification After Single Tool Calls

Observed signal: 361 MAS failures, 14.2% of all MAS failures. There are 87 matched direct-baseline successes and 60 same-question MAS positives.

Behavioral target:

After one tool call, the orchestrator should ask whether the returned result directly answers the question and whether any independent check is needed.

SFT examples to construct:

- One-shot wrong result followed by preferred verification action.
- Numeric result that does not match multiple-choice options followed by formula check.
- Search result with partial evidence followed by source-specific follow-up.

Reward actions:

- Penalize stopping after one tool call if the result is partial, generic, empty, off-format, or inconsistent with the question.
- Reward a second targeted verification call when the first result is uncertain.

Evaluation:

- Track accuracy and average turns together.
- Ensure verification does not create excessive tool use on simple questions where the first result is sufficient.

### Action 4 - Recover From Empty Outputs and Loops

Observed signal: 351 MAS failures, 13.9% of all MAS failures. There are 65 matched direct-baseline successes and 49 same-question MAS positives.

Behavioral target:

The orchestrator should recognize failed actions and change strategy. It should not repeat the same failed command until max turns or emit an empty final answer.

SFT examples to construct:

- Empty code output -> repair code/sub-goal.
- Empty text inspection -> ask a narrower file question or identify unsupported format.
- No helpful search result -> reformulate query or change source.
- Repeated command -> stop repeating and switch plan.

Reward actions:

- Strongly penalize empty final predictions.
- Penalize repeated identical commands after two failed attempts.
- Reward recovery actions that change the query, tool, or decomposition.

Evaluation:

- Count empty predictions.
- Count repeated commands.
- Count max-turn terminations.
- Measure whether recovery improves correctness without increasing unsupported refusals.

### Action 5 - Teach Tool Affordance and Unsupported-Modality Handling

Observed signal: 341 MAS failures, 13.5% of all MAS failures. However, many are caused by intentionally disabled image/video tools.

Behavioral target:

The orchestrator should know what the configured tools can and cannot do. When a required modality is unavailable, it should stop with a clear unsupported-tool conclusion or use a valid alternative if one exists.

SFT examples to construct:

- Image task with no image inspector -> unsupported.
- YouTube/audio task with no video/audio tool -> unsupported.
- Diagram-dependent math task -> either reconstruct from text if enough information exists or mark visual evidence required.
- Attached file task with text/CSV/PDF -> use text inspector correctly.

Reward actions:

- Penalize repeated calls to text tools for image-only evidence.
- Penalize hallucinated extraction from unavailable modalities.
- Reward correct unsupported-tool classification when no valid tool exists.

Evaluation:

- Separate "accuracy recovery" from "graceful failure recovery."
- Report both final-answer accuracy and unsupported-modality detection rate.

### Action 6 - Improve Computational Sub-Goal Formulation

Observed signal: 44 MAS failures, 1.7% of all MAS failures. Although less frequent, these failures are high value because many are recoverable with better sub-goals.

Behavioral target:

The orchestrator should send code tasks that fully preserve the mathematical or data-processing problem, then inspect the output before finalizing.

SFT examples to construct:

- Repaired code sub-goals with all constraints.
- Examples where the first computed result is inconsistent and must be debugged.
- Examples requiring exact arithmetic, exhaustive enumeration, or schema-aware parsing.

Reward actions:

- Reward code calls that print explicit outputs.
- Penalize empty code outputs.
- Penalize final answers from code results that do not match the requested unit, range, or option set.

Evaluation:

- Use AIME exact-computation cases and GAIA file/data calculation cases.
- Track code-call success rate and correctness of the printed final scalar.

### Action 7 - Normalize Final Answer Format and Units

Observed signal: 2 MAS failures, 0.1% of all MAS failures. The count is small, but the correction is straightforward.

Behavioral target:

Before final answer emission, the orchestrator should re-read the requested answer unit and output format.

SFT examples to construct:

- Raw hours vs. thousands of hours.
- Percent vs. fraction.
- Rounded integer vs. exact decimal.
- Multiple-choice letter vs. explanation.
- Comma-separated list with no whitespace.

Reward actions:

- Penalize answers that contain the right magnitude but wrong requested scale.
- Add deterministic format rewards where benchmark prompts specify exact formatting.

Evaluation:

- Use synthetic format perturbation tests.
- Verify exact-match improvement without changing reasoning trajectory.

---

## Data Augmentation Plan

Existing data is enough to start, but not enough for all targets.

| Target | Existing data sufficiency | Needed augmentation |
|---|---|---|
| Direct reasoning without action | High | Add positive direct-answer examples to avoid tool overuse |
| Retrieval/evidence acquisition | Medium | Human-repaired document-following traces; raw source inspection examples |
| Single-shot verification | Medium | Contrastive examples where one tool call is enough vs. not enough |
| Loop recovery | Medium | Synthetic failed-tool traces with repaired next actions |
| Tool affordance | Medium | Synthetic unavailable-tool scenarios across image/video/audio/web/file |
| Computational sub-goals | Low-medium | Repaired code prompts with visible code and stdout |
| Final answer normalization | Low observed count but easy | Synthetic format/unit transformations |

Synthetic data should be used carefully. It is appropriate for format normalization, tool-affordance classification, loop recovery, and simple verification decisions. It is less appropriate for domain-heavy GPQA/HLE reasoning unless reviewed by a human or checked by a reliable evaluator.

---

## Evaluation Protocol

Fine-tuning should be evaluated against both correctness and orchestration behavior.

Primary metrics:

| Metric | Why it matters |
|---|---|
| Overall benchmark accuracy | Main task outcome |
| Accuracy by failure-mode subset | Shows whether targeted behaviors improved |
| No-action failure rate | Measures action calibration |
| Single-tool-stop failure rate | Measures verification behavior |
| Empty prediction rate | Measures loop/stopping robustness |
| Repeated-command rate | Measures recovery policy |
| Unsupported-modality detection rate | Measures affordance awareness |
| Tool-call count per correct answer | Tracks efficiency without rewarding premature stops |
| Final-format error rate | Measures answer synthesis |

Recommended held-out splits:

- Hold out at least 20% of failure examples per failure mode.
- Hold out all examples from some question IDs across variants to avoid memorizing benchmark items.
- Keep a separate "MAS-only success preservation" set from the 105 MAS-only successes.
- Keep a "direct-answer preservation" set of correct no-tool/direct answers to detect tool overuse.

Regression checks:

- Accuracy should not improve by simply refusing more tasks.
- Average turns should not drop because the model stops prematurely.
- Tool use should not increase on easy tasks where direct answering is correct.
- MAS-only successes should not be lost after training.
- Unsupported-modality detection should be reported separately from answer accuracy.

---

## Implementation Checklist

### Data Pipeline

- Export selected MAS and baseline traces into a normalized training table.
- Attach failure-mode labels from `docs/failure_mode.md`.
- Create local decision examples for query analysis, next action, evidence sufficiency, and final answer synthesis.
- Build contrastive pairs from same-question MAS positives and matched direct-baseline successes.
- Add human repairs for representative examples before training.
- Add synthetic data only for controlled behaviors: formatting, unavailable tools, empty outputs, repeated commands.

### SFT

- Train on structured orchestrator decisions, not only final answers.
- Include both positive and negative examples.
- Preserve successful MAS decomposition traces.
- Include direct-answer positives to avoid tool overuse.
- Include explicit stop reasons in target outputs.

### Preference / RL

- Build preference pairs at trajectory and local-decision levels.
- Reward correctness first, then evidence sufficiency, then efficiency.
- Penalize empty predictions and repeated failed commands strongly.
- Penalize premature finalization when evidence is missing.
- Reward graceful unsupported-tool stops only when the needed modality is genuinely unavailable.

### Evaluation

- Evaluate on held-out failure-mode subsets.
- Report behavior metrics alongside accuracy.
- Compare against the original MAS and direct baseline.
- Audit a sample of improved and worsened traces manually.
- Track per-benchmark effects, especially GPQA/HLE where many failures are hard reasoning rather than pure orchestration.

---

## Recommended First Fine-Tuning Iteration

The first iteration should be narrow and measurable.

1. Build a supervised dataset for three behaviors: action-vs-answer calibration, evidence sufficiency checks, and loop recovery.
2. Train an SFT checkpoint on repaired orchestrator decisions.
3. Evaluate on held-out subsets for direct reasoning without action, retrieval failure, single-shot trust, and loop/empty-final failures.
4. Compare against original MAS using accuracy, no-action failure rate, empty prediction rate, repeated-command rate, and average turns.
5. Only after SFT improves control behavior should preference tuning or RL be added.

The first iteration should not prioritize modality-heavy accuracy recovery, because many modality failures are caused by disabled image/video tools. It should still include tool-affordance examples so the orchestrator learns to stop cleanly rather than loop.

## Bottom Line

The orchestrator should be fine-tuned as a controller, not as a general answer generator. The most important learned behaviors are:

- decide when evidence or computation is required;
- formulate sub-goals that preserve the original constraints;
- treat sub-agent outputs as evidence to be verified;
- recover from failed or empty tool calls;
- stop only when the answer is supported and correctly formatted;
- preserve the cases where MAS decomposition outperforms the direct-tool baseline.

If these behaviors improve, the fine-tuned orchestrator should gain accuracy from better coordination while avoiding the main risks exposed by the failure analysis: premature stopping, brittle one-shot tool trust, empty-output loops, and unsupported modality hallucination.
