# A Two-Month Fine-Tuning Approach: Evidence-Gated Next-Action SFT

## Summary

This document proposes **one feasible fine-tuning approach** that can be introduced, implemented, evaluated, and written up within a two-month thesis timeline.

The approach is:

> Fine-tune the orchestrator with supervised fine-tuning (SFT) on an **evidence-gated next-action policy**.

The goal is not to solve every failure mode identified in `docs/failure_mode.md`. Instead, the goal is to make one focused intervention that addresses the most important orchestration weakness: the orchestrator often stops too early, answers without enough evidence, or trusts a single weak tool result.

This is a suitable thesis-scale intervention because it:

- targets the largest observed failure mode directly;
- can be built from existing traces without needing a full RL pipeline;
- requires a manageable amount of human repair;
- gives clear before/after evaluation metrics;
- produces a concrete fine-tuning contribution for the thesis without claiming to solve all MAS failures.

---

## Why This One Approach

The full failure analysis identified several possible fine-tuning directions:

| Failure mode | MAS failures | Share of MAS failures |
|---|---:|---:|
| Direct reasoning without action | 998 | 39.4% |
| Retrieval/evidence acquisition failure | 437 | 17.2% |
| Single-shot tool trust | 361 | 14.2% |
| Tool loop or empty final answer | 351 | 13.9% |
| Modality/tool-coverage mismatch | 341 | 13.5% |
| Computational sub-goal or modeling error | 44 | 1.7% |
| Answer format/unit normalization error | 2 | 0.1% |

It is not realistic to address all of these within two months. RL-based recovery policies, tool-affordance training, document-following retrieval, modality handling, and computational sub-goal repair are each substantial projects.

The best single intervention is therefore to train the orchestrator's **next-action decision**:

- Should it answer now?
- Should it call a sub-agent?
- If it has already called a sub-agent, is the result sufficient?
- If not sufficient, what should it do next?

This target cuts across the three largest actionable failure modes:

| Targeted behavior | Related failure mode | Evidence |
|---|---|---|
| Do not answer directly when evidence or computation is needed | Direct reasoning without action | 998 failures; 322 matched direct-baseline successes; 188 same-question MAS positives |
| Do not stop after weak retrieval evidence | Retrieval/evidence acquisition failure | 437 failures; 84 matched direct-baseline successes; 56 same-question MAS positives |
| Do not trust a single tool result without checking sufficiency | Single-shot tool trust | 361 failures; 87 matched direct-baseline successes; 60 same-question MAS positives |

The intervention intentionally does **not** try to solve modality/tool-coverage failures. Many of those failures come from image or video requirements while image inspection was disabled. Fine-tuning can teach graceful refusal later, but it cannot create missing visual tools.

---

## Core Research Question

The thesis experiment can be framed as:

> Can supervised fine-tuning on evidence-gated next-action decisions improve the MAS orchestrator's ability to decide when to call tools, when to continue gathering evidence, and when to stop?

This question is narrow enough to answer empirically and broad enough to be meaningful for the MAS architecture.

---

## Method: Evidence-Gated Next-Action SFT

The fine-tuning dataset consists of training examples where the input is the orchestrator state and the target is the next correct decision.

### Input

Each training example contains:

```text
Original question
Available tools
Turn-0 query analysis, if present
Compact action history so far
Latest tool/sub-agent result, if any
```

### Output

The model is trained to produce one of three decision types:

```text
CALL_TOOL
FINAL_ANSWER
CONTINUE_OR_REPAIR
```

The output should include:

```text
decision: CALL_TOOL | FINAL_ANSWER | CONTINUE_OR_REPAIR
reason: short justification
sub_goal: required only for CALL_TOOL or CONTINUE_OR_REPAIR
tool_name: required only when a tool should be called
final_answer: required only for FINAL_ANSWER
```

This keeps the fine-tuning target close to the current AgentFlow design: the orchestrator already emits sub-goals, tool calls, and final answers. The SFT intervention changes the policy for choosing those actions.

---

## What the Model Should Learn

The approach teaches one policy:

> Stop only when the answer is supported by sufficient evidence; otherwise call or repair a sub-agent action.

This policy decomposes into five operational rules.

| Rule | Desired behavior |
|---|---|
| Evidence requirement detection | If the question asks for a source-specific fact, external lookup, file content, exact count, or computation, do not answer directly. |
| First-action selection | Choose the first tool/sub-agent that can obtain the missing evidence or computation. |
| Evidence sufficiency check | After a tool result, check whether it directly answers the original question. |
| Verification before stopping | If the tool result is partial, generic, empty, contradictory, or off-format, continue rather than finalize. |
| Final answer gating | Emit a final answer only after the evidence is sufficient and the requested format is satisfied. |

These rules deliberately avoid broader claims such as "improve reasoning ability" or "solve multi-agent coordination." The intervention is specifically about **orchestrator control decisions**.

---

## Training Data Construction

The dataset can be built from existing `raw_results.json` files plus a small amount of human repair.

### Positive Examples

Use three positive sources:

| Positive source | Count / availability | Use |
|---|---:|---|
| MAS successful trajectories | 758 MAS successes | Positive examples of valid stopping and successful tool use |
| MAS-only successes | 105 cases, 51 unique questions | High-value examples where AgentFlow succeeds and all selected direct baselines fail |
| Same-question MAS positives | Available for several failure modes | Contrastive examples where one MAS variant solves a question another MAS variant fails |

Not all MAS successes should be used equally. Easy direct-answer successes should be downsampled so the model does not learn to stop too often.

### Negative Examples

Use the targeted failure modes:

| Negative source | Count | Use |
|---|---:|---|
| Direct reasoning without action | 998 | Teach that some questions require action before answering |
| Retrieval/evidence acquisition failure | 437 | Teach that weak search evidence is not enough |
| Single-shot tool trust | 361 | Teach verification before stopping |

These 1,796 failures provide enough negative data for a focused SFT experiment.

### Human-Repaired Examples

A small amount of human repair is necessary because failed traces do not automatically contain the correct next action.

Recommended repair budget:

| Repair type | Target count | Purpose |
|---|---:|---|
| Direct reasoning repair | 150 examples | Add the missing first tool/sub-agent action |
| Retrieval repair | 100 examples | Add refined search, document-following, or evidence-sufficiency decision |
| Single-shot trust repair | 100 examples | Add verification or stop decision depending on tool result quality |
| Positive stop examples | 100 examples | Teach when stopping is justified |
| MAS-only success examples | 50 examples | Preserve useful AgentFlow decomposition behavior |

Total manually curated/repaired examples: approximately **500**.

This is realistic for two months if the repairs are short next-action annotations rather than full rewritten trajectories.

---

## Example Training Formats

### Example 1 - Direct Reasoning Without Action

Input:

```text
Question:
Define f(x)=||x|-1/2| and g(x)=||x|-1/4|. Find the number of intersections...

Available tools:
web_search, code_generator

Action history:
None

Previous model behavior:
Final answer: 12

Ground truth:
385
```

Target:

```text
decision: CALL_TOOL
reason: The problem requires exact enumeration or a verified symbolic/numeric computation; a direct answer is not sufficiently supported.
tool_name: code_generator
sub_goal: Compute the number of intersections using the periodic structure of both composed functions, validate over one fundamental domain, and print only the final count.
```

### Example 2 - Weak Retrieval Evidence

Input:

```text
Question:
What was the volume in m^3 of the fish bag calculated in the University of Leicester paper "Can Hiccup Supply Enough Fish to Maintain a Dragon's Diet?"

Latest tool result:
The search result identifies the paper and gives a PDF link, but does not mention the fish bag volume.
```

Target:

```text
decision: CONTINUE_OR_REPAIR
reason: The search result identifies the source but does not contain the requested volume.
tool_name: web_search
sub_goal: Search within or for the full text of the paper and extract the exact fish bag volume in m^3.
```

### Example 3 - Single-Shot Tool Trust

Input:

```text
Question:
Consider a rhombohedral crystal... What is the interplanar distance of the (111) plane?

Latest tool result:
5.330309234703367

Choices:
A. 8.95 Angstrom
B. 10.05 Angstrom
C. 9.54 Angstrom
D. ...
```

Target:

```text
decision: CONTINUE_OR_REPAIR
reason: The computed value does not match any answer choice, so the formula or implementation should be checked before finalizing.
tool_name: code_generator
sub_goal: Re-derive and recompute the rhombohedral d-spacing formula for the (111) plane, then compare the result to the choices.
```

### Example 4 - Valid Final Answer

Input:

```text
Question:
How many studio albums were published by Mercedes Sosa between 2000 and 2009?

Latest tool result:
The relevant Wikipedia discography entries are Corazon Libre (2005), Cantora 1 (2009), and Cantora 2 (2009), yielding three studio albums.
```

Target:

```text
decision: FINAL_ANSWER
reason: The tool result directly identifies the relevant albums and the requested count.
final_answer: \boxed{3}
```

---

## Training Objective

Use ordinary supervised fine-tuning on next-action targets.

The model is trained to map:

```text
(question, tools, query_analysis, action_history, latest_result)
```

to:

```text
(decision, reason, tool_name/sub_goal or final_answer)
```

No RL is required for the first thesis experiment. Preference tuning or RL can be described as future work.

### Why SFT Instead of RL

SFT is the correct first intervention for a two-month timeline because:

- it is simpler to implement and debug;
- it requires fewer infrastructure changes;
- it produces interpretable training examples;
- it can be evaluated directly against known failure modes;
- it avoids the complexity of designing a reward model for multi-turn tool use.

RL would be appropriate later for optimizing turn efficiency, recovery behavior, and long-horizon tool-use policies. For this thesis timeline, it is too broad.

---

## Dataset Split

Use question-level splitting to avoid leakage across variants.

Recommended split:

| Split | Share | Notes |
|---|---:|---|
| Train | 70% | Used for SFT |
| Validation | 15% | Used for checkpoint selection |
| Test | 15% | Held out for thesis results |

Important constraint:

All traces for the same `(benchmark, question_id)` should stay in the same split. This prevents training on one MAS variant of a question and testing on another variant of the same question.

### Suggested Dataset Size

A feasible first dataset:

| Data type | Approximate count |
|---|---:|
| Human-repaired negative next-action examples | 350 |
| Positive stop/tool-use examples | 150 |
| Automatically converted MAS success examples | 500-1,000 |
| Downsampled correct direct-answer examples | 100-200 |
| Total | roughly 1,100-1,700 examples |

This is large enough for a thesis-scale SFT experiment while still small enough to inspect manually.

---

## Evaluation

The evaluation should answer whether the fine-tuned orchestrator improves the targeted behavior, not whether it solves all MAS limitations.

### Primary Metrics

| Metric | Expected direction |
|---|---|
| Overall accuracy on held-out selected benchmarks | Increase or no degradation |
| Direct-reasoning-without-action failure rate | Decrease |
| Single-tool-stop failure rate | Decrease |
| Retrieval evidence failure rate | Decrease |
| Tool-call rate on evidence-required questions | Increase appropriately |
| Tool-call rate on easy direct-answer questions | No large increase |
| Average turns per correct answer | No severe increase |

### Behavioral Metrics

In addition to final accuracy, report:

| Behavioral metric | Why it matters |
|---|---|
| `no_action_wrong` | Measures premature stopping |
| `one_tool_wrong` | Measures single-shot trust |
| `insufficient_evidence_stop` | Measures evidence gating |
| `valid_first_tool_call` | Measures first-action selection |
| `format_correct_final_answer` | Ensures final answer quality is not degraded |

### Baselines

Compare:

| System | Purpose |
|---|---|
| Original MAS orchestrator | Main baseline |
| Direct-tools baseline | Contextual comparison, not the main training target |
| Fine-tuned MAS orchestrator | Proposed intervention |

The main comparison should be original MAS vs. fine-tuned MAS. The direct-tools baseline is useful for interpretation but should not be treated as the architecture to imitate wholesale, because there are 105 MAS-only success cases where MAS decomposition helps.

---

## Two-Month Timeline

### Weeks 1-2: Dataset Extraction and Labeling

Goals:

- Extract selected MAS and baseline traces.
- Assign existing failure-mode labels from `docs/failure_mode.md`.
- Build question-level train/validation/test split.
- Define the next-action JSON/text schema.
- Select examples for manual repair.

Deliverables:

- Normalized trace table.
- Split files.
- Annotation guidelines.
- First 50 repaired examples for sanity checking.

### Weeks 3-4: Human Repair and SFT Dataset Construction

Goals:

- Repair approximately 500 next-action examples.
- Convert MAS success traces into positive examples.
- Downsample easy direct-answer examples.
- Validate that examples follow the output schema.

Deliverables:

- Final SFT training set.
- Validation and test sets.
- Dataset statistics by failure mode and benchmark.

### Week 5: Fine-Tuning Run

Goals:

- Run SFT on the orchestrator model or the available fine-tuning proxy model.
- Track validation loss and behavior-specific validation metrics.
- Save at least two checkpoints for comparison.

Deliverables:

- Fine-tuned checkpoint.
- Training logs.
- Validation report.

### Week 6: Evaluation

Goals:

- Run original MAS and fine-tuned MAS on the held-out evaluation subset.
- Compute accuracy and behavioral metrics.
- Compare failure-mode distributions before and after fine-tuning.

Deliverables:

- Evaluation table.
- Error analysis of improved and worsened cases.
- Cost/turn analysis.

### Week 7: Thesis Analysis

Goals:

- Interpret results.
- Identify which targeted behaviors improved.
- Identify regressions, such as tool overuse or excessive continuation.
- Relate findings back to the MAS architecture.

Deliverables:

- Thesis-ready results section.
- Qualitative examples.
- Limitations section.

### Week 8: Final Write-Up and Cleanup

Goals:

- Finalize the fine-tuning chapter/section.
- Document reproducibility details.
- Prepare final plots/tables.
- Freeze dataset and checkpoint references.

Deliverables:

- Final thesis text.
- Final tables.
- Reproducibility appendix.

---

## What This Approach Does Not Cover

This single approach intentionally leaves several issues unresolved.

| Not covered | Reason |
|---|---|
| Full RL optimization | Too much infrastructure and reward-design complexity for two months |
| Image/video recovery | The tools were disabled or unavailable in the analyzed run |
| Deep computational reasoning repair | Requires richer code traces and problem-specific verification |
| Full retrieval pipeline redesign | Requires changes to search/document tools, not only orchestrator fine-tuning |
| Sub-agent fine-tuning | The thesis scope fine-tunes the orchestrator only |
| All benchmark failures | Many GPQA/HLE failures are domain knowledge or reasoning failures, not pure orchestration failures |

These limitations should be stated explicitly in the thesis. The contribution is not "we solve all failures"; it is "we show that targeted SFT can improve one central orchestrator control policy."

---

## Thesis Framing

A concise thesis framing could be:

> Based on the failure analysis, I selected the largest actionable class of orchestrator failures: premature or insufficiently supported stopping. I fine-tuned the orchestrator using supervised next-action examples that teach it to distinguish when to answer directly, when to call a sub-agent, and when to continue gathering evidence. This intervention is intentionally narrower than the full set of identified failure modes, but it directly targets 1,796 observed failures across direct no-action answers, retrieval evidence failures, and single-shot tool trust.

This framing is honest and defensible:

- It is grounded in measured failure counts.
- It explains why only one approach was chosen.
- It fits a two-month implementation window.
- It produces a clear empirical comparison.
- It preserves the larger failure analysis as motivation for future work.

---

## Expected Contribution

The expected contribution is a focused empirical result:

> A supervised fine-tuning intervention on evidence-gated next-action decisions can improve orchestrator control behavior in a multi-agent system, especially by reducing premature direct answers and unsupported finalization after weak tool evidence.

Even if the final accuracy gain is modest, the experiment can still be valuable if it shows measurable improvement in:

- no-action failure rate;
- single-tool-stop failure rate;
- evidence-sufficiency decisions;
- preservation of MAS-only successes;
- qualitative quality of orchestrator trajectories.

This is a realistic and thesis-appropriate fine-tuning study: narrow enough to finish, but directly connected to the empirical failure analysis and the core design choice of the MAS.
