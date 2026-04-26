# Failure Mode and Fine-Tuning Documentation

This folder contains the thesis-facing analysis that connects empirical Multi-Agent System (MAS) failures to concrete orchestrator fine-tuning decisions.

The central message across these documents is:

> The main weakness of the MAS is not the existence of sub-agents themselves. It is the orchestrator's control policy: when to answer directly, when to call a sub-agent, how to formulate sub-goals, how to interpret tool results, and when to stop.

The documents move from empirical diagnosis to broad fine-tuning design, then to a thesis-feasible two-month implementation plan.

## Document Map

### 1. [`failure_mode.md`](failure_mode.md)

Use this as the empirical foundation.

This document is the full failure mode analysis of the MAS against the direct-tools baseline. It defines the data schema, lists the selected result runs, quantifies MAS performance by benchmark, and assigns failures to bottom-up categories.

Key contents:

- selected MAS and baseline run inventory;
- benchmark-level MAS outcomes;
- seven empirical failure modes with exact counts and source question IDs;
- representative examples for each failure mode;
- MAS vs. baseline overlap analysis;
- fine-tuning implications grounded in observed failures.

Main quantitative result:

- 3,292 MAS question-run cases were analyzed.
- 758 were correct and 2,534 failed.
- The largest failure mode was **Direct reasoning without action**: 998 failures, 39.4% of all MAS failures.

The main failure modes are:

- **Direct reasoning without action**: the orchestrator answers without using tools when evidence or computation is needed.
- **Retrieval/evidence acquisition failure**: the system searches or inspects but does not obtain the needed evidence.
- **Single-shot tool trust**: the orchestrator stops after one weak tool result.
- **Tool loop or empty final answer**: the system loops, hits the turn budget, or produces no usable answer.
- **Modality/tool-coverage mismatch**: the task needs image/video/diagram evidence unavailable in the run.
- **Computational sub-goal or modeling error**: the orchestrator delegates an incorrect or incomplete computation.
- **Answer format/unit normalization error**: the reasoning is nearly correct but the final answer format is wrong.

Read this first if you need the evidence base for the thesis chapter.

### 2. [`fine_tuning_plan.md`](fine_tuning_plan.md)

Use this as the comprehensive design space.

This document translates the failure analysis into a broad fine-tuning action plan for the orchestrator. It is intentionally wider than what can be implemented in two months. It is useful for showing that the project considered the full failure landscape before narrowing the thesis intervention.

Key contents:

- executive action points ranked by failure frequency and expected impact;
- fine-tuning targets by orchestrator responsibility:
  - initial query analysis,
  - sub-agent selection,
  - sub-goal formulation,
  - result interpretation,
  - stopping decision;
- recommended training stages:
  - data construction,
  - supervised fine-tuning,
  - preference tuning,
  - RL or reward modeling;
- failure-mode-specific training actions;
- data augmentation plan;
- evaluation protocol;
- implementation checklist.

Main takeaway:

> Fine-tuning should not try to teach the model generic knowledge. It should train the orchestrator's tool-use and stopping policy.

Use this document when discussing the full set of possible interventions and future work.

### 3. [`fine_tuning_plan_narrowed_down.md`](fine_tuning_plan_narrowed_down.md)

Use this as the implementable thesis plan.

This document narrows the broad plan to one feasible two-month fine-tuning approach:

> **Evidence-Gated Next-Action SFT**

The idea is to fine-tune the orchestrator to choose the next correct action from the current state:

- `CALL_TOOL`
- `FINAL_ANSWER`
- `CONTINUE_OR_REPAIR`

This approach targets the largest actionable failure modes without requiring a full RL pipeline.

Primary target behaviors:

- do not answer directly when evidence or computation is needed;
- do not stop after weak retrieval evidence;
- do not trust a single tool result without checking sufficiency;
- call `code_generator` when a problem needs exact enumeration, arithmetic verification, or symbolic/numeric checking;
- emit the final answer only when evidence is sufficient and the required format is satisfied.

Why this is thesis-feasible:

- it uses existing `raw_results.json` traces;
- it requires a manageable amount of human repair;
- it can be trained with supervised fine-tuning only;
- it gives clear behavioral metrics before and after fine-tuning;
- it is narrow enough to implement and write up in two months.

Use this document as the main practical fine-tuning proposal.

### 4. [`coding_failures.md`](coding_failures.md)

Use this as the focused investigation of the AIME coder ablation.

This document addresses the surprising ablation result that AIME performs better when the coder sub-agent is removed:

- full system: 55.0% on AIME;
- no-coder system: 60.0% on AIME.

Main conclusion:

> This is not evidence that the coder sub-agent is bad. In the full AIME run, `code_generator` was called zero times.

The no-coder improvement is therefore better explained as a prompt/tool-availability effect: changing the available tool list changes the orchestrator's direct reasoning trajectory, even when no tool is used.

The document includes:

- exact full-system vs. no-coder run statistics;
- question IDs where no-coder wins;
- question IDs where full-system wins;
- concrete AIME examples;
- fine-tuning implications for computational delegation.

Use this document when explaining why the narrowed SFT plan should include coder-related control behavior without expanding into coder sub-agent fine-tuning.

## Recommended Reading Order

For a thesis reader:

1. Read [`failure_mode.md`](failure_mode.md) for the empirical diagnosis.
2. Read [`fine_tuning_plan.md`](fine_tuning_plan.md) to see the complete intervention design space.
3. Read [`fine_tuning_plan_narrowed_down.md`](fine_tuning_plan_narrowed_down.md) for the selected two-month thesis approach.
4. Read [`coding_failures.md`](coding_failures.md) for the AIME coder-ablation interpretation.

For implementation:

1. Start with [`fine_tuning_plan_narrowed_down.md`](fine_tuning_plan_narrowed_down.md).
2. Use [`failure_mode.md`](failure_mode.md) to select training examples and held-out evaluation slices.
3. Use [`coding_failures.md`](coding_failures.md) to add computational-delegation examples.
4. Use [`fine_tuning_plan.md`](fine_tuning_plan.md) only for extensions beyond the two-month scope.

## Thesis Narrative

The documents support the following thesis narrative:

1. The MAS was evaluated across AIME, GAIA, GPQA, HLE, and MuSiQue.
2. The failure analysis shows that most errors are not isolated tool failures but orchestrator control failures.
3. The most frequent failure is premature direct answering without tool use.
4. Retrieval and single-tool stopping failures show that even when tools are used, the orchestrator often lacks an evidence-sufficiency policy.
5. Some failures are outside fine-tuning scope, especially unavailable image/video modalities.
6. A broad fine-tuning plan identifies many possible interventions, but a two-month thesis needs one focused approach.
7. Evidence-Gated Next-Action SFT is the selected approach because it targets the largest actionable failure modes with a feasible supervised dataset.
8. The AIME no-coder result reinforces this conclusion: the problem is not that the coder sub-agent is inherently bad, but that the orchestrator needs better computational-delegation and verification behavior.

## Key Fine-Tuning Target

The selected fine-tuning target is:

> Given the original question, available tools, query analysis, action history, and latest tool result, train the orchestrator to decide whether to call a tool, continue or repair a tool action, or produce the final answer.

This directly targets the behavior that caused the largest share of observed failures.

## What Is In Scope

The narrowed thesis intervention covers:

- supervised fine-tuning of the orchestrator only;
- next-action decision training;
- evidence sufficiency checks;
- first-tool selection;
- recovery from weak or incomplete tool results;
- computational verification with `code_generator` when appropriate;
- final-answer gating.

## What Is Out of Scope

The narrowed thesis intervention does not claim to solve:

- full RL optimization;
- image or video recovery when tools are disabled;
- deep sub-agent fine-tuning;
- all mathematical reasoning errors;
- all retrieval pipeline limitations;
- every benchmark failure mode.

These limitations are important to state explicitly. They make the contribution more defensible: the thesis does not claim to solve MAS tool use in general, but instead tests one focused, measurable improvement to orchestrator control.

## How the Documents Fit Together

The relationship between the files is:

```text
failure_mode.md
  -> identifies and quantifies failures
  -> motivates fine-tuning targets

fine_tuning_plan.md
  -> maps failures to a broad set of possible training actions
  -> defines the full intervention space

fine_tuning_plan_narrowed_down.md
  -> selects one feasible two-month approach
  -> defines the SFT task, data, metrics, and timeline

coding_failures.md
  -> explains the AIME no-coder ablation
  -> clarifies that coder behavior should be treated as orchestrator delegation, not sub-agent fine-tuning
```

## Bottom Line

This documentation package supports a clear and defensible thesis claim:

> Empirical failure analysis shows that the dominant MAS failures are orchestrator control failures. A practical two-month fine-tuning intervention should therefore train the orchestrator's evidence-gated next-action policy: when to answer, when to call a sub-agent, when to verify, and when to continue rather than stop.
