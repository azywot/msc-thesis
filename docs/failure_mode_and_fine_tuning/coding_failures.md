# Coding Failure Investigation: AIME No-Coder Ablation

## Question

The ablation table shows an apparently surprising result on AIME:

| Condition | AIME accuracy |
|---|---:|
| Full system: web searcher + coder + structured memory | 55.0% |
| No coder sub-agent: web searcher + structured memory | 60.0% |

At first glance, this suggests that removing the coder sub-agent improves AIME. The key question is whether this is evidence of a **coder sub-agent limitation**.

## Short Answer

For AIME, this result is **not evidence that the coder sub-agent is producing bad code or bad outputs**.

The decisive reason is that the full-system AIME run did not invoke the coder at all:

| Run | Source | Correct / total | Accuracy | `code_generator` calls | `web_search` calls | Total action-history entries |
|---|---|---:|---:|---:|---:|---:|
| Full system | `experiments/results/1_milestone_no_img_no_mindmap_AgentFlow/aime/qwen8B_subagent_tools_orchestrator/train_2026-03-15-21-30-27_20752258/raw_results.json` | 33 / 60 | 55.0% | 0 | 0 | 0 |
| No coder | `experiments/results/subagent_orchestrator_ablation/aime/qwen8B_subagent_orch_no_code_generator/train_2026-03-28-20-20-42_21276365/raw_results.json` | 36 / 60 | 60.0% | 0 | 1 | 1 |

Therefore, the 5-point improvement cannot be attributed to avoiding bad code execution. It is better interpreted as a **tool-availability prompt effect** or **reasoning-trajectory variance**: changing the set of available tools changes the system prompt and the model's direct reasoning path, even when no tool is actually called.

## What Changed

The no-coder run gains 3 net questions:

| Outcome comparison | Count | Question IDs |
|---|---:|---|
| No-coder correct, full-system wrong | 9 | 6, 11, 18, 20, 22, 23, 78, 80, 85 |
| Full-system correct, no-coder wrong | 6 | 8, 21, 24, 65, 76, 83 |
| Both correct | 27 | Not listed here |
| Both wrong | 18 | Not listed here |

The improvement is concentrated in AIME 2025:

| Year | Full system | No coder | Difference |
|---|---:|---:|---:|
| 2024 | 19 / 30 | 19 / 30 | 0 |
| 2025 | 14 / 30 | 17 / 30 | +3 |

This pattern is consistent with stochastic or prompt-induced reasoning differences rather than a systematic sub-agent execution failure.

## Why This Is Not a Coder Sub-Agent Limitation

The coder sub-agent can only be blamed if at least one of the following happens:

1. The orchestrator calls `code_generator`.
2. The generated code solves the wrong problem, crashes, silently returns an empty result, or returns an incorrect result.
3. The orchestrator receives that bad output and trusts it.

In this AIME comparison, condition 1 does not occur in the full-system run. The full system has `code_generator` available, but every question has an empty `action_history` and zero `code_generator` calls. The failures are therefore failures of direct mathematical reasoning, final-answer extraction, or termination control.

The surprising result is still important, but it diagnoses the orchestrator and prompt design, not the coder implementation.

## Concrete Examples Where No-Coder Wins

| Question ID | Ground truth | Full-system prediction | No-coder prediction | What happened |
|---:|---:|---:|---:|---|
| 18 | 106 | 6017 | 106 | Both runs solved the logarithmic product directly with no tools. The full system made an algebraic telescoping error: it simplified the rational product to `1984/65`, then multiplied by 3. The no-coder run split the rational product into `(k-1)/(k-2)` and `(k+1)/(k+2)`, obtaining `31 * 1/13`, then multiplied by 3 to get `93/13`, so `m+n=106`. This is pure reasoning variance, not a code failure. |
| 20 | 293 | empty | 293 | The full-system run produced a very long coordinate-geometry derivation and reached a max-length style failure with no final extracted answer. The no-coder run gave a shorter coordinate setup for the two circles and rectangle, formed the area equality, and reached the correct final value. This is a termination/final-answer failure in the full system, not a coder error. |
| 22 | 610 | 600 | 610 | Both runs reasoned directly about the greedy coin algorithm. The full system undercounted the values of `N` for which greedy is optimal. The no-coder run identified the correct count. No code was executed, even though this is exactly the kind of finite enumeration problem where a coder call could have verified the count. |
| 23 | 149 | 145 | 149 | Both runs solved the trigonometric zero-counting problem without tools. The full system miscounted the number of zeros or tangent cases for `sin(7 pi sin(5x))`. The no-coder run counted the endpoint and tangency cases correctly. |
| 80 | 211 | 24 | 211 | Both runs derived the base-`b` digit condition `ab+c=(a+c)^2`. The full system followed an incorrect counting path and returned a much too small base. The no-coder run found the correct least base. Again, there was no coder output to blame. |
| 85 | 080 | 240 | 80 | The full system used a plausible but wrong geometric heuristic, effectively selecting the harmonic mean of `200, 240, 300`. The no-coder run used the correct relation `s = abc/(ab+bc+ca)` and returned 80. |

These examples show that the no-coder run wins mostly because it happens to follow a better direct reasoning trajectory on some items. It is not recovering from bad code.

## Concrete Examples Where Removing Coder Hurts

The no-coder condition is not uniformly better. It also loses 6 questions that the full system answered correctly.

| Question ID | Ground truth | Full-system prediction | No-coder prediction | What happened |
|---:|---:|---:|---:|---|
| 24 | 907 | 907 | 9 | The no-coder run misread "no person sits next to two other people" as "no two selected chairs are adjacent." The full system correctly interpreted the constraint as "no three consecutive occupied chairs" and counted the valid subsets with a generating function. |
| 83 | 045 | 45 | 1000 | The full system correctly reduced the digit-grid constraints to `A+B+C=8`, giving `C(10,2)=45`. The no-coder run overcomplicated the case analysis and produced an invalid final count. |
| 65 | 104 | 104 | 422 | The full system correctly used the disphenoid structure of the tetrahedron and computed the inradius expression. The no-coder run followed a longer derivation and ended with the wrong expression and final sum. |
| 76 | 468 | 468 | 520 | Both runs used geometry formulas involving the circumcenter and incenter. The no-coder run made an algebraic/formula error after deriving `OI^2=13`; the full system kept the derivation on track. |
| 8 | 62 | 62 | empty | The no-coder run failed to produce a final answer after a long rotated-parabola derivation. The full system reached the correct value. |

This matters because it rules out a simple conclusion such as "coder hurts AIME." The no-coder run is better by net 3, but the switched cases go both ways.

## More Likely Explanation

The most plausible explanation is:

> The presence or absence of the coder changes the prompt and reasoning trajectory, even though the model never actually calls the coder on AIME.

Several details support this:

- AIME is primarily a self-contained mathematical reasoning benchmark. It does not require external retrieval or file inspection.
- In the full run, `code_generator` is available and the system prompt includes a code-tool schema plus a worked code-tool example, but the orchestrator still chooses direct reasoning for every question.
- In the no-coder run, the tool list is shorter and the model follows slightly different direct reasoning paths. Some paths are better, some are worse.
- The net gain is small: 36/60 versus 33/60. At the question level, 9 items flip in favor of no-coder and 6 flip in favor of full-system.

This is best understood as a **prompt-conditioned reasoning instability**, not as a demonstrated limitation of the coder sub-agent.

## What This Means for Fine-Tuning

The AIME ablation does not support removing the coder globally. It supports a narrower fine-tuning target:

> Teach the orchestrator when computation should actually be delegated, and require verification when a problem is enumerable, arithmetic-heavy, or algebraically easy to check.

For AIME, the failure is almost the opposite of "bad coder use": the orchestrator had access to code but did not use it even on problems where code could have provided a sanity check. Examples include:

- Question 22: count values of `N` from 1 to 1000 for a greedy coin algorithm.
- Question 23: count zeros and tangencies of a trigonometric function over a finite interval.
- Question 80: search for the least base `b` satisfying a digit equation.
- Question 83: count digit assignments satisfying linear constraints.

These are natural candidates for a fine-tuning rule:

> If the problem can be reduced to a finite enumeration, exact arithmetic check, symbolic simplification check, or recurrence/counting verification, the orchestrator should call `code_generator` with a precise sub-goal before finalizing.

The correct training target is therefore not "avoid coder." It is:

1. Do not let the mere presence of tools perturb direct reasoning unnecessarily.
2. Use the coder only when it can verify a specific computational sub-goal.
3. When coder is available and the task is enumerable, produce a concrete code sub-goal rather than silently solving everything in the query analysis.
4. Do not finalize after a long derivation without checking that a boxed answer was actually produced.

## Conclusion

The no-coder AIME improvement is real in the measured results, but the causal interpretation is subtle. Since the full AIME run made **zero** coder calls, the result does **not** show that the coder sub-agent is weak. It shows that tool availability changes the orchestrator's prompt context and direct-reasoning trajectory.

For the thesis, this should be reported as an orchestration and prompt-sensitivity failure:

> On AIME, removing the coder improved accuracy from 55.0% to 60.0%, but this was not because bad code was avoided. The full system never invoked the coder. The improvement comes from different direct-reasoning trajectories under a different tool set. The fine-tuning implication is to train explicit computational-delegation and verification behavior, not to remove the coder.
