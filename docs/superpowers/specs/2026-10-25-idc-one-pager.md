# IDC: Interventional Decision Credit (one-pager)

Agata Żywot, for Xinyi Chen, 25 October 2026. Target: ICML 2027 (paper deadline around 22 January, to be confirmed).

## Problem

Our Qwen3-8B orchestrator calls frozen Qwen3-1.7B sub-agents (search, code). Most failures are the orchestrator's decisions: when to use a tool and when to stop. GRPO fine-tuning did not fix them, and we think there are two reasons:

1. **Good tool calls get punished.** If the sub-agent fails, the whole rollout gets zero reward, including the orchestrator's good call. Over time the orchestrator learns to avoid tools.
2. **No signal when every rollout fails.** About 25% of question groups are all-wrong, so GRPO learns nothing from them. Doubling the group size barely helps.

## Idea

Run two cheap checks on failed rollouts and use the results to correct the credit:

- **Was the tool call good?** For search: the gold answer is in the pages it retrieved (free). For code: rerun the same task with a 32B coder and see if the episode then succeeds. If the call was good, don't punish it.
- **Would another choice have worked?** For all-wrong groups: replay the rollout to a decision, force each other option (search, code, answer) and let the model finish. If one succeeds, push the model toward it.

Everything else is plain GRPO. The checks only produce labels; their text is never trained on. Extra cost: about 15 to 25% more compute, which we also give every baseline.

## Why it is new

Recent work already tries alternative actions to assign credit (ASCT, FGPO, TWICE), but in single agents. Nobody separates "the planner chose badly" from "the planner chose well and the sub-agent failed".

**Results we aim for:** IDC beats compute-matched GRPO, and gains more than GRPO when stronger sub-agents are swapped in at test time.

## Week 1: go / no-go tests (no training needed)

| Test | Pass if |
|---|---|
| 0. Did tool use drop in our past GRPO runs, more where sub-agents fail? | Yes, in at least one domain |
| 1. What share of failed tool calls were actually good? | At least 10% |
| 2. In all-wrong groups, does forcing another option recover the answer? | At least 15% of groups, and more often than resampling |

All pass: full method paper. Some pass: a smaller paper (a diagnosis, or half of the method). Tests 0 and 2 fail: stop and choose a new direction.

## Timeline

| Weeks | Work |
|---|---|
| 1 | Go / no-go tests |
| 2 to 3 | Implementation |
| 4 | First IDC vs GRPO runs |
| 5 to 8 | Main results |
| 9 to 10 | Ablations |
| 11 to 13 | Writing and submission |

## To decide together

1. **Split of work.** Proposal: you run RL training and three baselines; I build the checks, evaluation and writing.
2. **Strong coder** for the code check: Qwen3-32B, or the cheaper Qwen3-30B-A3B if its labels agree.
