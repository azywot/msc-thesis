# Interventional Decision Credit for Small-LM Orchestrators

<!-- **Research plan for an ICML 2027 submission.**
Author: Agata Żywot. Collaborator: Xinyi Chen. Written 2026-10-25.
Status: **proposed direction, gated by week-1 tests** (Section 8).

This is the complete plan for the post-thesis project. It is self-contained: every number, design choice and source needed to understand, review or execute it is in this document. It supersedes the earlier version of the plan (called CEPO, Section 13) and incorporates an external review of that version, after every literature claim and number in the review was checked. -->

**Contents**

| § | What it answers |
|---|---|
| 0 | Summary on one page |
| 1 | Background: the system and what the thesis found |
| 2 | Evidence we already have |
| 3 | What the literature already covers, and what is left for us |
| 4 | The method (4.0 is a one-page overview) |
| 5 | Theory |
| 6 | Goals, non-goals, anti-goals, contributions |
| 7 | Experiments |
| 8 | **Week 1: kill tests and the decision they drive** |
| 9 | **Action plan, week by week** |
| 10 | **Risks and reviewer objections** |
| 11 | Relation to the three papers we were given (MASS, tool-use survey, SpecAgent) |
| 12 | Decisions: made, open, and left to the implementer |
| 13 | What changed from the earlier version, and why |
| A | Glossary |
| B | Sources, with verification status |
| C | Code and data pointers |

---

## 0. Summary on one page

**The problem.** Our multi-agent system is a trainable Qwen3-8B orchestrator that coordinates frozen Qwen3-1.7B sub-agents (a web searcher, a coder, a text inspector). It fails mostly because of the orchestrator's decisions: when to call a tool, when to verify, when to stop. Outcome-based reinforcement learning (GRPO: reward 1 for a correct final answer, 0 otherwise) did not fix these decisions in the thesis. GRPO's credit is broken in two ways in this setting:

1. **Wrong credit in mixed groups.** When a rollout fails because the frozen sub-agent (the *executor*) failed, every turn of that rollout gets negative credit, including a dispatch that was perfectly good. Outcome RL cannot tell "bad decision" from "good decision, weak executor". Over training this can teach the orchestrator to avoid tools.
2. **No credit in all-wrong groups.** When every rollout for a question fails, GRPO's advantage is exactly zero and nothing is learned. About a quarter of our groups are like this, and larger groups barely help.

**The idea.** Run two kinds of cheap counterfactual check on failed rollouts and use the outcomes as *credit labels*. No counterfactual text ever enters the training loss.

- **Dispatch adequacy (fixes problem 1).** For a failed rollout that dispatched a sub-agent, ask: would a competent executor have succeeded with *exactly this dispatch*? For search this needs no model at all: the dispatch was adequate if the gold answer appears in the pages its query retrieved. For code, re-run the identical task with a strong coder (Qwen3-32B) and continue the episode. An adequate dispatch is not punished for the executor's failure (the *gate*).
- **Control forcing (fixes problem 2).** In all-wrong groups, go back to the first and last decision, force each other choice (search, code, answer), and let the policy finish. If another choice succeeds, pull the orchestrator's choice probabilities toward it (the *control target*).

Everything else (query text, code tasks, answers) is trained by ordinary GRPO on the policy's own rollouts.

**The claim to test.** With frozen weak executors, outcome RL teaches the orchestrator to avoid tools, and the avoidance spreads beyond the cases where the executor actually fails. Interventional credit prevents this, gives signal where GRPO has none, and yields an orchestrator that **benefits when its executors are upgraded** at test time, which a GRPO-trained orchestrator should not.

**Status.** Three cheap tests in week 1 decide whether this becomes a method paper, a diagnostic paper, a control-credit paper, or is stopped (Section 8). Implementation starts only after that decision.

**Working names.** Method: **IDC (Interventional Decision Credit)**. Paper: *"Don't Blame the Planner: Interventional Credit Assignment for Orchestrators with Frozen Executors"*. Rename freely.

---

## 1. Background

### 1.1 The system (CoSMAS, built in the thesis)

| Component | Detail |
|---|---|
| Orchestrator | Qwen3-8B, trainable. Writes a plan, dispatches sub-agents, reads their results through structured working memory, decides when to answer. At most 15 turns. |
| Sub-agents (executors) | Web Searcher, Coder, Text Inspector. Qwen3-1.7B in the fine-tuning experiments. Frozen, decoding greedily (temperature 0). |
| What a sub-agent sees | **Only the dispatch, never the original question.** The searcher receives the orchestrator's query plus the pages that query retrieved, and returns an extract. The coder receives the task text, writes code, and returns the execution output. |
| Control options per turn | `SEARCH` (a `web_search` tool call), `CODE` (a `code_generator` tool call), `ANSWER` (a final answer, no tool call). |
| Trajectory length | Median 2 orchestrator decisions per question. |
| Training stack | verl with Flow-GRPO-style credit (every turn of a rollout gets the rollout's final reward), binary correctness reward, and a prefix-replay mechanism that can re-run any recorded rollout up to a given turn without repeating tool calls. |

### 1.2 What the thesis found

- The system is **planner-limited, not executor-limited.** Over 86% of failures are orchestrator act, verify or stop decisions. Scaling the sub-agents from 1.7B to 32B barely moves GAIA (21.8% to 23.3%).
- The largest single failure mode is **premature direct answering** (answering from memory instead of acting): 1,032 failures. Of those, 318 (30.8%) were solved by at least one baseline system: 140 by a baseline that called a tool, 178 by reasoning alone.
- Two rounds of GRPO fine-tuning left the direct-answer failures essentially unchanged (246 to 242 with thinking off, 260 to 263 with thinking on), although pass@8 sat far above pass@1 (DeepMath 0.80 vs 0.54). A learnable signal existed and went unused.

### 1.3 Why a new method and not more of the same

Plain GRPO, larger groups, Prefix-RFT (RL that starts from a teacher's partial trajectory) and LoRA fine-tuning were all tried. None of them reached the failing decision. The paper must change *what credit the failing decision receives*, not how much compute is spent on it.

---

## 2. Evidence we already have

All numbers were re-checked against their original source (thesis, fine-tuning diagnosis, run logs) on 2026-10-25.

| Finding | Number | Source | What it implies |
|---|---|---|---|
| Failures are orchestrator decisions | >86% | thesis failure analysis | The decision-maker is the bottleneck |
| Executor scaling barely helps | GAIA 21.8% → 23.3% (1.7B → 32B sub-agents) | thesis | Same; also warns that the share of executor-caused failures may be small (risk R3) |
| GRPO does not move direct answering | 246 → 242 (thinking off), 260 → 263 (thinking on) | thesis fine-tuning diagnosis | Outcome RL does not reach the failing decision |
| GRPO reduced tool use on AIME | 1.32 → 1.05 tool calls per question | thesis fine-tuning diagnosis | A first hint of learned tool avoidance; test 0 checks it properly |
| Unused signal | pass@8 0.80 vs pass@1 0.54 (DeepMath) | thesis fine-tuning diagnosis | Correct behaviour is reachable by the policy |
| Recoverable direct-answer failures | 318 of 1,032; 140 via a tool, 178 via reasoning | thesis failure analysis | Many failures flip under a different control choice |
| Dead groups, n = 8 → n = 16 | all-wrong 27.1% → 25.1%; all-correct 21.0% → 16.9%; mixed 51.9% → 58.0% | Flow-GRPO runs `ft_310076` (n = 8) and `ft_335104` (n = 16), Aug to Sep 2026 | Doubling the group size (about 1.7x the rollouts) buys +6 pp of usable groups; all-wrong groups barely move |
| All-wrong by domain, n = 8 → 16 | DeepMath 19.6 → 17.5%, HotpotQA 35.0 → 32.0%, NQ 35.9 → 37.0% | same runs | Search has the largest all-wrong share |
| Tool-use rate over all rollouts (n = 8) | DeepMath 24.4%, HotpotQA 94.6%, NQ 87.1% | same runs | In search the orchestrator almost always uses a tool; the leverage there is the stop decision and dispatch quality |
| All-wrong rate of the tool-using vs no-tool subgroup (n = 8) | DeepMath 38.2% vs 26.8%; HotpotQA 34.9% vs 60.0%; NQ 39.3% vs 70.6% | same runs | Math: rollouts that call the coder fail more often (consistent with learned tool avoidance, but confounded by difficulty, since hard questions trigger the coder). Search: the opposite. Test 0 separates these |
| Best warm start | SFT on Qwen3-30B-A3B-Instruct traces (no thinking): +6.70 pp macro-average. SFT on Qwen3-32B thinking traces: −4.01 pp | SFT and GRPO results table | Thinking-off is the right regime (decision D1) |
| Other fine-tuning results | GRPO on an 1,800-question set: +6.32 pp. Prefix-RFT: +1.84 pp | same table | Prefix-RFT is under the significance bar |
| Significance bar | macro-average standard error ±1.7 pp, so about +3.5 pp is the 2σ bar | same table | Small benchmarks (AIME, GAIA) can only serve as transfer evidence |

Two claims from earlier drafts were **removed** because they could not be supported: a "25.2% of failures solved by the no-MAS baseline" figure (no traceable source) and the causal claim that search all-wrong groups exist "because of" the frozen searcher (the cause is undetermined; test 1 measures it).

---

## 3. What the literature already covers, and what is left for us

### 3.1 Already covered

The earlier version of this plan (CEPO: try every control option at a decision point and train the choice with an "all-action" policy gradient) is no longer novel. Neither is "add a step-level reward". Checked on 2026-10-25:

| Work | What it does | What it closes for us |
|---|---|---|
| **ASCT** (arXiv 2609.35215, Sep 2026) | At states the agent visits, evaluates **every** legal action from the same prefix, centres the action values by the agent's own probabilities, trains with PPO. Stop/Continue is an action. Qwen3-4B with LoRA, HotpotQA retrieval workflow, discrete action menu, query and answer text frozen | Enumerating actions with actor-centred values. Its key side result: on-policy continuations did *worse* than uniform random continuations (test utility 0.5975 vs 0.6146) at 2.5x the extra tokens |
| **FGPO** (arXiv 2609.10221, Sep 2026), "Why Sample What You Can Enumerate?" | Exact all-action objective over small enumerable tool sets. Questions with no learning signal rise from 0.2% under a uniform policy to 20.8% after GRPO as the policy concentrates | The "dead groups need enumeration" argument. Single-step, precomputed rewards |
| **TPO** (arXiv 2604.06159), **LPO** (arXiv 2605.06139) | Train toward an explicit target distribution built from old-policy probabilities and scores | The target update we use for control choices. Credited, not claimed |
| **TWICE** (arXiv 2610.31544, 22 Oct 2026, concurrent) | Counterfactual "twin" turns that share the original episode's sampling noise, giving turn-level advantages. Search-R1 protocol, Qwen2.5-3B/7B, +5.3 / +4.5 exact match over GRPO; beats Tree-GRPO, VinePPO, GiGPO and IGPO | Counterfactual turn credit on the standard search benchmarks. Single agent, alternatives sampled, no executor |
| **DAS** (arXiv 2602.03304) | Forces the opposite Search/Answer action through an instruction prompt, builds preference pairs where the outcome flips, offline DPO. +0.7 to +2.1 exact match | The idea of forcing a decision. Offline, single agent |
| **When Should LLMs Search?** (arXiv 2607.05752) | Compares no-search and forced-search runs per question to label NO_SEARCH / SEARCH / UNSOLVED; SFT and DPO on the labels | Forced-option labels at the first decision. Offline, single agent |
| **HALT** (arXiv 2608.02009) | Forces a search agent to continue past its own stop; finds under-search on 51 / 64 / 79% of questions (HotpotQA / 2Wiki / MuSiQue) for a 7B agent; exact match roughly doubles on that subset | Premature stopping is real and recoverable. Inference-time only |
| **AXPO** (arXiv 2605.28774), **TAPO** (arXiv 2606.05784) | Tool-call credit in single agents. AXPO resamples the tool call in all-wrong tool-using subgroups | "GRPO under-trains tool calls" in single agents |
| **When2Tool** (arXiv 2605.09252); hidden-state probes (arXiv 2610.22046) | Tool need is decodable from hidden states; an 8B value probe falls below a trace-length baseline | Learned value probes as a cheap substitute for rollouts |
| **APPO** (arXiv 2606.12384) | Chooses where to branch by uncertainty | Uncertainty-based targeting. We target by outcome instead |
| **StepSearch** (arXiv 2505.15107), **HiPRAG** (arXiv 2510.07794), **Search-R2** (arXiv 2602.03647) | Step-level process rewards for search agents: information gain and redundancy, optimal search vs non-search steps, evidence density | "Reward the trajectory, not just the answer" as a contribution. Single agent, search only |

### 3.2 Open: credit across the planner/executor boundary

None of the work above intervenes on the **executor** to separate "the planner chose badly" from "the planner chose well and the frozen executor failed". The closest work:

| Work | What it does | Our difference |
|---|---|---|
| **CCPO** (arXiv 2603.21563) | Counterfactual credit for co-trained agents by *removing* an agent (two-agent math pipelines) | Removal, not a capacity intervention; co-trained, not frozen. Removing our executor is the same as forcing `ANSWER`, which is only one of our checks |
| **MPO** (arXiv 2503.02682) | A meta-planner trained with DPO against a frozen executor | Plan-level, offline, no attribution |
| **MATPO** (arXiv 2510.04678) | Planner and worker roles in one model, trained jointly with RL | Avoids the confound by training the executor too; we keep the executor frozen, which is the common deployment case |
| AXPO, TAPO | Tool-call credit | Single agent; the tool is part of the environment, not a model whose capacity we can vary |

### 3.3 What we claim

1. **Finding.** Outcome RL with frozen weak executors teaches the orchestrator *learned tool avoidance*: good dispatches are penalised whenever the executor fails, and the avoidance spreads to cases where the executor would succeed.
2. **Method.** Dispatch-adequacy labels gate the negative credit on adequate dispatches; control-forcing labels give a target update where GRPO has no signal. Counterfactual rollouts produce labels only.
3. **A new evaluation axis: executor-upgrade transfer.** An orchestrator trained with IDC should gain more than a GRPO-trained one when its executors are replaced by stronger ones at test time, without retraining.
4. **Diagnosis.** A 2x2 taxonomy of failed decisions (planner-recoverable × dispatch-adequate), observable during training.

We credit, and do not claim: enumeration with actor-centred values (ASCT, FGPO), the target update (TPO, LPO), forcing a decision (DAS), counterfactual agent credit (CCPO), counterfactual turn credit on search (TWICE), step-level search rewards (StepSearch, HiPRAG, Search-R2).

---

## 4. The method

### 4.0 Overview

**In one paragraph.** IDC is GRPO plus two targeted checks on failed rollouts. In **mixed groups** (some rollouts right, some wrong), GRPO gives the failed rollouts negative credit on every turn. For each failed rollout that dispatched a sub-agent, we check whether that dispatch was adequate; if it was, its turn is not punished. In **all-wrong groups**, GRPO gives no credit at all. There we pick one or two rollouts, go back to their first and last decision, force each other control option from the rollout's own replayed prefix, and let the policy finish. If another option succeeds, the orchestrator's choice probabilities at that decision are pulled toward it. Content (queries, code tasks, answers) is trained by ordinary GRPO on the policy's own rollouts. Counterfactual rollouts produce labels, never training text.

**The loop.**

```
GRPO step: batch of questions, n rollouts each, policy pi_theta, weak (1.7B) executors
  |
  +-- all-correct groups ----------------------------------> nothing (advantage is zero)
  |
  +-- mixed groups: ordinary GRPO, plus the GATE
  |     for each failed rollout, for each dispatch it made:
  |       search: adequate if the gold answer is in the pages its query retrieved  (free)
  |       code  : re-run the identical task with the strong coder, continue
  |               on-policy; adequate if the episode then succeeds                  (capped)
  |     adequate dispatch turn: advantage max(A, 0) instead of A (A < 0 here)
  |     all later turns keep their ordinary advantage
  |
  +-- all-wrong groups: GRPO gives zero, so add the CONTROL TARGET
        pick 1 to 2 rollouts; at their first and last decision:
          replay the rollout's own prefix (recorded tool results, no new calls)
          for each other option c': prefill the header for c', continue on-policy
                                    -> Q_hat(s, c') in {0, 1}
        if some Q_hat = 1: add KL(pi_star || pi_theta) at that decision,
                           pi_star(c|s) proportional to pi_old(c|s) * exp(Q_hat(s,c) / tau)
        also log dispatch adequacy (for the 2x2 diagnosis; no training effect here)
```

**What each label does.**

| Label (of a decision in a failed rollout) | How it is obtained | Training effect |
|---|---|---|
| Dispatch-adequate, in a mixed group | Adequacy check (4.3) | Gate: that dispatch turn gets max(A, 0) |
| Planner-recoverable, in an all-wrong group | Control forcing (4.3) | Control target toward the option that succeeded |
| Dispatch-adequate, in an all-wrong group | Adequacy check | None (advantage is already zero); logged for the diagnosis |
| Neither | Nothing recovers | None; logged |

**What is new compared with GRPO, in three lines.**
1. Negative credit is withheld from good dispatches that failed only because the executor failed.
2. Signal appears in all-wrong groups, which GRPO ignores.
3. The control update is large enough to move a concentrated policy (a target update, not a plain gradient).

**How this differs from the earlier version (CEPO).** Control forcing is built the same way (replay the rollout's own prefix, prefill a different header, continue on-policy). What changed:

| | CEPO (earlier version) | IDC (this plan) |
|---|---|---|
| Where it branches | Any decision with probability q, reweighted by 1/q, or more often in all-wrong groups | Control forcing only in all-wrong groups at the first and last decision; adequacy checks on failed dispatches in mixed groups |
| Executor intervention | None | Dispatch adequacy: free evidence check for search, strong-coder swap for code |
| Branches per forced decision | 2 (one per other option) | Same |
| What a branch produces | An action value for the gradient, **and its tokens joined the GRPO group** | A label only; **no branch tokens enter the loss** |
| How action values are used | All-action gradient Σ_c ∇π(c \| s)·(Q̂ − V̂) | Target π* ∝ π_old·exp(Q̂/τ), trained with a KL term |
| Extra budget | About +25% rollouts | About +15 to 25%, matched for every baseline |

The 1/q random-branching variant is dropped because its only purpose was to keep CEPO's gradient unbiased; IDC's target update makes no unbiasedness claim. Branching by outcome is a deliberate bias that spends the budget where GRPO's credit is missing or wrong, and its effect is measured by ablations A3 and A7.

### 4.1 Setting and notation

- Orchestrator policy π_θ. Weak executors E_weak (frozen Qwen3-1.7B, greedy, the ones deployed). A strong coder E_strong (Qwen3-32B or Qwen3-30B-A3B-Instruct, decision D11), used only to produce labels: never deployed, never producing tokens that enter the loss.
- A decision point s_k is the start of orchestrator turn k. Its control option is c_k ∈ C = {SEARCH, CODE, ANSWER}; its content x_k is the rest of the turn.
- π(c | s) is read from the log-probabilities of the three header variants at the decision point: the probability of `<tool_call>` followed by the tool name for a tool, and one minus the probability of opening a tool call for `ANSWER`. This takes one short teacher-forced pass, no generation.
- **Thinking-off** in training and evaluation (decision D1). With thinking on, the stop decision is made inside the thinking text, so the header after `</think>` is almost deterministic and forcing it would train the header to contradict the model's own reasoning. Thinking-off also matches the best warm start.
- **Decision-point rule.** If thinking-off turns rarely contain free text before `<tool_call>`, or only a short line, the decision point is the start of the turn and the forced header is prefilled there. If such pre-text is common and long, the decision point moves to the first action token, and the pre-text is trained by GRPO only. Task 1.5 measures which case holds.

### 4.2 Where the checks run

| Group type | Check | Where | Cap |
|---|---|---|---|
| Mixed | Dispatch adequacy (gate) | Every dispatch of every failed rollout | Search: none (free). Code: at most 2 strong-coder swaps per group, chosen at random |
| All-wrong | Control forcing (target) | First and last decision of 1 to 2 random rollouts | 2 forced continuations per decision |
| All-wrong | Dispatch adequacy (diagnosis only) | Same rollouts | Same as above |
| All-correct | None | | |

Target budget: +15 to 25% rollout compute per step (forced continuations are partial episodes, prefixes cost no tool calls). Every baseline gets the same budget.

### 4.3 The checks

All counterfactual rollouts replay the policy's own prefix with the prefix-replay mechanism: recorded tool results are served, so the prefix costs no executor or search calls, and the orchestrator rebuilds every prompt exactly as at inference.

| Check | Procedure | Output |
|---|---|---|
| **Control forcing** | For each option c′ ≠ c_k: prefill the header for c′ (for `ANSWER`, the existing "provide your final answer now" instruction), continue on-policy with E_weak, score with the outcome reward | Q̂(s_k, c′) ∈ {0, 1}. Q̂(s_k, c_k) is the factual reward, 0 in an all-wrong group |
| **Dispatch adequacy, search** | No rollout. The dispatch is adequate if the gold answer (or an alias) appears in the pages its query retrieved. For multi-hop datasets with annotated intermediate answers or supporting facts (HotpotQA, 2WikiMultiHopQA, MuSiQue), those also count, so a correct first-hop query is not mislabelled | Adequate / not adequate |
| **Dispatch adequacy, code** | Keep the task text byte-identical, run it with E_strong instead of E_weak, then continue on-policy with E_weak for any later calls. Adequate if the episode then succeeds | Adequate / not adequate |

**Why this definition of adequacy.** The question the gate needs answered is "would a competent executor have succeeded with exactly this dispatch?" Because sub-agents never see the original question, the only way a strong executor could "cheat" is by adding facts from its own memory. For search, the evidence check avoids that by construction: it acts as an oracle reader over the very pages the orchestrator's query retrieved, at zero cost. For code, the output is execution output, and delegating a whole sub-problem to a capable coder is a legitimate dispatch, so the swap needs no extra safeguard.

### 4.4 The 2x2 diagnosis

For decisions in failed rollouts where both checks ran (all-wrong groups):

| | Dispatch adequate | Dispatch not adequate, or no dispatch |
|---|---|---|
| **Some forced option recovers** | Both | **Planner-recoverable.** Sub-types: stop-limited (forcing a tool at the last decision recovers), dispatch-limited (forcing a tool at the first decision, or switching tool, recovers), over-search (forcing `ANSWER` earlier recovers) |
| **No forced option recovers** | **Executor-limited**: the dispatch was fine, the executor failed it | **Neither**: tool-limited or too hard within budget |

Labels are logged per decision, per step and per domain from the first run.

### 4.5 The loss

Three parts. **Only the policy's own sampled rollouts contribute tokens.**

1. **Content term.** Standard GRPO with Flow-GRPO credit on the policy's own rollouts, with the existing clipping and KL penalty (β = 0.01) and a learning rate below the thesis run's.
2. **Gate (mixed groups).** For a failed rollout's dispatch labelled adequate, the tokens of that dispatch turn (header and dispatch content) get advantage max(A, 0) = 0 instead of A < 0. Later turns keep their ordinary negative advantage, so the orchestrator still learns to handle a bad tool result (retry, verify, rephrase) rather than being told the episode went well.
3. **Control target (all-wrong groups).** At a decision where some forced option recovered, build

   π*(c | s) ∝ π_old(c | s) · exp(Q̂(s, c) / τ),  τ = 0.25 by default,

   and add L_ctrl = KL(π* ‖ π_θ(· | s)) over the three control options. Its gradient with respect to the option logits is π_θ − π*. This is the TPO / LPO target operator applied to the control choice, credited as such.

**Why a target and not the plain all-action gradient.** The all-action gradient on the logit of a rescuing option is π(c′)·(Q − V). At π(c′) = 0.02 and a reward gap of 1, that is about 0.02 per step. The target gives π*(c′) ≈ 0.53 at τ = 0.25, a logit gradient of about 0.51, roughly 25 times larger. A nonzero gradient in dead groups is not enough; it has to be large enough to move a concentrated policy. Ablation A5 tests this.

### 4.6 The reward and the trajectory

The reward stays the verifiable outcome: 1 if the final answer is correct, else 0. The trajectory enters through **credit** and **measurement**, not through hand-designed reward terms.

- **What currently ignores the trajectory is the credit, not the reward.** Flow-GRPO gives every turn of a rollout the same final reward, so a good first search and a premature final answer are treated alike. IDC changes the credit of the specific decisions the checks implicate.
- **Trajectory quality is measured and reported** for every method (Section 7.6): evidence recall, grounded vs ungrounded correct answers, premature stops with retrievable evidence, redundant calls, recovery after a failed tool result.

**Why no hand-designed process reward in the objective.**
1. **It would not be new.** Step rewards for search agents are covered by StepSearch, HiPRAG and Search-R2 (Section 3.1).
2. **The usual shaping terms push the wrong way for us.** Our main failure is acting too little. Cost or redundancy penalties make that worse, and tool-call bonuses invite over-calling.
3. **Proxies get gamed.** IDC's labels come from the true objective under interventions, which keeps the claim clean.

**The obvious alternative is still tested.** One process reward is cheap and verifiable here: the **evidence reward** R_evid = 1 if the gold answer appears in the pages retrieved by any dispatch in the trajectory (the same test as the search adequacy check). It also gives signal in some all-wrong groups. It enters as:
- **Baseline B10:** evidence-shaped GRPO, R = R_outcome + λ·R_evid, λ tuned on validation in {0.1, 0.2, 0.3}.
- **Ablation A11:** IDC plus evidence shaping.
- **Hypothesis H8:** IDC beats evidence-shaped GRPO on search and also helps on math, where no evidence reward exists. If B10 matches IDC on search, we report it.

### 4.7 Deliberately not in the method

- No counterfactual or strong-executor tokens in the loss.
- No learned critic or value probe (closed by When2Tool and the hidden-state probe work; our horizons are short anyway).
- No uncertainty-based targeting (APPO's axis).
- No hand-designed process reward in the objective (Section 4.6).
- No training of the sub-agents in the main result (MATPO's axis; stretch ablation only).

### 4.8 Implementation sketch

| Piece | Where (paths in Appendix C) | Size |
|---|---|---|
| Forced-option hook | Extend the prefix-replay rollout: replay the policy's own prefix to decision k, return a header prefill | Small; reuses the Prefix-RFT replay path |
| Search adequacy check | String and alias match of the gold answer (and annotated intermediate answers) against the recorded retrieved pages | Small, CPU only |
| Code swap hook | In the replay tool registry: run call k with E_strong, serve recorded results elsewhere | Small |
| Control probabilities | Teacher-forced log-probs of the 3 header variants | One function, unit-tested (sums to about 1) |
| Labels and logging | Per decision: option, π(c \| s), Q̂, adequacy, label; also the optimizer step in every rollout record (today step boundaries have to be inferred) | W&B panels |
| Loss | verl driver: gate mask on adequate dispatch turns, control KL term, exclusion of counterfactual rollouts | Driver change plus a toy-group unit test |

---

## 5. Theory

Modest by design: it makes the mechanism precise and turns the main ablations into tests of predictions.

| ID | Statement | Status |
|---|---|---|
| **P1. Frozen-executor confound** | Model a dispatch as succeeding with probability q·e·r: q is dispatch adequacy, e the executor's success rate on adequate dispatches, r the success rate of the rest of the episode. The outcome-RL advantage of dispatching is q·e·r − V(s). For small e it is negative even when q = 1, so GRPO lowers π(dispatch \| s) for adequate dispatches. With shared parameters the decrease spreads to states where e is high. Gating adequate dispatches removes the negative contribution that comes only from e | New and simple; the formal version of the finding |
| **P2. Gate error** | Bound the bias of the gate under label noise. False positives: the gold string appears in the pages by coincidence (very short answers), or the strong coder rescues an inadequate task. False negatives: an adequate first-hop query whose page lacks the final answer and has no annotation. Short and yes/no answers are excluded from the search check; the false-negative direction is conservative (it only withholds the gate) | Needed to defend the labels |
| **P3. Coverage, stated correctly** | Forcing every option and resampling the turn B times have the **same expected control gradient**; they differ in variance and in the chance of a nonzero signal per step: 1 − (1 − π(c′))^B for resampling (7.8% at π = 0.02, B = 4) vs 1 for forcing whenever some option recovers | Corrects the earlier version, which wrongly claimed resampling's *expected* signal vanishes |
| **P4. Content term is unbiased** | Only on-policy rollouts contribute content tokens, so the content gradient is the ordinary GRPO estimator | One line |

The size comparison in Section 4.5 is a remark in the method section, credited to TPO / LPO.

---

## 6. Goals and contributions

### 6.1 Goals

1. **Mechanism:** show that under GRPO, tool use falls where the executor fails, and also spreads to questions where the executor usually succeeds (H1).
2. **Accuracy:** beat budget-matched GRPO, ReOPD, AXPO-style resampling, Prefix-RFT, privileged-executor GRPO and evidence-shaped GRPO with the deployed weak executor, by more than the significance bar on the in-domain suite where the mechanism shows (H3, H5, H8).
3. **Executor-upgrade transfer:** IDC gains more than GRPO when executors are upgraded at test time (H4).
4. **Target update:** the control target beats resampling and the plain all-action gradient at the same points and budget (H6).

### 6.2 Non-goals (for the January deadline)

- New memory mechanisms or sub-agent designs.
- A harness rewrite beyond the hooks in Section 4.8.
- Multimodal tools.
- Training sub-agents as the main result.
- A second model family, τ²-bench and the 4B model size, unless ahead of schedule.

### 6.3 Anti-goals (failure even if the numbers go up)

- Gains that vanish when baselines get the same rollout and GPU-hour budget.
- Gains only on AIME or GAIA, inside the significance bar.
- Any counterfactual or strong-executor token entering the loss.
- Adequacy labels that rest on the strong executor's own knowledge rather than on the dispatch.
- Accuracy rising while tool calls explode without accounting. Calls and GPU-hours are always reported.
- Claiming novelty that belongs to ASCT, FGPO, TPO, LPO or TWICE.

### 6.4 Paper contributions

1. **Finding:** learned tool avoidance under frozen weak executors, with its mechanism (P1) and measurements across training.
2. **Method:** interventional decision credit (adequacy gate in mixed groups, forced-control target in all-wrong groups, labels only).
3. **Evaluation axis:** executor-upgrade transfer, and the 2x2 failure taxonomy observed during training.
4. **Experiments:** multi-agent orchestrator and token-matched single agent, search and math, budget-matched baselines, 3 seeds.

---

## 7. Experiments

### 7.1 Settings

| | Setting A: orchestrator (main) | Setting B: single agent |
|---|---|---|
| Policy | Qwen3-8B orchestrator, thinking-off | Qwen3-8B, thinking-off, calls tools directly |
| Tools | Frozen Qwen3-1.7B sub-agents (greedy) | Raw retriever and Python execution (the system's single-agent baseline mode) |
| Checks | Control forcing and dispatch adequacy | Control forcing only (no executor to gate) |
| Why | Our system and our evidence | Reviewers expect a single agent with the same token budget (arXiv 2604.02460 shows single agents can match multi-agent systems at equal budget). An explicit equal token budget is enforced |

### 7.2 Training setup

- **Warm start:** the SFT checkpoint trained on Qwen3-30B-A3B-Instruct traces (step 1200, +6.70 pp), shared by every RL method.
- **Full fine-tuning** (it beat LoRA in our August runs), KL kept (β = 0.01), learning rate below the thesis run's.
- **Data:** the existing 3k training mix (search: HotpotQA and NQ; math: DeepMath), scaled toward about 30k if time allows.
- **Retrieval:** a local Wikipedia-2018 index with the E5 retriever (the Search-R1 setup) for training and search evaluation. Serper web search stays for GAIA.
- **Validation:** a new ~500-item held-out set (Search-R1 dev sets plus DeepMath). The old 60-item set did not track benchmark scores. We report the final checkpoint, not the best-on-validation one.

### 7.3 Evaluation

| Group | Benchmarks | Role |
|---|---|---|
| In-domain search | NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle (the Search-R1 suite) | Standard and large. Kept for comparability; the claim does not rest on beating TWICE's numbers here |
| In-domain math | DeepMath held-out, AIME 2024 and 2025, AMC, MATH500 | Where the data hints at tool avoidance |
| Transfer | GAIA (text-only), GPQA | Transfer only; differences are inside the ~3.5 pp bar |
| **Executor upgrade** | Every trained orchestrator evaluated with 1.7B, 8B and 32B executors, no retraining | Tests H4; the new axis |
| Stretch | τ²-bench retail and airline | Only if ahead of schedule |

### 7.4 Baselines (same SFT start, same rollout and GPU-hour budget)

| ID | Baseline | Status | Why |
|---|---|---|---|
| B1 | SFT only | have | Lower bound |
| B2 | GRPO with Flow-GRPO credit, group size raised to match the budget | have (needs budget matching) | Primary competitor |
| B3 | Prefix-RFT | have | Teacher-prefix method |
| B4 | **ReOPD** (on-policy distillation with teacher prefix replay, arXiv 2607.04763) | to implement | Strongest published prefix-replay method in the same search setup |
| B5 | AXPO-style resampling of the turn at the same decision points | to implement | Sampled instead of forced branching |
| B6 | DAS-style offline DPO on the same forced pairs | to implement | Online labels vs offline preferences |
| B7 | **Privileged-executor GRPO**: train with E_strong for every call, deploy with E_weak | to implement | The baseline that could kill IDC: if it matches, the gate adds nothing |
| B8 | Token-matched single agent (Setting B) | to implement | Reviewer requirement |
| B9 | TWICE | run in Setting B if its code is released; otherwise cite its numbers with the backbone caveat | Concurrent, same benchmark suite |
| B10 | **Evidence-shaped GRPO** (Section 4.6) | to implement (reuses the search adequacy check) | The process-reward alternative reviewers will ask for |

### 7.5 Ablations

| ID | Ablation | Tests |
|---|---|---|
| A1 | Gate only vs control target only vs both | Which part carries the gain |
| A2 | Strong coder: Qwen3-32B vs Qwen3-30B-A3B-Instruct; search adequacy with vs without intermediate-answer annotations | Label quality vs cost (D11) |
| A3 | Budget +12 / +25 / +50% | Gain per compute |
| A4 | On-policy vs uniform continuations for Q̂ | Answers ASCT's finding that uniform continuations won |
| A5 | Target update (τ sweep) vs plain all-action gradient | Whether gradient size matters |
| A6 | Forcing vs resampling at the same points and budget | P3 |
| A7 | Where to force: first / last / both decisions | Where the failure lives |
| A8 | Scaling: distractor tools so the number of control options goes 3 → 10 → 30 (force the top-k by π plus one random tail option); usable groups and accuracy per GPU-hour for GRPO at n = 8 / 16 / 32, resampling and IDC | "Forcing every option does not scale" |
| A9 (stretch) | Orchestrator and sub-agents share one model, both trained | MASS / MATPO connection |
| A10 (stretch) | A teacher writes the content at the forced decision | SpecAgent connection; puts teacher text in the loss, so ablation only |
| A11 | IDC plus evidence shaping | Whether process reward and interventional credit add up |

### 7.6 Metrics

- **Accuracy:** mean over 3 seeds with confidence intervals; pass@k.
- **Trajectory level:** evidence recall (gold answer in any retrieved page); grounded vs ungrounded correct answers (correct without ever retrieving the evidence counts as ungrounded); premature stops with retrievable evidence (measured on a 300-question subset by forcing one extra search after the native stop); redundant calls (repeated or near-duplicate queries); recovery after a failed tool result (a new dispatch after an empty or wrong extract).
- **Decision level:** tool rate on questions whose failed dispatches are mostly labelled adequate, over training (the core prediction: falls under GRPO, holds under IDC); premature-stop rate; direct-answer-without-action rate; tool calls per question; label mix over training.
- **Training health:** share of dead groups, gradient norm, KL, entropy.
- **Cost:** rollouts, strong-coder calls, GPU-hours. Per-step and per-compute gains reported separately.

### 7.7 Hypotheses

| | Hypothesis | Test |
|---|---|---|
| H1 | Under GRPO, tool rate falls more on questions with high executor failure (a), and also falls on questions where the executor usually succeeds (b, spillover) | Test 0, then training curves |
| H2 | Under IDC, tool rate on questions with adequate-but-failed dispatches holds | Decision metrics |
| H3 | IDC ≥ budget-matched GRPO with the deployed weak executor everywhere, and better by more than the bar where H1 holds | Main table |
| H4 | IDC gains more than GRPO from an executor upgrade at test time | Executor-upgrade evaluation |
| H5 | IDC matches or beats privileged-executor GRPO (B7) with a small fraction of its strong-executor calls | B7 |
| H6 | The control target beats resampling and the plain all-action gradient | A5, A6 |
| H7 | Online labels beat offline DPO on the same forced pairs | B6 |
| H8 | IDC beats evidence-shaped GRPO on search and also helps on math | B10, A11 |

---

## 8. Week 1: kill tests and the decision they drive

All tests run per domain and use the **actual warm-start checkpoint** (the 30B-A3B SFT checkpoint, thinking-off). For tests 1 and 2, generate fresh groups with that checkpoint (about 500 training questions × 8 rollouts, about 4k rollouts, inference only) rather than reusing prefixes from older runs, so the prefixes come from the policy we will train.

| # | Test | Cost | Pass | Fail |
|---|---|---|---|---|
| **0** | **Tool avoidance under GRPO.** For each existing GRPO run, compare tool rate at its starting checkpoint vs its final checkpoint on fixed evaluation questions, split by each question's executor failure rate (measured from the starting checkpoint's rollouts: a dispatch whose result lacks the gold answer for search, or whose code errored or printed a wrong value for math). Also use the training rollouts of the Flow-GRPO runs, binned into training thirds, for questions revisited across epochs. Measure (a) whether the drop is larger where executor failure is high and (b) whether tool rate also drops where executor failure is low | Hours, no GPU | (a) holds in at least one domain (permutation test). Record (b) and which domains | (a) holds in no domain |
| **1** | **Share of adequate-but-failed dispatches.** Over all failed rollouts with a dispatch, in both mixed and all-wrong fresh groups: run the search adequacy check (free) and the code swap with Qwen3-32B | Search: minutes, no GPU. Code: inference | ≥ 10% of failed dispatches are adequate; ≥ 20% is a strong pass | < 10% |
| **2** | **Forcing vs resampling.** On all all-wrong fresh groups (with and without tool use), at the first and last decision: force each other option (2 rollouts) vs resample the whole turn (2 rollouts). Log π(c′ \| s) of any recovering option and the share of decision points where every Q̂ = 0 | Inference | Forcing recovers ≥ 15% of groups and clearly more than resampling; recovering options have low π | Forcing ≤ resampling, or < 10% |
| **cov** | **Coverage.** Share of all-wrong groups where some option at the first or last decision was never sampled across the 8 / 16 rollouts | Minutes, no GPU | Reported; informs P3 | n/a |

### 8.1 Decision after week 1

Every combination of results maps to exactly one outcome:

| Test 0 | Test 1 | Test 2 | Outcome | Venue |
|---|---|---|---|---|
| pass | pass | pass | **Branch A: full IDC method paper** (this document as written) | ICML 2027 |
| pass | pass | fail | **Branch A without the control target**: gate plus diagnosis; A1, A5, A6 dropped | ICML 2027 |
| pass | fail | pass | **Branch B: diagnostic paper** (mechanism, P1, 2x2 taxonomy) with the control target as the fix | Workshop or TMLR; ICML only if the fix clears the bar |
| pass | fail | fail | **Branch B, analysis only** | Workshop or TMLR |
| fail | any | pass | **Branch C: control-credit paper** in the multi-agent setting (forcing + target + diagnosis; ASCT, FGPO, TWICE credited prominently). If test 1 passed, the gate stays as an ablation | ICML if gains are large; otherwise NeurIPS 2027 |
| fail | any | fail | **Stop this line.** One week lost, not a quarter. Choose the next direction with Xinyi | n/a |

### 8.2 What changes in each branch

| Branch | Keep | Drop or change |
|---|---|---|
| A | Everything | (without target: drop A1, A5, A6, the target loss and its tasks) |
| B | Test-0 analysis at scale, 2x2 diagnosis, P1, executor-upgrade evaluation of existing checkpoints, control target if test 2 passed | Gate, code swap hook, B7, H2, H4 as a method claim; schedule shortens to about 8 weeks |
| C | Control forcing, target loss, diagnosis, B5, B6, B9, A4 to A8 | Gate unless test 1 passed (then ablation only), B7; the paper's claim becomes "forced-control credit in planner/executor systems" |

Sections 9 onwards assume Branch A.

---

## 9. Action plan

Thirteen weeks from Monday 26 October 2026. **ICML 2027 is projected at abstract 16 January 2027 and paper 22 January 2027 (anywhere on Earth).** The call for papers is not out yet (ICML 2026 deadlines were 23 and 28 January); confirm in week 1 and shift this schedule if needed. Owners are placeholders until D10 is decided.

### Week 1 (26 Oct to 1 Nov): kill tests and reading

| Task | Detail | Depends on | Owner | Output |
|---|---|---|---|---|
| 1.1 Locate checkpoints | Paths of the warm-start SFT checkpoint, and the start and final checkpoints of each GRPO run | | Agata | Paths added to Appendix C |
| 1.2 Test 0 | As in Section 8. Group rollouts with the group-reconstruction script; step boundaries are not logged, so bin by file order within a run and state the caveat | 1.1 | Agata | Table and plot: tool-rate change vs executor failure, per domain, with spillover |
| 1.3 Coverage count | From the same groups | | Agata | One number per domain |
| 1.4 Fresh groups | ~500 × 8 rollouts with the warm-start checkpoint, thinking-off, local retriever or Serper as available | 1.1 | Agata | Mixed and all-wrong groups for 1.5 and 1.6 |
| 1.5 Test 1 | Search adequacy check on all failed search dispatches; code swap with Qwen3-32B on all failed code dispatches | 1.4 | Agata | Adequate share per domain |
| 1.6 Test 2 and decision-point check | Forcing vs resampling; π(c′ \| s); share of all-zero Q̂. Also measure how often thinking-off turns have text before `<tool_call>`, and how long it is | 1.4 | Agata | Recovery table; decision-point rule fixed |
| 1.7 Reading | TWICE in full (method, budget rule, code release), ASCT ablations, CCPO method, TAPO, APPO, StepSearch, HiPRAG, Search-R2; confirm the ICML 2027 call for papers | | Agata | Half-page note on anything that changes Section 3 |
| 1.8 Weekly meeting | Present 1.2 to 1.7; choose the branch (Section 8.1); settle D10 and D11 | all | Agata, Xinyi | Branch decision |

### Weeks 2 to 3 (2 to 15 Nov): implementation

| Task | Detail | Output |
|---|---|---|
| 2.1 Forced-option hook | Extend the prefix-replay rollout | CPU unit tests for replay plus prefill |
| 2.2 Adequacy checks | Search: gold and intermediate-answer matching against recorded pages, excluding short and yes/no answers. Code: per-call strong-coder swap in the replay tool registry | Unit tests: byte-identical dispatch, swapped executor only at call k |
| 2.3 Control probabilities | Header log-probs at the decision point | Function plus test (sums to about 1) |
| 2.4 Labels and logging | Section 4.4 labels; optimizer step in every rollout record | W&B panels |
| 2.5 Loss | Gate mask, control KL, exclusion of counterfactual rollouts, check sampler with caps | verl driver change plus toy-group test |
| 2.6 Local retriever | Wikipedia-2018 + E5; Search-R1 evaluation in the runner | Base Qwen3-8B numbers in the published range |
| 2.7 Setting B | Single-agent mode with raw tools and an enforced token budget | Smoke run |
| 2.8 Validation set | ~500 held-out items | New validation config |
| 2.9 Executor-upgrade evaluation | Config to evaluate any orchestrator with 1.7B / 8B / 32B executors | One command per checkpoint |
| 2.10 Theory | Draft P1 and P4 | Overleaf section |

### Week 4 (16 to 22 Nov): first runs

- First IDC and budget-matched GRPO runs: Setting A, 8B, 3k mix.
- Manual audit of 100 adequacy labels (50 search, 50 code) from these runs; fix the check if the error rate is above about 10% (risk R2).
- **Exit criterion:** the core prediction shows in training curves (tool rate on questions with adequate-but-failed dispatches falls under GRPO and holds under IDC); the share of dead groups falls under IDC; no instability.

### Weeks 5 to 8 (23 Nov to 20 Dec): main results

- Main table, Settings A and B: B1 to B8, B10 and IDC. Three seeds for GRPO and IDC, one for the rest at first.
- Executor-upgrade evaluation of every checkpoint.
- Theory: P2 and P3.
- **Checkpoint at week 6:** if IDC is not ahead of GRPO in-domain, stop adding baselines and debug.

### Weeks 9 to 10 (21 Dec to 3 Jan): ablations

- A1 to A8 and A11. A1, A2 and A8 are never cut (they answer the most likely reviewer objections).
- Queue long runs before the holidays.
- Stretch items (A9, A10, τ²-bench, 4B) only if ahead of schedule.

### Weeks 11 to 12 (4 to 15 Jan): writing

- Full draft with figures: method overview (check → label → credit), tool-avoidance curves (GRPO vs IDC), 2x2 taxonomy over training, executor-upgrade plot, forcing-vs-resampling and compute curves.
- Internal review with Xinyi and the supervisor; fill missing runs.
- **16 Jan:** abstract registration.

### Final days (16 to 22 Jan): submit

- Polish, appendix (label audit, adequacy-check error analysis, full per-benchmark tables), code release preparation, submission.

### Standing items for the weekly meeting with Xinyi

1. Status against the current phase's exit criterion.
2. New arXiv papers on counterfactual or enumerated agent credit (weekly alert; TWICE appeared three days before this plan was written).
3. Compute queue for the coming week.

### Cut order if behind

τ²-bench → 4B → A9 / A10 → B9 → Setting B seeds beyond one. The minimum paper is Setting A at 8B with B1 to B8, B10, ablations A1 and A2, and the executor-upgrade evaluation.

---

## 10. Risks and reviewer objections

### 10.1 Technical and conceptual

| ID | Risk | Likelihood | Mitigation |
|---|---|---|---|
| R1 | **Tool avoidance is rational under a weak executor.** If the deployed executor really fails, avoiding it is correct, and the gate could hurt accuracy with the weak executor | medium | The gate only removes negative credit from adequate dispatch turns; later turns keep it, so the orchestrator still learns to recover. H3 requires no regression with the weak executor; H4 is where IDC should clearly win. Both are reported |
| R2 | **Adequacy labels are wrong.** Coincidental string matches, or a strong coder rescuing a poor task | medium | Short and yes/no answers excluded; intermediate-answer annotations for multi-hop sets; a 100-label manual audit in week 4; A2; P2 bounds the bias |
| R3 | **Few failed dispatches are adequate.** The thesis shows executor scaling barely moves GAIA | medium | Test 1 measures it in week 1; Branch B or C if it fails |
| R4 | **The mechanism is math-only.** Search rollouts use tools 87 to 95% of the time | medium | Report per domain. In search the control target still addresses stopping; the scope narrows to "tool-heavy domains" if needed |
| R5 | **Q̂ is mostly zero.** With 0/1 rewards most forced options may fail | medium | Measured in test 2; use 2 continuations per option at forced points if needed |
| R6 | **Decision point unclear in thinking-off text** | medium | Measured in task 1.6; rule in Section 4.1 |
| R7 | **Training diffuses** (as in the thesis run: flat reward, gradient norm pinned, KL budget wasted) | medium | Lower learning rate, decision-level logging from day one, week-4 exit criterion |
| R8 | **On-policy continuations are worse than uniform ones** (ASCT's result) | low to medium | Ablation A4; switch if it reproduces |
| R9 | Extra rollouts and strong-coder calls slow down training steps | low | Search checks are free; code swaps capped; forcing only in all-wrong groups; GPU-hours reported |

### 10.2 Reviewer objections and answers

| Objection | Answer |
|---|---|
| "This is ASCT / FGPO plus GRPO" | We do not claim action enumeration; ASCT and FGPO are credited in the method section. Our contribution is the executor intervention, the frozen-executor mechanism and the executor-upgrade result, which they do not address. ASCT also freezes all generated text and uses a discrete menu; we train free-text content |
| "This is CCPO" | CCPO removes co-trained agents. We test the adequacy of a dispatch to a frozen executor; removal corresponds to our forced `ANSWER`, one of three forced options |
| "Just deploy the bigger executor" | Cost and latency; and the thesis shows executor scaling alone barely moves GAIA. The problem is the training signal, not capacity |
| "Just train with the bigger executor" | That is B7. IDC uses no strong model at all for search and a capped number of strong-coder calls for code, only as labels (H5) |
| "Tool avoidance is the right thing to learn" | R1: weak-executor accuracy (no regression required) and executor-upgrade gain are reported separately |
| "The labels leak teacher capability" | No strong-executor tokens enter the loss. Sub-agents never see the question; search labels use only the retrieved pages; code labels come from execution |
| "Single-agent work already shows tool calls are under-trained" (AXPO, TAPO) | The frozen-executor mechanism is specific to planner/executor systems; test 0 and H1 show it, and Setting B shows what remains without an executor |
| "Forcing every option does not scale" | The control layer is a small set of decision types, not individual tools; force the top-k plus one tail option for larger sets; forcing only in all-wrong groups; A8 measures cost against the number of options and compares signal per GPU-hour with larger GRPO groups. Limitations paragraph: least useful when the option set is large, unstructured and the horizon long |
| "Why not just add a process reward?" | Step rewards for search agents exist (StepSearch, HiPRAG, Search-R2). We test the cheapest verifiable one directly (B10, A11). IDC also works on math, where no such reward exists |
| "Gains come from extra compute" | All baselines budget-matched; GPU-hours reported; per-step and per-compute gains separate |
| "Gains are within noise" | Large search suites and DeepMath held-out, 3 seeds, confidence intervals; AIME and GAIA are transfer only |
| "A single agent beats a multi-agent system at equal budget" (arXiv 2604.02460) | Setting B with an enforced equal token budget |
| "TWICE already does counterfactual credit on Search-R1" | Concurrent; single agent, sampled alternatives, no executor. Compared if code is available (B9) |

### 10.3 Project

| Concern | Mitigation |
|---|---|
| Thirteen weeks is tight | Minimum paper defined (Section 9, cut order); the week-1 gate prevents sunk cost |
| Holidays (weeks 9 to 10) | Queue long runs beforehand; do writing-adjacent work |
| RL runs in weeks 4 to 8 are the bottleneck | D10: decide the split with Xinyi in week 1 |
| ICML dates differ from the projection | Confirm the call for papers in week 1 (task 1.7) |
| More concurrent work appears | Weekly alert; the executor-specific claims are the defensible core |

---

## 11. Relation to the three papers we were given

### 11.1 MASS: Recursive Self-Improvement through Multi-Agent Self-Supervision (arXiv 2610.12176)

**What we take**
- **Multi-agent traces make better training data.** MASS reports that a student trained on multi-agent traces beats one trained on single-agent traces with 1.4x more tokens. Our Setting A vs Setting B comparison asks the same question with verifiable rewards and RL.
- **"Scaffold, then internalise."** MASS removes the workflow prompt before SFT so that delegation moves into the weights (spontaneous delegation rises from 1/110 to 87/110). Our forced headers are temporary scaffolding, and the control target moves the behaviour into π(c | s). Our direct-answer-without-action rate is the analogue of their delegation count, and the gate protects delegation from being unlearned when executors fail.
- **Assistant-only masking.** Same rule as ours: no replayed or forced tokens are trained.
- **"When information flows" matters more than "who the agents are".** Our control layer is exactly that: when to call whom, and when to stop.
- **Same model for all roles.** Their orchestrator and sub-agents share weights: our stretch ablation A9, which also answers the open question of fine-tuning the orchestrator and sub-agents together.

**Where we differ.** MASS targets non-verifiable tasks with self-judging and does SFT on searched workflows; we have verifiable rewards and do RL on single decisions. MASS's bottleneck is proposing better workflows; ours is assigning blame between planner and executor.

**Caveats when citing it:** 8 training tasks, 119 trajectories, 3 private test tasks, LLM-judge win rates with ties dropped, and the headline metric is score per output token.

### 11.2 Survey: The Evolution of Tool Use in LLM Agents (arXiv 2603.22862)

- **Formalism.** The survey writes an action as a tool with arguments, or Finish(answer). Our control/content split is that formalism taken literally: control is which tool or Finish, content is the arguments or the answer.
- **Its open problem on "unclear responsibility boundaries"** in multi-agent systems is the problem this paper addresses directly: the adequacy check says whether the planner or the executor was responsible.
- **All credit assignment it cites is temporal and within one policy.** IDC assigns credit *across* the planner/executor boundary.
- **Its cost-aware objective (reward minus a cost term)** motivates reporting tool calls and GPU-hours next to accuracy.
- **DeepAgent / ToolPO** needs per-call correctness labels; our adequacy checks produce per-call labels without annotation.
- **Tool-Star**'s split by {direct answer correct?} × {tool answer correct?} is an observational version of our interventional 2x2 taxonomy.
- **Must-cite from the survey:** MPO (closest planner-vs-frozen-executor training work), Plan-and-Act, α-UMi, HuggingGPT, RouteLLM and FrugalGPT, LLMCompiler, GTPO, PORTool.
- **Benchmarks:** τ²-bench is our out-of-domain stretch; ToolHop and TRAJECT-Bench are fallbacks if reviewers ask for trajectory-level benchmarks. Large tool inventories are the natural limit of forcing every option (A8).

### 11.3 SpecAgent: Trajectory-Level Speculation with Semantic Verification (arXiv 2610.21463)

- **The connection.** SpecAgent lets a weak model draft a trajectory, holds the first state-changing action, and lets a strong model accept a prefix and repair the first rejected step. Our code swap is a training-time analogue: a strong model stands in for one step to *judge* it, and its output is a label, never training text.
- **Teacher repair** at the forced decision is ablation A10 (it puts teacher text into the loss, so ablation only; it links to Prefix-RFT and ReOPD).
- **Read vs write.** SpecAgent only holds state-changing actions; on τ²-bench, forcing must never replay or force an irreversible write.
- **Cost accounting.** SpecAgent reports strong-model calls and wall time next to quality; we report strong-coder calls and GPU-hours for every method.
- **Complementary.** An IDC-trained orchestrator should need fewer repairs as a SpecAgent drafter. A discussion point, not an experiment.

---

## 12. Decisions

### 12.1 Made

| ID | Decision | Reason |
|---|---|---|
| D1 | **Thinking-off** in training and evaluation | Stop decisions are made inside thinking, making the post-thinking header near-deterministic; thinking SFT data scored −4.01 pp vs +6.70 pp for instruct data |
| D2 | Counterfactual rollouts produce **labels only** | Removes bias in the content gradient and any leakage |
| D3 | **Control target** (TPO-style), not the plain all-action gradient | Gradient size at low π (Section 4.5) |
| D4 | **Local retriever** (Wikipedia-2018 + E5) for training and search evaluation | Reproducible, cheap, comparable to published numbers; Serper for GAIA only |
| D5 | **Full fine-tuning**, KL kept, lower learning rate | August runs; thesis run diagnosis |
| D6 | 4B, a second model family and τ²-bench are stretch items | Schedule |
| D7 | Target checks by **outcome** (mixed and all-wrong groups), not by uncertainty | APPO owns the uncertainty axis |
| D8 | **Reward stays the verifiable outcome**; the trajectory enters through credit and metrics; the evidence reward is baseline B10 and ablation A11 | Section 4.6 |
| D9 | **Dispatch adequacy**: free evidence check for search, strong-coder swap for code; the gate acts in mixed groups | The gate only matters where advantages are negative; sub-agents never see the question |

### 12.2 Open (with recommendations)

| ID | Decision | Recommendation |
|---|---|---|
| D10 | Division of work with Xinyi | Xinyi owns the RL runs and baselines B4, B5, B7 in weeks 4 to 8; Agata owns hooks, labels, evaluation and writing. Decide at the week-1 meeting |
| D11 | Strong coder: Qwen3-32B or Qwen3-30B-A3B-Instruct | 32B for the week-1 test (cleaner label); switch to 30B-A3B for training if its labels agree with 32B's and it is cheaper (A2) |
| D12 | Venue if Branch B or C | Branch B: workshop or TMLR. Branch C: ICML if gains clear the bar, otherwise NeurIPS 2027 |

### 12.3 Left to the implementer

- Exact prefill strings for each forced option.
- Answer normalisation and alias matching for the search check, and the minimum answer length.
- verl plumbing for the gate, the control KL and the check sampler.
- Group size, learning rate, step count (start from the August GRPO settings with a lower learning rate).
- τ (start at 0.25, sweep in A5).
- Retriever index details, as long as they match the Search-R1 setup.

---

## 13. What changed from the earlier version, and why

The earlier version (CEPO, "Enumerate the Control, Sample the Content") proposed forcing every control option at chosen decision points, adding the forced rollouts to the GRPO group, and training the control choice with an all-action policy gradient. An external review and our own checks led to these changes:

| Earlier version (CEPO) | This plan | Reason |
|---|---|---|
| Core claim: enumerate control options with an all-action gradient | Credited to ASCT and FGPO; the claim moved to the executor intervention and the frozen-executor mechanism | ASCT and FGPO published it (verified) |
| Forced rollouts joined the GRPO group | Counterfactual rollouts are labels only | Forced rollouts pick options uniformly, not from the policy, so their tokens biased the content gradient |
| Branching at randomly sampled or targeted decisions, forcing only | Forcing in all-wrong groups; adequacy checks on failed dispatches in mixed groups | The executor intervention is the new contribution; the gate only has an effect where advantages are negative |
| "Resampling's expected signal vanishes as π → 0" | P3: same expected gradient, different variance and chance of a nonzero signal | The earlier statement was wrong |
| Plain all-action gradient | Target update | The plain gradient is about 25x too small at low π |
| Leaning toward thinking-on | Thinking-off, decided | The decision is made inside thinking; SFT evidence |
| Search-R1 suite as the main stage | Kept for comparability; math and executor upgrade carry the claim | Search rollouts already use tools 87 to 95% of the time; TWICE owns counterfactual credit there |
| No ReOPD, no strong-executor baseline | ReOPD (B4), privileged-executor GRPO (B7), evidence-shaped GRPO (B10) | B7 and B10 are the obvious killer baselines |
| One kill test on base Qwen3-8B with older prefixes | Three tests on the warm-start checkpoint with fresh groups, one needing no GPU | Cheaper, and measures the policy we will train |
| An external review proposed a "placebo" swap to detect leakage | Dropped; replaced by the evidence check | Sub-agents never see the question, so a generic dispatch retrieves different pages and tests nothing useful |
| "25.2%" figure; "search all-wrong because of the searcher" | Removed | Untraceable or unsupported |

---

## Appendix A. Glossary

| Term | Meaning |
|---|---|
| Orchestrator | The trainable planner model that dispatches sub-agents and decides when to answer |
| Executor, sub-agent | A frozen model that carries out a dispatched task (search, code, text inspection). It sees only the dispatch, not the question |
| Dispatch | An orchestrator turn that calls a sub-agent; its text is the query or task |
| Control option | Which kind of action a turn takes: `SEARCH`, `CODE` or `ANSWER` |
| Content | The free text of a turn: the query, the code task or the answer |
| GRPO | Group Relative Policy Optimization: sample n rollouts per question, give each the advantage (reward − group mean) / group std |
| Flow-GRPO credit | Every turn of a multi-turn rollout gets that rollout's advantage |
| Group | The n rollouts of one question in one training step |
| Mixed group | Some rollouts correct, some wrong: GRPO gives failed rollouts negative advantage |
| Dead group | All rollouts get the same reward, so every advantage is zero |
| All-wrong group | A dead group where every rollout failed |
| Forced rollout | Replays a rollout's own prefix, prefills a different control option, continues on-policy |
| Dispatch adequacy | Whether a competent executor would have succeeded with exactly this dispatch (search: gold answer in the retrieved pages; code: strong-coder swap succeeds) |
| Gate | Withholding negative advantage from an adequate dispatch turn |
| Control target | A target distribution over control options that up-weights options whose forced rollout succeeded |
| Q̂(s, c) | Observed success (0 or 1) of taking option c at decision s and continuing on-policy |
| Executor-upgrade transfer | How much an orchestrator's accuracy rises when its executors are replaced by stronger ones at test time, without retraining |
| SFT warm start | The supervised fine-tuned checkpoint every RL method starts from |
| Prefix-RFT | RL that starts each rollout from a teacher's partial trajectory |
| ReOPD | On-policy distillation in which a teacher gives token-level feedback on replayed prefixes |
| Search-R1 setup | The standard search-agent benchmark protocol: Wikipedia-2018, E5 retriever, seven QA datasets |
| pass@k | Probability that at least one of k samples is correct |
| Macro-average | Unweighted mean of per-benchmark accuracies |
| Significance bar | About +3.5 pp macro-average, two standard errors on our evaluation mix |

## Appendix B. Sources

**Read on 2026-10-25** (abstract or full text):
- ASCT: https://arxiv.org/abs/2609.35215 (full text: root coverage, frozen text generation, sampled-action PPO, continuation result in its Table 7)
- FGPO: https://arxiv.org/abs/2609.10221
- TPO: https://arxiv.org/abs/2604.06159
- LPO: https://arxiv.org/abs/2605.06139
- TWICE: https://arxiv.org/abs/2610.31544 (first half of the full text)
- CCPO: https://arxiv.org/abs/2603.21563
- When Should LLMs Search?: https://arxiv.org/abs/2607.05752
- HALT: https://arxiv.org/abs/2608.02009 (full text: forced continuation, under-search rates)
- DAS: https://arxiv.org/abs/2602.03304
- AXPO: https://arxiv.org/abs/2605.28774
- ReOPD: https://arxiv.org/abs/2607.04763

**Found on 2026-10-25 at abstract level, to read in task 1.7:**
- StepSearch (EMNLP 2025): https://arxiv.org/abs/2505.15107
- HiPRAG (ICLR 2026): https://arxiv.org/abs/2510.07794
- Search-R2: https://arxiv.org/abs/2602.03647

**From earlier scans and the external review, to re-read in task 1.7:**
- TAPO: https://arxiv.org/abs/2606.05784
- APPO: https://arxiv.org/abs/2606.12384
- When2Tool: https://arxiv.org/abs/2605.09252
- Hidden-state probes for localizing reasoning failures: https://arxiv.org/abs/2610.22046
- Single agent vs multi-agent at equal budget: https://arxiv.org/abs/2604.02460
- MATPO: https://arxiv.org/abs/2510.04678
- MPO: https://arxiv.org/abs/2503.02682
- Tree-GRPO: https://arxiv.org/abs/2509.21240
- IGRPO: https://arxiv.org/abs/2607.06223
- Agent-G²: https://arxiv.org/abs/2608.23318
- Expected Policy Gradients: https://arxiv.org/abs/1706.05374
- Survey, credit assignment from reasoning to agentic RL: https://arxiv.org/abs/2604.09459
- Survey, RL for LLM multi-agent systems through orchestration traces: https://arxiv.org/abs/2605.02801

**The three papers we were given:**
- MASS: https://arxiv.org/abs/2610.12176
- The Evolution of Tool Use in LLM Agents: https://arxiv.org/abs/2603.22862
- SpecAgent: https://arxiv.org/abs/2610.21463

**Venue dates:**
- ICML 2027 projection: https://www.opencurious.com/ai-conference-deadlines/icml-2027
- ICML 2026 dates: https://icml.cc/Conferences/current/Dates

## Appendix C. Code and data pointers

Paths are relative to the root of the thesis code repository (CoSMAS). Only needed by whoever implements the plan.

| What | Where |
|---|---|
| Orchestrator and control parsing (`parse_tool_call`, `extract_answer`) | `src/agent_engine/core/orchestrator.py` |
| Sub-agents (what they receive) | `src/agent_engine/tools/web_search.py`, `src/agent_engine/tools/code_generator.py` |
| Training rollout, greedy sub-agents | `src/fine_tuning/rollout.py` |
| Prefix replay (`ReplayController`, `ReplayToolRegistry`, `PrefixOrchestratorRollout`) | `src/fine_tuning/prefix_replay.py`, `src/fine_tuning/prefix_rollout.py` |
| Outcome reward (`OrchestratorReward`) | `src/fine_tuning/reward.py` |
| Group reconstruction and all-wrong statistics | `src/agent_engine/analysis/fine_tuning/rollout_groups.py`, `src/agent_engine/analysis/fine_tuning/all_wrong.py` |
| GRPO rollout records | `experiments/results/fine_tuning/qwen3-8b-grpo-search-math-v2/idx_<domain>_<n>/<k>_rollout_<hash>.json` |
| Flow-GRPO dead-group runs | run IDs `ft_310076` (n = 8) and `ft_335104` (n = 16) |
| Warm-start and GRPO checkpoint paths | to be filled in task 1.1 |
