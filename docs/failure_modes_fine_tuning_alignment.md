# Failure Modes → Fine-Tuning Alignment

**Date:** 2026-05-07
**Related spec:** `docs/superpowers/specs/2026-05-06-orchestrator-finetuning-design.md`
**Related thesis section:** §4 Failure Analysis (SQ3) and §4 Adaptation and Fine-Tuning (SQ4)

---

## TL;DR — Can you proceed?

**Yes.** The GRPO design is well-motivated for the two largest failure modes
(direct reasoning without action, retrieval/evidence failure) and will very
likely produce measurable improvement on GAIA and AIME. The main gap is
GPQA/HLE: the dominant failure there is also "direct reasoning without action,"
but on expert scientific questions where web search does not help — and the
training data does not cover that domain. Treat GPQA/HLE gains as a bonus
rather than a primary target.

Four concrete issues to watch (details below):
1. `data.max_response_length` set to 4096 (doubled from AgentFlow default) to cover multi-turn orchestrator trajectories — if OOM, reduce `gpu_memory_utilization` to 0.55.
2. Binary reward does not penalise direct-reasoning rollouts that happen to be correct — important for any training question Qwen3-8B can answer from memory.
3. **AIME must not be the validation set** — AIME is also an evaluation benchmark; using it for checkpoint selection introduces selection bias. Switch to a held-out DeepMath slice (details in §6.3).
4. Training uses `THINKING_MODE: ORCHESTRATOR_ONLY` to match the evaluation condition — this is correct but adds ~500–1500 tokens per rollout; watch `val/reward_mean` on the DeepMath val split in epoch 1 for signs of truncation (details in §6.5).
5. The base model (Qwen3-8B) already calls tools on retrieval tasks when thinking is enabled; RL may be reinforcing an existing behaviour rather than fixing a broken one.

---

## 1. Failure Mode Taxonomy (recap)

From the full MAS trace analysis across 2,534 failures:

| Failure mode | Total | Share | Primary benchmarks |
|---|---|---|---|
| **Direct reasoning, no action** | 998 | **39.4%** | GPQA 89.6%, HLE 59.9%, AIME 49.0% |
| Retrieval/evidence failure | 437 | 17.2% | MuSiQue 42.1%, GAIA 21.0% |
| Single-shot tool trust | 361 | 14.2% | MuSiQue 31.2%, GAIA 14.2% |
| Tool loop / empty final answer | 351 | 13.9% | GAIA 22.1%, AIME 17.7% |
| Modality / tool-coverage gap | 341 | 13.5% | GAIA 27.3%, HLE 18.0% |
| Computational sub-goal error | 46 | 1.8% | AIME, GAIA |

The first three modes account for ~70% of failures and are the only ones where
the fine-tuning design has a plausible lever. The bottom three require
configuration changes (modality tools) or architectural changes (halt criteria,
sub-goal verification) that RL on the current rollout setup cannot provide.

---

## 2. Primary Target: Direct Reasoning Without Action (39.4%)

### What goes wrong

The orchestrator emits a final prediction after its planning turn without
dispatching any sub-agent. The action history is empty. On GPQA (89.6%) and
HLE (59.9%) this is the nearly universal failure pattern. On AIME (49.0%) the
orchestrator solves problems in its reasoning trace using natural-language
arithmetic rather than calling `code_generator`.

Crucially, 322 of the 998 direct-reasoning failures (32.3%) were solved by at
least one direct-tools baseline run — meaning the correct answer was reachable
with tool use on those questions. This is the recoverable slice that RL should
move.

### How the FT design targets it

The mechanism is indirect but sound: GRPO creates pressure toward tool use by
making tool-less rollouts lose.

**Search-R1 (NQ + HotpotQA):** These questions require specific entity-level
facts that Qwen3-8B does not reliably hold in parametric memory. A rollout that
reasons directly will produce a wrong answer on most of them → reward = 0. A
rollout that calls `web_search` and follows the retrieved evidence will tend to
produce the correct answer → reward = 1. GRPO amplifies the advantage of the
tool-using rollout. The mechanism is self-supervised: you do not need to label
which rollouts used tools.

**DeepMath-103K:** Hard competition mathematics with precise numerical answers.
Direct reasoning introduces arithmetic drift on multi-step problems. A `code_generator`
call returns the exact value. Same advantage dynamic as above.

Across both domains, the policy gradient pushes the orchestrator to dispatch
tools when the question is hard enough that direct reasoning fails consistently
across the 8 rollouts (i.e., when reward variance is high). This is exactly the
failure mode you want to fix.

### Where it does not reach

The GPQA/HLE slice of "direct reasoning without action" is structurally
different. On GPQA the question is designed to be google-proof — web search
returns nothing useful, and the correct answer requires applying a formula or
domain identity. The model should call `code_generator` for numerical
verification, not `web_search`. But Search-R1 teaches "call web_search on
factual questions," not "call code on expert scientific questions." DeepMath
helps here only if the GPQA question has a computational sub-component (some do,
e.g. thermodynamics, crystallography). Pure conceptual GPQA questions (biology,
organic chemistry) are not covered.

**Expected outcome:** GPQA/HLE "direct reasoning without action" will improve
modestly (the computational sub-cases) but not uniformly. Do not cite GPQA as a
primary validation target for this intervention.

---

## 3. Secondary Target: Retrieval/Evidence Failure (17.2%)

### What goes wrong

The orchestrator invokes `web_search` but either stops at the summary without
following the linked document, or accepts the first near-miss result rather than
re-querying. On MuSiQue (42.1%) the chain fails at the second or third hop.

### How the FT design targets it

HotpotQA in Search-R1 is a direct match: it is a multi-hop retrieval task where
both bridge entities must be resolved. A single-hop search rollout will fail
because the second-hop entity is not in the first search result. GRPO rewards
rollouts that chain queries correctly. The signal is not explicit ("you should
re-query") but outcome-driven: only multi-hop search strategies succeed, so only
they collect positive reward.

NQ is single-hop, so it reinforces basic search call behaviour rather than
chaining. The HotpotQA fraction of the training mix is what drives improvement
on retrieval failure.

### Open question

The training mix is 50/50 search vs math by default (10k+10k for fast runs,
50k+50k full). If retrieval failure is a larger concern than math failure in
your evaluation (which it is — 437 retrieval failures vs 46 computational errors),
consider shifting the mix toward 60/40 or 70/30 search-heavy.

---

## 4. Tertiary Target: Single-Shot Tool Trust (14.2%)

### What goes wrong

The orchestrator makes exactly one tool call and stops with the returned answer
without cross-checking. On MuSiQue (31.2%) and GAIA (14.2%) the sub-agent
returns an underdetermined answer that the orchestrator accepts without follow-up.

### How the FT design incidentally addresses it

HotpotQA training examples that require two hops implicitly teach the
orchestrator not to stop after a single search: if the first query returns an
intermediate entity, stopping there yields reward = 0. Only rollouts that issue
a second targeted query on the bridge entity succeed. This trains a basic
verify-before-stop instinct for retrieval tasks.

This is a side-effect of the multi-hop training data, not an explicit design
choice. The training does not cover single-shot trust on code calls (the GPQA
crystallography failure type). That sub-pattern would require training on
questions where the first code result is intentionally misleading or
ambiguous — not present in DeepMath.

---

## 5. Failure Modes the FT Design Does Not Target

| Mode | Why RL cannot fix it | What would fix it |
|---|---|---|
| Tool loop / empty final answer | The orchestrator needs a halt criterion: "if tool returns empty twice, switch strategy." Binary reward cannot teach this — the rollout just fails silently. | Rule-based guard in `orchestrator.py`; or a shaped reward that penalises repeated identical tool calls. |
| Modality / tool-coverage gap | The tool is not in the registry. No amount of RL training can call a tool that doesn't exist. | Enable image inspection for GAIA/HLE runs. |
| Computational sub-goal error | The sub-goal specification is wrong (wrong formula, missing constraint). The code agent executes correctly; the error is in what was asked. Binary reward doesn't distinguish wrong sub-goal from wrong code. | Process-level reward or SFT on correct sub-goal traces. |

---

## 6. Design Decisions: Issues to Watch

### 6.1 `data.max_response_length: 4096`

Set to 4096 (doubled from AgentFlow's default of 2048). The AgentFlow default
applies to single-step Planner rollouts; the msc-thesis orchestrator runs a
full multi-turn conversation, so total response tokens are higher:

- Planning turn: ~300–500 tokens (including thinking trace)
- Tool call 1 → result: ~200–400 tokens
- Tool call 2 → result: ~200–400 tokens
- Synthesis + final answer: ~100–200 tokens

Total per rollout: ~800–1500 tokens typical, up to ~3500 on harder questions
with long search results or code outputs. 4096 covers the tail without
preemptive auditing.

Note: total sequence length is now `18432 (prompt) + 4096 (response) = 22528`.
If OOM occurs on the first batch, drop `gpu_memory_utilization` from `0.6` to
`0.55`, or halve `ppo_micro_batch_size_per_gpu` from `4` to `2`.

### 6.2 Binary reward does not penalise tool-less correct rollouts

If Qwen3-8B already knows the answer to some NQ questions from parametric
memory, a direct-reasoning rollout collects reward = 1 without using any tool.
GRPO will not push the model away from this: the advantage for tool-using
rollouts over direct-reasoning rollouts is zero on questions where both succeed.

This is not catastrophic — it only affects the easy tail of the training
distribution, where direct reasoning already works. The hard questions (where
reward variance is high across rollouts) drive the gradient signal. But it does
mean the model will not learn to call tools on easy retrieval questions, which
is fine — you don't want it to.

**Action:** No change needed. Just be aware that the training is teaching
"call tools when uncertain," not "always call tools."

### 6.3 Validation set: use held-out DeepMath, not AIME

**AIME must not be the validation set.** AIME is one of the five evaluation
benchmarks reported in the thesis. Using AIME 2024 for checkpoint selection
means the best checkpoint is chosen based on the exact questions you later report
results on — selection bias. Even though the model never trains on those
questions, the checkpoint selection procedure optimises for them, which makes
the reported AIME numbers optimistic.

**Fix (already reflected in the spec):** Carve out 200 DeepMath questions before
building `combined_train.parquet` and write them to
`data/training/val/deepmath_val.parquet`. These questions are:
- In-distribution with the math training data — gives a stable per-epoch signal
- Never in the training set (excluded before subsampling)
- Never evaluated on in the final benchmark runs
- Handled by the existing `mode="gen"` branch of `OrchestratorReward` with no
  code change (`data_source="deepmath"` already maps there)

This gives you a clean learning curve: if val accuracy on DeepMath is rising,
`code_generator` use is being learned. You lose the per-epoch AIME learning
curve, but you gain clean final AIME 2024+2025 numbers that a reviewer cannot
question.

**Note on retrieval validation:** The DeepMath val split only confirms math tool
use is improving. It gives no signal on `web_search` learning. If you want a
retrieval learning curve, add a separate 200-question HotpotQA held-out split
(`data_source="hotpotqa"`, `mode="qa"`) — but this is optional; the primary
concern is removing AIME from the validation loop.

### 6.5 Thinking mode: train/eval consistency and token budget

**Why thinking is enabled during training (`THINKING_MODE: ORCHESTRATOR_ONLY`):**
Training must match the evaluation condition. The thesis shows that orchestrator
thinking is the dominant performance driver; the fine-tuned model will be evaluated
with thinking enabled. Training without thinking would create a distribution
mismatch: the model learns to dispatch tools based on input patterns that differ
from what it sees at eval time, because the `<think>...</think>` block precedes
every action token and changes what the model attends to when making the
tool-dispatch decision.

More directly: with thinking enabled, the "direct reasoning without action"
failure is *visible in the training rollout*. The model reasons internally,
reaches a confident-but-wrong answer in the thinking trace, skips the tool call,
and gets reward=0. GRPO can push it to dispatch a tool instead. Without thinking
during training, this exact failure pattern never appears and RL cannot fix it.

**Token budget implication:** Qwen3-8B thinking traces add ~300–800 tokens on
simple NQ/HotpotQA questions and up to ~1500 tokens on harder DeepMath problems.
Combined with tool calls and synthesis, most rollouts still fit within the 4096
token budget. The risk is the hard tail of DeepMath — if truncation occurs,
reward collapses to 0 on those rollouts and the gradient spuriously pushes the
model away from long thinking traces. Watch `val/reward_mean` on the DeepMath
val split in W&B epoch 1. If it stays near zero while training reward rises,
increase `data.max_response_length` to `8192`.

### 6.4 Base model already uses tools on retrieval tasks (with thinking)

Your thesis finding is that Qwen3-8B with orchestrator thinking already reduces
tool calls by half and improves accuracy. This means the base policy already
calls tools selectively on retrieval tasks — RL is refining an existing
capability, not installing a new one. This is good (stable training) but means
the marginal gain from RL may be smaller than if starting from a model that
never calls tools.

**Expected trajectory:** Fast initial gains as the model learns to call tools on
the hardest training questions, then plateauing as the remaining failures are
either easy (model already calls tools) or out-of-distribution (GPQA expert
science).

---

## 7. Summary

| Failure mode | Training data coverage | Reward signal alignment | Expected impact |
|---|---|---|---|
| Direct reasoning, no action (retrieval) | ✅ NQ + HotpotQA directly cover this | ✅ Direct reasoning → reward=0 on these questions | High — GAIA, MuSiQue |
| Direct reasoning, no action (math) | ✅ DeepMath covers this | ✅ Same dynamic | High — AIME |
| Direct reasoning, no action (expert sci.) | ❌ No training data analogue | ⚠️ Web search doesn't help; model may still reason directly | Low — GPQA, HLE |
| Retrieval/evidence failure | ✅ HotpotQA multi-hop | ✅ Single-hop → reward=0 on multi-hop questions | Moderate — MuSiQue, GAIA |
| Single-shot tool trust | ⚠️ Partial (HotpotQA multi-hop) | ⚠️ Side-effect only | Moderate — MuSiQue |
| Tool loop / empty final | ❌ Not addressable by binary reward | ❌ | None |
| Modality / tool-coverage gap | ❌ Configuration issue | ❌ | None |
| Computational sub-goal error | ❌ Reward doesn't distinguish wrong sub-goal | ❌ | None |

**Proceed — with one required change before running.** Switch the validation set
from `aime24.parquet` to a 200-question held-out `deepmath_val.parquet` (already
updated in the spec and config). Everything else is sound. Watch
`max_response_length` truncation in epoch 1. Frame GPQA/HLE results as
exploratory rather than primary validation targets.
