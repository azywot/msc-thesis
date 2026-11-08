# RSI phase 1: overview

**Status (2026-11-08):** implemented and tested; first 10-round searches running.
**Code:** `feat/mass-rsi` ([PR #37](https://github.com/azywot/msc-thesis/pull/37)).
**More detail:** `docs/pipelines/rsi.md` (how to run it),
`docs/superpowers/specs/2026-11-04-recursive-self-improvement-design.md` (design).

## 1. Goal

Recursive self-improvement (RSI) adapts MASS (Lee et al., arXiv:2610.12176) to
CoSMAS. One cycle has two steps:

1. **Search:** the model, acting as a *reflector*, rewrites the prompts of its
   own agent team.
2. **Internalise:** the model generates trajectories under the improved prompts
   and is fine-tuned on them, with the prompts swapped back to the originals.

Question: can a small multi-agent system improve itself from its own outputs and
a verifier, and how does that compare with the other fine-tuning directions?

**Phase 1 is the search step only, on the base model.** Qwen3-8B plays every
role: orchestrator, both sub-agents (`web_search`, `code_generator`) and
reflector. No weights change. Phase 2 (fine-tuning) starts only after Phase 1
has been inspected.

## 2. How the search works

The search keeps one current best prompt set, the *champion*, starting from
today's prompts (the *seed*). Each round:

1. The champion answers 16 questions; the reflector reads these trajectories,
   including sub-agent inputs and outputs and the gold answers.
2. The reflector proposes 3 candidate edits. Each edits **one unit**: the
   orchestrator prompt, the `web_search` template or the `code_generator`
   template.
3. Candidates that change protected parts (tool signatures, call format, answer
   format) or leak training data are rejected.
4. The rest are scored on 16 screening questions. The best one, if it is at
   least as good as the champion, is scored on 50 held-out questions and replaces
   the champion only if it is **strictly better**.

There are two domains, each with its own prompt set: **search** (HotpotQA + NQ)
and **math** (DeepMath). Only the training set is used, 900 questions per
domain, split into 300 shown, 200 screening, 50 held-out and 350 unused (kept
for confirmation). A run is 10 rounds on one H100, about 5 minutes per round.

## 3. Differences from MASS

| MASS | Here | Why |
|---|---|---|
| The model judges its own outputs | A verifier with gold answers | Our tasks have ground truth |
| One workflow per task | One prompt set per domain | 1,800 short questions, not 8 long tasks |
| 27B optimiser, rewrites everything | 8B reflector, one unit per candidate | A smaller model gets a smaller job |
| Optimiser sees final outputs only | Reflector also sees sub-agent I/O and gold answers | Needed to improve sub-agent prompts; leakage is blocked by checks |

## 4. Safeguards

- **Leakage:** a candidate is rejected if it copies an 8-word sequence from any
  training question, or a gold answer it was shown.
- **Overfitting:** separate pools for showing, screening and promotion.
- **Winner's curse:** the champion is selected and scored on the same 50
  held-out questions, so its gain is optimistic. A confirmation run re-scores
  seed and champion on the 200 unused questions; only that number is reported.
- **Weak reflector:** a control with Qwen3-32B as reflector (same 8B team, same
  budget) separates "the method fails" from "the 8B reflector fails".

## 5. What was built

- **Editable prompts:** the orchestrator prompt is split into named parts, and
  the sub-agent templates can be swapped at runtime. With no edits the prompts
  are byte-identical to today's, so existing experiments are unaffected.
- **The search loop** (`src/rsi/`, entry point `scripts/run_prompt_search.py`):
  resumable round by round, saves every reflector input, proposal and score,
  and writes a markdown report per run.
- **Jobs:** configs and SLURM jobs for the 8B search, the 32B control and the
  confirmation run.
- **Optional:** any benchmark can be run under a searched prompt set
  (`prompt_bundle_path`).

All 1,182 tests pass; the GPU pipeline was checked with 1-round smoke runs.

## 6. Status and first results

All runs so far use Qwen3-8B in every role, reflector included.

| Domain | Seed held-out | Rounds done | Champion so far |
|---|---|---|---|
| search | 38% | 5 of 10 | 40% (one promotion, +1 question: within noise) |
| math | 54% | 6 of 10 | seed (no promotion) |

All 33 proposals so far edited a sub-agent template (`web_search` in search,
`code_generator` in math); none edited the orchestrator. 14 of the 33 were
rejected, all for dropping a required template placeholder.

## 7. Next steps

1. Finish the 10-round runs and confirm any champion on the 200 unused questions.
2. If the 8B's confirmed champion beats the seed: start Phase 2 (self-RSI plus a
   no-search control that fine-tunes on the model's own correct trajectories).
3. If not: run the 32B-reflector search. If that also fails, revisit the search
   design before any fine-tuning.

## 8. Open questions

**Which parts should a candidate change, and how granularly?** The reflector
edits whichever tool the visible failures point to, and never the orchestrator.
That limits what it can improve: in math the team uses a tool on only about 1 in
5 questions, so a `code_generator` edit cannot affect the rest. Options:

- **Free choice of one unit** (current).
- **A fixed unit per candidate:** of the 3 candidates per round, one edits the
  orchestrator, one `web_search`, one `code_generator`. Every round then tries an
  orchestrator edit, and each gain is still traceable to one change.
- **Several units per candidate:** also reaches the orchestrator, but one bad
  part rejects the whole candidate and gains can no longer be attributed.

Proposed rule after the 10-round runs: if nothing is promoted and the edits stay
on sub-agent templates, rerun with a fixed unit per candidate and compare; if
the edits are spread across units and still nothing improves, go to the 32B
control instead.
