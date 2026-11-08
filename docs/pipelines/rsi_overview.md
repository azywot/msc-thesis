# RSI phase 1: overview and progress

**Status (2026-11-08):** implemented, tested, first GPU smoke runs in progress.
**Branch:** `feat/mass-rsi` ([PR #37](https://github.com/azywot/msc-thesis/pull/37)).
**Detail:** the operational guide is `docs/pipelines/rsi.md`; the design is
`docs/superpowers/specs/2026-11-04-recursive-self-improvement-design.md`; the
task-by-task plan is `docs/superpowers/plans/2026-11-08-rsi-phase1-prompt-search.md`.

This page is a summary for review: what the work is for, how it works, what was
built, how it is safeguarded, what runs next, and what is still open.

---

## 1. What this is and why

Recursive self-improvement (RSI) implemented here consists of:

1. **Search:** the model, acting as a reflector, improves the prompts of its own
   agent team (orchestrator and sub-agents).
2. **Internalise:** the model generates trajectories under the improved prompts,
   the prompts are swapped back to the originals, and the model is fine-tuned on
   those trajectories, so it keeps the improved behaviour without the improved
   prompts.

The question it answers: **can a small multi-agent system improve itself using
only its own outputs and a verifier, and how does that compare with the other
fine-tuning directions?**

**Phase 1 (this work) is the search step only, run once, on the base model.**
Qwen3-8B plays every role: orchestrator, `web_search` sub-agent,
`code_generator` sub-agent, and reflector. No weights change. Phase 2
(collection and fine-tuning) is deliberately not built yet: it is launched by
hand only after Phase 1 has been inspected.

## 2. How it differs from MASS

| MASS | Here | Why |
|---|---|---|
| The model judges its own outputs pairwise | A verifier (`evaluate_answer`, ground truth) scores everything | Our tasks have gold answers; a self-judge adds noise for no benefit |
| One workflow per task (8 long tasks) | One prompt bundle per domain: **search** (HotpotQA + NQ) and **math** (DeepMath) | 1,800 short questions; a per-question search would be too expensive and could write the answer into the prompt |
| 27B optimiser, 262K context | 8B reflector, 32K context, a smaller job per round | It edits one unit per round, sees failure labels, and 3 candidates are sampled per round |
| The optimiser reads deliverables only | The reflector reads sub-agent inputs and outputs, and the gold answer | Needed to improve sub-agent prompts and to tell format errors from wrong facts; leakage is blocked by checks instead (section 5) |

## 3. How one round works

The search keeps a single **champion** prompt set (elite of one), starting from
the seed prompts we use today. Each round:

```
champion runs on 16 "shown" questions
        |
reflector reads: current + seed prompts, the 16 trajectories (incl. sub-agent
I/O and gold answers), past edits with scores, last round's rejection reasons
        |
3 candidate edits sampled (each edits ONE unit; failed parses re-sampled up to 2x)
        |
validation (protected parts unchanged)  ->  leakage guard
        |
screen: candidates vs champion on 16 "screen" questions; best one goes on
        only if it scores >= champion
        |
held-out: that candidate on 50 held-out questions; promoted only if
        STRICTLY better than the champion
```

**What the reflector may edit** (one unit per proposal):

| Unit | Editable parts |
|---|---|
| `orchestrator` | opening and closing guidance, the worked example, the planning-turn and acting-turn prompts, tool description strings, the tool-use note |
| `web_search` | the sub-agent's page-analysis template |
| `code_generator` | the sub-agent's code-task template |

Tool names and parameters, the tool-call format, the `\boxed{}` answer rule and
the memory-turn structure are protected and checked before any rollout.

**Data.** Only the training parquet is used (1,800 rows); no benchmark data is
touched. Each domain is split into three disjoint pools: shown 300, screen 200,
held-out 50. The remaining 350 rows per domain are kept unused for a final
confirmation (section 5).

**Default budget:** 10 rounds per domain, 3 candidates per round, one H100,
24 h job limit; every round is saved, so an interrupted job resumes.

## 4. What was built

All code is new or opt-in; existing experiments behave exactly as before. The
seed prompts render byte-identically to today's prompts, and the existing
prompt characterisation tests pass unchanged.

| Area | Change |
|---|---|
| Prompt plumbing (`src/agent_engine/prompts/`, `tools/`, `core/`) | The orchestrator prompt is now built from named, overridable parts; the `web_search` and `code_generator` sub-agent templates can be replaced at runtime; the orchestrator logs every sub-agent call (input and output) so the reflector can see it |
| `src/rsi/bundle.py` | The **prompt bundle**: a plain-data object holding every editable prompt, saved as `bundles/v<N>.json` |
| `src/rsi/validate.py` | Rejects proposals that touch protected parts, break template placeholders, or contain an invalid worked example |
| `src/rsi/leakage.py` | Leakage guard (section 5) |
| `src/rsi/data.py` | Domain loading and stratified, seeded pools |
| `src/rsi/executor.py` | Runs the AgentFlow team under a given bundle and scores it with the verifier |
| `src/rsi/reflector_input.py`, `reflector.py` | Builds the reflector prompt within a 14K-token budget; two reflector back-ends (8B in-process on the executor's own vLLM engine, or any OpenAI-compatible server for the 32B control); parses proposals |
| `src/rsi/search.py` | The round loop, resumable from disk, with per-round records |
| `src/rsi/report.py`, `confirm.py` | A markdown report per run; a confirmation run on unused questions |
| `scripts/run_prompt_search.py` | Entry point, with `--dry-run`, `--report-only` and `--confirm N` |
| `experiments/configs/rsi/phase1/`, `jobs/rsi/` | Configs and SLURM jobs for the 8B reflector, the 32B control and confirmation |
| `prompt_bundle_path` (experiment config) | Optional: run any benchmark under a searched bundle, analogous to the GEPA inference runs |

**Tests:** 1,182 passed, 3 skipped, 1 expected failure (full suite). Every task
was reviewed separately, followed by a whole-branch review whose findings were
fixed before the first GPU run.

## 5. Safeguards against fooling ourselves

| Risk | Guard |
|---|---|
| The reflector copies a question or answer into the prompt (it sees gold answers) | Reject any candidate containing an 8-word sequence from any training question, or a gold answer it was shown. Very short answers ("0", "yes") are exempt, since banning them would reject almost every prompt; only leaks the champion does not already contain count |
| Overfitting to the questions used for selection | Separate pools for showing, screening and promotion; promotion needs a **strict** gain on held-out |
| Winner's curse: the champion is picked and scored on the same 50 held-out questions, so "beats seed" is optimistic | `--confirm` re-scores seed and champion on 200 questions outside every pool, with paired counts (both right, seed only, champion only, neither). Only the confirmed number is a result |
| A weak 8B reflector hides whether the method works | A 32B-reflector control arm with the same executor and budget |
| Noise | 50 held-out questions means one question is two points; the report says so, and decisions use the confirmation |

**Why the n-gram window is 8.** A larger window would miss short questions
entirely: NQ questions have a median of 9 words, so with a 13-word window 97%
of NQ questions could be pasted whole without detection. At 8, only 5 phrases
are shared by 5 or more training questions (all DeepMath boilerplate such as
"determine whether the following statement is true or false").

## 6. Experiments and what decides the next step

| # | Run | Resources | Purpose |
|---|---|---|---|
| 1 | Smoke: 1 round per domain, 8B reflector | 1 GPU | Check the GPU path end to end (**done**) |
| 2 | Main: 10 rounds per domain, 8B reflector | 1 GPU, 24 h each | Can the 8B improve its own team's prompts? (**running**) |
| 3 | Control: 10 rounds per domain, 32B reflector | 3 GPUs, 24 h each | Is the 8B reflector the bottleneck? |
| 4 | Confirmation of each finished run | 1 GPU, 8 h | The reportable seed-vs-champion comparison |
| 5 | Optional: benchmarks under the searched prompts | as normal inference | Diagnostic only (GAIA, HLE, MuSiQue for search; AIME for math) |

**Decision rule for Phase 2** (from the design):
1. The 8B reflector's champion beats the seed (confirmed): run Phase 2 with the
   self-RSI arm and the no-search control (fine-tuning on the model's own
   correct trajectories under the seed prompts).
2. It does not: run the 32B-reflector search.
   - 32B succeeds: run the 32B-reflector arm and the control; report "8B cannot
     yet improve its own team's prompts, 32B can".
   - 32B also fails: revisit the search design before any fine-tuning.

A flat or negative result is acceptable as long as it can be attributed (search
versus internalisation, weak reflector versus no signal).

## 7. Current status

- **Model:** all runs so far use Qwen3-8B in every role, reflector included.
  The 32B-reflector control has not been run yet.
- **Done:** implementation, reviews, documentation, a dry run of both domains
  (pools of 300 / 200 / 50 built correctly).
- **Smoke runs (1 round per domain):** the pipeline runs end to end on the GPU,
  about 10 to 13 minutes per round.
  - Seed held-out accuracy: **search 38%, math 54%** (50 questions each).
  - All 6 reflector proposals parsed; 4 passed validation. All 6 edited a
    sub-agent template, none the orchestrator.
  - No candidate was promoted after one round (search tied, math one question lower).
- **Next:** the 10-round main runs for both domains are running.

## 8. Open risks and known limitations

- **The 8B drops template placeholders.** The JSON format was not a problem in
  the smoke run, but 2 of 6 proposals rewrote a sub-agent template without its
  required placeholders and were rejected. To watch over 10 rounds.
- **Signal versus noise:** gains of a few held-out questions are within noise
  until confirmed on the 200 unused questions.
- **Not in the reflector's view:** the orchestrator's per-step reasoning text is
  not recorded in the trajectories, only its actions and results.
- **Minor:** the 32B client sends its 3 requests one after another (slower, not
  wrong); a hard kill can lose the latest web-search cache entries (rounds
  already on disk are safe).

## 9. Open questions

**Which parts should a proposal change, and how granularly?** In the smoke run
the reflector edited only sub-agent templates, never the orchestrator. It
followed the visible failures: in math, the only two wrong answers in its
minibatch had used the code tool, while most questions are solved without tools,
so a code-template edit reaches few questions. Options considered:

- **Keep free choice of one unit per proposal** (current).
- **Fixed unit per candidate slot:** of the 3 candidates per round, one edits
  the orchestrator, one `web_search`, one `code_generator`. Every round tries an
  orchestrator edit, and each candidate still changes one thing.
- **Several units per proposal:** reaches the orchestrator too, but a mistake in
  any part rejects the whole candidate, outputs get longer, and a gain can no
  longer be traced to one change. Not preferred for an 8B reflector.

Decided after the 10-round runs:
1. A champion is promoted: keep the current setup and confirm it.
2. No promotion, and most proposals (about 80% or more) edit sub-agent
   templates: rerun with a fixed unit per slot and compare with the free-choice run.
3. No promotion, with edits spread across units: the bottleneck is the 8B's
   proposals or held-out noise, so run the 32B-reflector control instead.
