# Prefix-RFT token mode: a mid-turn prefix, selectable against the step prefix

**Date:** 2026-08-19
**Status:** proposed
**Extends:** `docs/superpowers/specs/2026-08-17-prefix-rft-design.md`
**Paper:** Huang et al., ICML 2026, arXiv:2507.01679v3

> **Source note.** `papers/` and `repos/` were local reference material and are not
> in this checkout (they were never tracked by git). Every quotation and line
> citation below is carried forward from the 2026-08-17 spec and from the module
> docstrings that were written against the sources while they were present. Anyone
> re-deriving this work should restore both directories first.

## Goal

Prefix-RFT currently measures prefix length in teacher *decisions*. The paper
measures it in *tokens* and its prefix can end anywhere, including halfway through a
sentence. This spec adds the paper's token measure as a second mode and puts a switch
between them, so the two can be compared on the same data with everything else held
fixed.

The question being answered is not "which is more faithful", since token mode plainly
is. It is whether a mid-turn prefix is a sensible object in a multi-agent setting at
all. Handing the model the first 40 tokens of a teacher's tool call and asking it to
finish that call is either a useful curriculum or a source of malformed actions, and
the run will say which.

Success means: both modes run end to end, `prefix_mask` is correct token for token in
each, step mode is bit-identical to what it is today, and the difference between the
two is visible in the metrics.

## What the paper does, and what we do today

**Paper (§3, A.2).** A demonstration is one response `y*`. Draw `l ~ U(low_t, 0.95)`
with `low_t` cosine-decaying from 0.95 to 0.05. The prefix is the first `l` fraction
of `y*`'s tokens. The policy continues from that token. A guard forces
`prefix_len <= demo_len - 1` (`recipe/prefix_rft/rl_dataset.py:300-301`) so there is
always a continuation to score.

**Today (step mode).** A demonstration is `m` decisions. `k = clamp(floor(l * m), 0, m-1)`
whole decisions are replayed and the model takes over at decision `k+1`
(`src/verl_ext/prefix_rft/schedule.py:117`). The prior spec recorded two accepted
consequences of this and named the fix:

> `k` is a staircase, not a continuous ratio. Teacher trajectories average 2.98
> decisions, so the cosine decay is quantised into roughly three levels. [...] 273 of
> the 1358 demonstrated questions have a single decision. For them `k <= m-1 = 0`
> always, so they never carry a prefix. [...] The mitigation, if it later matters, is
> a token-level prefix on the boundary decision, which restores continuous control.

This spec is that mitigation, built as an alternative rather than a replacement.

## Design

### The split

Let the demonstration's decisions have response token lengths `n_1 .. n_m` and total
`T = sum(n_i)`, where each `n_i` is the length of
`tokenizer.encode(step["response"], add_special_tokens=False)`, the same call
`ReplayController` already makes.

```
B = clamp(floor(l * T), 0, T - 1)          token budget for this rollout
j = largest index with n_1 + ... + n_j <= B      decisions replayed whole
r = B - (n_1 + ... + n_j)                        leftover tokens
```

Decisions `1..j` are replayed exactly as they are today. If `r > 0`, decision `j+1` is
**split**: its first `r` tokens are handed to the model as a prefill and the model
writes the rest of that same turn. If `r == 0` the rollout is indistinguishable from
step mode at `k = j`.

The `B <= T - 1` clamp is the paper's guard applied to the concatenated demonstration,
so at least one token is always generated. Because of it, `j < m` whenever `r > 0`, so
the split decision always exists.

```
y* = [resp_1 | resp_2 | resp_3]        teacher, m = 3, T = 120 tokens

l = 0.80   B = 96    ####|####|##~~~~     j = 2, r = 16, decision 3 split
l = 0.35   B = 42    ####|##~~|~~~~~~     j = 1, r = 12, decision 2 split
l = 0.00   B = 0     ~~~~|~~~~|~~~~~~     j = 0, r = 0,  plain GRPO

####  teacher tokens, prefix_mask = 1, in the loss
~~~~  on-policy continuation, prefix_mask = 0
```

### What token mode unlocks

The 273 single-decision demonstrations become prefixable. In step mode `m = 1` forces
`k <= 0`, so they always run as plain GRPO. In token mode the guard is a token guard,
so a single-decision demonstration can be split partway through its only response.
That takes prefixable coverage from 1085 questions to all 1358 with a demonstration.
This is a substantive difference between the modes, not a side effect, and it should be
reported as one when the two runs are compared.

### Where the mode lives

The driver owns the schedule because it owns `global_step`. It keeps owning the choice
too, and the workers infer the mode from what arrives in the task payload:

| mode | driver dispatches | worker behaviour |
|---|---|---|
| `steps` | `prefix_k: int` | replay `k` whole decisions (unchanged) |
| `tokens` | `prefix_l: float` | compute `B`, `j`, `r` locally and split |

There is deliberately no `PREFIX_MODE` environment variable. `PREFIX_DEMOS_PATH` is
set twice today, once in `env` for the workers and once in `overrides` for the driver,
and two settings that must agree are two settings that can disagree. Keying off the
dispatched payload makes disagreement unrepresentable.

The worker raises if both keys are present or if `prefix_l` arrives without a
tokenizer, rather than downgrading silently. Every existing silent downgrade in
`_make_controller` is deliberate and documented; this one would mean the driver and
worker disagree about the experiment, which is not a downgrade but a wrong run.

### Serving a split turn

A fully replayed turn never reaches vLLM. A split turn must, because the model has to
continue it. vLLM 0.10.1.1 supports this through `continue_final_message`
(`vllm/entrypoints/openai/protocol.py:466`), which formats the final message
open-ended with no EOS. It rejects `continue_final_message` and `add_generation_prompt`
both being true (`protocol.py:925`), so the request sets `add_generation_prompt: false`.

`ReplayProvider` handles the split turn in four steps:

1. Build `prompt_ids = apply_chat_template(messages, add_generation_prompt=True, tokenize=True)`
   on the tool-to-user remapped messages. This is the same call the proxy makes
   (`fine_tuning/agentflow/verl/daemon.py:216-225`), so the prompt is tokenised as if
   the turn had been generated normally.
2. Send `messages + [{"role": "assistant", "content": prefix_text}]` through the
   capturing provider with `continue_final_message: true, add_generation_prompt: false`.
3. The capturing provider appends its own entry to `captured_turns`
   (`rollout.py:201-210`). Overwrite that entry: `prompt_ids` from step 1,
   `response_ids = prefix_ids + encode(continuation)`,
   `response_text = prefix_text + continuation`, `is_prefix = True`,
   `prefix_len = len(prefix_ids)`.
4. Return a `GenerationResult` whose text is the full turn, so the orchestrator parses
   the whole decision rather than only the part the model wrote.

Step 3 is why the vendored daemon does not need touching. The proxy will compute
`prompt_token_ids` for the request it saw, which includes the partial assistant
message and is wrong for training. We discard it. The overwrite asserts that the entry
being corrected is the one the call just produced, so a future change to the capturing
provider's batching surfaces as a failure rather than as a misaligned mask.

### Passing the flag through the provider

`OpenAIProvider._generate_single` builds `extra_body` from `self.config`
(`api_provider.py:106-117`) and there is no per-request seam. The JSON payload
protocol it already parses (`messages`, `use_thinking`) gains one optional key,
`continue_final_message`, which sets `continue_final_message` and
`add_generation_prompt: false` on the request. This is an additive change to
`agent_engine`, inert for every caller that does not set it.

### Tool calls in a split turn

Only fully replayed decisions arm the teacher's stored tool result. A split decision's
tool call is whatever the model wrote after the prefill, so it must execute for real.
`ReplayController._pending_tool_result` is already single-use and set only in
`next_response` for a complete decision, so this needs no new logic, only a test that
pins it.

A tool call split in the middle of its JSON arguments may come back malformed,
especially early in training when `l` is near 0.95. That is a finding about mid-turn
prefixes in an agentic setting, which is what the comparison is for. It will show up as
depressed `actor/reward_with_prefix` in the first steps.

### Mask contract: bool becomes int

`build_prefix_mask` fills a row with 1s when `trace["is_prefix"]` is true
(`masks.py:28`). A split turn needs a row that is 1 on its first `r` tokens and 0
after, so the trace carries `prefix_len: int` and the fill becomes:

```python
n_prefix = int(trace.get("prefix_len", 0) or 0)
if trace.get("is_prefix", False) and n_prefix == 0:
    n_prefix = length          # fully replayed turn, back-compatible
n_prefix = min(n_prefix, length)
rows.append([1] * n_prefix + [0] * (max_response_length - n_prefix))
```

The `is_prefix` flag stays and stays meaningful: it marks a turn as carrying prefix
tokens, which is what `is_prefix_rollout` aggregates over
(`daemon.py:218`). A split turn sets it true with `prefix_len < length`.

This changes a contract three other places read, so all three move together:

- `prefix_replay.py` writes `prefix_len` into `captured_turns`
- `rollout.py:349` carries it in the `Triplet` metadata alongside `prefix`
- `daemon._rebuild_prefix_rows` reads it back out of that metadata

### What does not change

`advantage.py` and `entropy.py` take `prefix_mask` as a token-level tensor and never
assume a row is uniform. A partially filled row is already a legal input to both. The
actor, the worker, the entropy keep ratio and the group statistics are untouched.

Step mode is untouched end to end. `prefix_k` still means what it means, the schedule
class is unchanged, and `prefix_len` defaults to the full response length for a
replayed turn, which reproduces today's mask exactly. The point of the comparison is
that only one thing differs.

## Surface

**Config.** `prefix_rft.mode: steps` in `src/verl_ext/prefix_rft/config/prefix_rft_trainer.yaml`,
defaulting to `steps` so every existing config and the completed 012 run keep their
meaning.

**Flag.** `--prefix-mode {steps,tokens}` on `scripts/launch_verl.py`, which appends the
Hydra override `prefix_rft.mode=<value>`. It goes on the driver only, because the driver
is the process that owns the schedule and the workers follow what it dispatches.
`scripts/train_orchestrator.py` needs no flag and gets none.

**New experiment config.** `experiments/configs/fine_tuning/config_prefix_rft_tokens.yaml`,
identical to `config_prefix_rft.yaml` but for `prefix_rft.mode: tokens`, so the
comparison is a one-line diff.

## Metrics

| metric | mode | meaning |
|---|---|---|
| `actor/prefix_l` | tokens | mean sampled `l`, the paper's own quantity |
| `actor/prefix_split_fraction` | tokens | share of prefixed rollouts with `r > 0` |
| `actor/prefix_steps` | steps | mean `k`, unchanged |
| `actor/off_ratio` | both | prefix tokens over response tokens |

`_summarise_prefix_dispatch` (`daemon.py:152`) builds this block from the dispatched
`ks` and gains a token-mode branch that summarises `ls` instead. It must not take a
further draw from the schedule: it already calls `sample_l` a second time to report
`low` and `high`, which advances the RNG, and doubling that would change the curriculum
the run actually sees.

`off_ratio` is the one that makes the modes directly comparable, and in token mode it
becomes directly comparable to the paper's reported 5 to 10 percent as well.

## Risks

**Token boundaries do not always round-trip.** `decode(ids[:r])` re-encoded need not
give back `ids[:r]`, because the split can land inside a multi-token character or a
mergeable pair. The prefill is sent as text, so a bad boundary silently shifts every
prefix token. Mitigation: back the boundary off one token at a time until the round
trip holds, and assert the property in the check script rather than trusting it.

**The chat template may not concatenate.** The design assumes
`apply_chat_template(msgs, add_generation_prompt=True) + encode(prefix_text)` is what
the model actually conditions on when the request uses `continue_final_message`. Qwen3's
template with thinking disabled inserts an empty think block, and it is not obvious
without checking that the two paths agree. This is verified offline before any GPU time,
by the same script that already guards the fully-replayed case.

**Malformed actions from split tool calls.** Discussed above. A finding, not a defect,
but it could depress early training enough to make the comparison unflattering for
reasons that are about JSON rather than about learning. Recorded so the read of the
results accounts for it.

**Mask misalignment is silent.** This is the standing risk of the whole method, as
`check_prefix_replay_tokenisation.py` says: a wrong mask trains normally and reports
success. Token mode widens the surface, so the check script grows a token-mode case
and stays a preflight gate rather than an optional tool.

## Verification

- `scripts/check_prefix_replay_tokenisation.py --mode tokens`: for real demonstrations,
  assert the round trip at the chosen boundary and assert that prompt plus prefix
  matches the concatenation the training batch will contain.
- Unit tests, all CPU and all in `agent_engine`:
  - budget arithmetic: clamps, `l = 0`, `l = 0.95`, `m = 1`, `T = 1`, exact-boundary `r = 0`
  - `ReplayController` in token mode: split position, replay counts, and that a split
    decision does not arm the teacher's tool result
  - `build_prefix_mask`: partial rows, back-compatibility with `is_prefix` alone,
    truncation to `max_response_length`
- Existing suites must pass unchanged, which is the evidence that step mode did not move.
- A tiny GPU run on the existing tiny split before anything longer, checking that
  `off_ratio` is non-zero and `prefix_split_fraction` is near 1.
