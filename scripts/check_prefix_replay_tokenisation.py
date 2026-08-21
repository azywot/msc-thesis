"""Check that replayed turns tokenise exactly as the daemon's proxy would.

A replayed teacher turn never reaches vLLM, so its token IDs are produced locally
by ``ReplayController``. The daemon's proxy produces them for generated turns
(``fine_tuning/agentflow/verl/daemon.py:216-225``) with exactly two calls:

    tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    tokenizer.encode(response_text, add_special_tokens=False)

with no ``enable_thinking`` kwarg and no appended EOS, on the messages the provider
sent *after* its ``tool`` -> ``user`` remap (``api_provider.py:94-100``).

If the two disagree, every prefix triplet is misaligned: ``prefix_mask`` marks
tokens that are not the teacher's, the prefix advantage lands on the wrong
positions, and training proceeds normally and reports success. No metric in the run
would reveal it. Hence this check.

CPU only; needs the tokenizer, which is cached after any previous run.

Usage:
    python scripts/check_prefix_replay_tokenisation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from fine_tuning.prefix_replay import ReplayController
from verl_ext.prefix_rft.demos import DemoStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demos", type=Path, default=Path("data/training/prefix_rft/prefix_demos.parquet")
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--n", type=int, default=10, help="demonstrations to check")
    parser.add_argument(
        "--mode",
        choices=["steps", "tokens", "both"],
        default="both",
        help="which replay mode's tokenisation to check",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    frame = pd.read_parquet(args.demos)
    store = DemoStore.from_parquet(args.demos)

    # _fail returns 1, so OR-ing accumulates a non-zero exit from either half while
    # still running both: one failing mode should not hide the other's result.
    rc = 0
    if args.mode in ("steps", "both"):
        rc |= _check_steps_mode(store, frame, tokenizer, args.n, args.model)
    if args.mode in ("tokens", "both"):
        rc |= _check_token_mode(store, frame, tokenizer, args.n)
    return rc


def _check_steps_mode(store, frame, tokenizer, n, model):
    """The existing check, unchanged: whole replayed turns tokenise as the proxy would."""
    checked = 0
    for i in range(min(n, len(frame))):
        row = frame.iloc[i]
        steps = store.steps(row["question"])
        if not steps:
            return _fail(f"row {i}: the store does not resolve its own question text")

        messages = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": row["question"]},
        ]
        payload = json.dumps({"messages": messages, "use_thinking": False})

        replayed = ReplayController(steps, k=1, tokenizer=tokenizer).next_response(payload)

        proxy_prompt = list(
            tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        )
        proxy_response = list(
            tokenizer.encode(steps[0]["response"], add_special_tokens=False)
        )

        if replayed["prompt_token_ids"] != proxy_prompt:
            return _fail(
                f"row {i}: prompt tokenisation differs from the proxy's\n"
                f"  replay: {replayed['prompt_token_ids'][:12]}\n"
                f"  proxy:  {proxy_prompt[:12]}"
            )
        if replayed["response_token_ids"] != proxy_response:
            return _fail(
                f"row {i}: response tokenisation differs from the proxy's\n"
                f"  replay: {replayed['response_token_ids'][:12]}\n"
                f"  proxy:  {proxy_response[:12]}"
            )
        checked += 1

    # The provider remaps tool turns to user before the request leaves it, so the
    # proxy templates the remapped list and replay must too.
    steps = store.steps(frame.iloc[0]["question"])
    remap_got = ReplayController(steps, k=1, tokenizer=tokenizer).next_response(
        json.dumps({"messages": [{"role": "tool", "content": "search result"}]})
    )["prompt_token_ids"]
    remap_want = list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "search result"}],
            add_generation_prompt=True,
            tokenize=True,
        )
    )
    if remap_got != remap_want:
        return _fail("the tool -> user remap is not applied before templating")

    print(f"PASSED: replay tokenisation matches the proxy's on {checked} demonstrations,")
    print(f"        and the tool -> user remap is applied ({model}).")
    return 0


def _check_token_mode(store, frame, tokenizer, n):
    """Check the split turn the way the training batch will see it.

    Two properties, both of which fail silently in a real run:

    1. The teacher tokens sent as a prefill must survive the round trip through text.
       ``_safe_prefix`` backs the boundary off until they do, so what it returns must
       re-encode to itself exactly.
    2. The response row the batch sees is ``prefix_ids + encode(continuation)``, and
       ``prefix_mask`` marks its first ``len(prefix_ids)`` positions. Those ids must be
       a true prefix of the teacher's own encoding of that decision, or the mask marks
       tokens the teacher never wrote.
    """
    checked = 0
    for i in range(min(n, len(frame))):
        row = frame.iloc[i]
        steps = store.steps(row["question"])
        if not steps:
            return _fail(f"row {i}: the store does not resolve its own question text")

        messages = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": row["question"]},
        ]
        payload = json.dumps({"messages": messages, "use_thinking": False})

        ctrl = ReplayController.from_token_fraction(steps, l=0.8, tokenizer=tokenizer)
        for _ in range(ctrl.k):
            ctrl.next_response(payload)
        partial = ctrl.next_partial(payload)
        if partial is None:
            # A budget that landed on a decision boundary. Legal, nothing to check.
            continue

        prefix_ids = partial["prefix_ids"]
        if list(tokenizer.encode(partial["prefix_text"], add_special_tokens=False)) != list(
            prefix_ids
        ):
            return _fail(
                f"row {i}: the prefill text does not re-encode to the ids it came from; "
                "_safe_prefix should have backed the boundary off further"
            )

        full = tokenizer.encode(
            str(steps[ctrl.split_index]["response"]), add_special_tokens=False
        )
        if list(full[: len(prefix_ids)]) != list(prefix_ids):
            return _fail(
                f"row {i}: the prefill ids are not a prefix of the teacher's own encoding "
                "of that decision; prefix_mask would mark tokens the teacher never wrote"
            )

        proxy_prompt = list(
            tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        )
        if partial["prompt_token_ids"] != proxy_prompt:
            return _fail(f"row {i}: the split turn's prompt ids differ from the proxy's")

        # 3. The load-bearing one. The training row is prompt + prefix_ids + continuation,
        # so prefix_ids must be exactly the tokens vLLM sees between the generation
        # prompt and whatever the model writes next. Checking prompt_token_ids against
        # the proxy above does NOT establish this: both sides are the same call, so it
        # is true by construction and proves nothing about the prefill.
        #
        # Note the enable_thinking=False on the right-hand side. With thinking off,
        # Qwen3's template emits "<think>\n\n</think>\n\n" after the generation prompt,
        # and continue_final_message emits it too. The proxy omits it for every turn in
        # this pipeline, prefixed or not, so the recorded prompt is 4 tokens shorter than
        # what vLLM conditioned on. That gap is pre-existing and identical for whole
        # replays, generated turns and split turns; it is deliberately not asserted here.
        # What must hold is that nothing extra appears *between* those 4 tokens and the
        # teacher's, which is what this comparison pins down.
        continued = list(
            tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": partial["prefix_text"]}],
                continue_final_message=True,
                add_generation_prompt=False,
                tokenize=True,
            )
        )
        rendered = list(
            tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, enable_thinking=False
            )
        )
        if continued != rendered + list(prefix_ids):
            return _fail(
                f"row {i}: the prefill does not concatenate. What vLLM conditions on under "
                f"continue_final_message ({len(continued)} tokens) is not the rendered "
                f"prompt ({len(rendered)}) followed by the teacher's {len(prefix_ids)} "
                "tokens, so response_ids would not be the tokens the model actually saw "
                "and prefix_mask would mark the wrong positions."
            )

        checked += 1

    print(f"PASSED (token mode): {checked} split turns tokenise as the batch will see them")
    return 0


def _fail(message: str) -> int:
    print(f"FAILED: {message}")
    print(
        "\nEvery prefix triplet would be misaligned. Fix ReplayController before running "
        "any training, and check it against daemon.py:216-225 and api_provider.py:94-100."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
