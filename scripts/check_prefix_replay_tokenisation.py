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
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    frame = pd.read_parquet(args.demos)
    store = DemoStore.from_parquet(args.demos)

    checked = 0
    for i in range(min(args.n, len(frame))):
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
    print(f"        and the tool -> user remap is applied ({args.model}).")
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
