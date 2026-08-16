"""Pre-flight gate: assert a folded SFT parquet trains on exactly the inference context.

Run this before launching training. A format mismatch is visible at step 0, so this catches
the class of bug that made the first SFT run underperform the base model: the adapter was
trained on a native multi-turn transcript it never sees at evaluation, and val loss could
not detect it because the val split was in the same wrong format.

Checks, per row:
  1. the conditioning prompt equals `apply_chat_template([system, user],
     add_generation_prompt=True, enable_thinking=False)` token-for-token — i.e. exactly
     what VLLMProvider._render_messages sends at inference;
  2. the supervised span is exactly the assistant target plus its end-of-turn token;
  3. no thinking content is supervised;
  4. no simulated tool result is supervised (they live in the user turn as prose);
  5. action targets keep their <tool_call> (the converse guard);
  6. no degenerate rows (empty target) and nothing exceeds --max-length.

Usage:
    python scripts/check_sft_folded_format.py \\
        --folded data/training/sft/sft_folded_train.parquet \\
        --native data/training/sft/sft_train.parquet
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# Tool results shorter than this are substrings of ordinary prose ("1", "oo", "[15]") and
# would produce false leak reports; the structural guarantee is checked by gate 2 anyway.
_MIN_PROBE_LEN = 20


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folded", required=True, help="Folded parquet to check.")
    parser.add_argument("--native", default=None,
                        help="Native parquet it was folded from (enables the tool-leak "
                             "and question-coverage checks).")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Tokenizer to render with.")
    parser.add_argument("--max-length", type=int, default=16384,
                        help="Must match data.max_length in the training job.")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from verl_ext.folded_sft_dataset import FoldedSFTDataset

    tok = AutoTokenizer.from_pretrained(args.model)
    folded = pd.read_parquet(args.folded)

    # Build the dataset from the config TRAINING will use, not a friendlier one. A gate that
    # constructs its own minimal config tests a code path training never takes: verl passes
    # the whole `data` config, which ships keys we would otherwise never see — including
    # `enable_thinking_default: none` (the string "none"), which silently suppressed the
    # empty think block in the prompt while this gate kept passing. If verl is importable we
    # use its real defaults; otherwise we hard-code the same keys so the gate is no weaker.
    cfg = {"enable_thinking_key": "enable_thinking", "enable_thinking_default": "none",
           "pad_mode": "no_padding", "truncation": "error", "max_length": 1024}
    try:
        from omegaconf import OmegaConf
        import verl  # noqa: F401
        verl_defaults = Path(verl.__file__).parent / "trainer" / "config" / "sft_trainer_engine.yaml"
        if verl_defaults.is_file():
            loaded = OmegaConf.load(verl_defaults)
            cfg = OmegaConf.to_container(loaded.get("data", loaded), resolve=False)
            logger.info("Using verl's real data-config defaults from %s", verl_defaults)
    except Exception as e:  # noqa: BLE001 — the hard-coded mirror above is the fallback
        logger.info("verl not importable (%s); using the hard-coded mirror of its defaults.", e)

    cfg.update({"messages_key": "messages", "max_length": args.max_length,
                "truncation": "right", "pad_mode": "no_padding"})
    ds = FoldedSFTDataset(parquet_files=[args.folded], tokenizer=tok, config=cfg)

    tool_results: dict = {}
    native = None
    if args.native:
        native = pd.read_parquet(args.native)
        for _, r in native.iterrows():
            tool_results.setdefault(r["question"], []).extend(
                m["content"] for m in r["messages"] if m["role"] == "tool"
            )

    failures: Counter = Counter()
    lens = []
    n_tool_call_targets = 0

    for k in range(len(ds)):
        msgs = [dict(m) for m in folded.iloc[k]["messages"]]
        item = ds[k]
        input_ids = item["input_ids"].tolist()
        loss_mask = item["loss_mask"].tolist()
        lens.append(len(input_ids))

        if 1 not in loss_mask:
            failures["row_with_zero_supervised_tokens"] += 1
            continue

        prompt_ids = input_ids[: loss_mask.index(1)]
        expected = tok.apply_chat_template(
            msgs[:2], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if prompt_ids != expected:
            failures["prompt_differs_from_inference"] += 1

        supervised_ids = [t for t, m in zip(input_ids, loss_mask) if m == 1]
        expected_supervised_ids = tok(msgs[2]["content"], add_special_tokens=False)["input_ids"]
        expected_supervised_ids.append(tok.eos_token_id)
        if supervised_ids != expected_supervised_ids:
            failures["supervised_span_is_not_the_target"] += 1
        supervised = tok.decode(supervised_ids)
        if "<think>" in supervised or "</think>" in supervised:
            failures["thinking_in_supervised_span"] += 1
        for tr in tool_results.get(folded.iloc[k]["question"], []):
            probe = tr.strip()
            if len(probe) >= _MIN_PROBE_LEN and probe[:200] in supervised:
                failures["tool_result_in_supervised_span"] += 1
                break
        if not msgs[2]["content"].strip():
            failures["degenerate_empty_target"] += 1
        if len(input_ids) > args.max_length:
            failures["row_over_max_length"] += 1
        if "<tool_call>" in msgs[2]["content"]:
            n_tool_call_targets += 1

    a = np.array(lens)
    logger.info("=" * 70)
    logger.info("Rows checked:            %d", len(ds))
    logger.info("Targets with <tool_call>: %d", n_tool_call_targets)
    logger.info("Tokens: total %d | mean %.1f | p95 %d | max %d",
                a.sum(), a.mean(), np.percentile(a, 95), a.max())
    if native is not None:
        covered = set(native["question"]) == set(folded["question"])
        logger.info("Every native question present: %s", covered)
        if not covered:
            failures["questions_lost_or_added"] += 1
        logger.info("Trajectories %d → rows %d (%.2fx)",
                    len(native), len(folded), len(folded) / max(len(native), 1))

    if n_tool_call_targets == 0:
        failures["no_tool_call_survived_into_any_target"] += 1

    logger.info("=" * 70)
    if failures:
        for name, count in failures.most_common():
            logger.error("FAIL  %-40s %d", name, count)
        logger.error("Pre-flight FAILED — do not launch training.")
        return 1

    logger.info("All gates passed. Training context == inference context on every row.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
