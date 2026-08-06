"""Single-turn SFT dataset whose rendered prompt equals the inference prompt exactly.

``MultiTurnSFTDataset`` tokenises each turn in isolation, so an assistant target always
begins right after ``<|im_start|>assistant\\n``. Qwen3's template, rendering a
conversation as a whole (and the orchestrator at inference with thinking off), first
emits an empty ``<think>\\n\\n</think>\\n\\n`` block. The adapter is therefore trained to
start generating at a position it is never sampled from, on every target: a constant
4-token gap that ``data.ignore_input_ids_mismatch=True`` silences rather than fixes.

Because memory-folded rows are single-turn by construction
(``[system, user, assistant]``, see ``scripts/build_sft_parquet.py --format folded``),
the fix is small: render the prompt with ``add_generation_prompt=True``, which is
byte-for-byte the string ``VLLMProvider._render_messages`` sends at inference, then
supervise the target plus its end-of-turn token and nothing else.

Wired in via VERL's supported hook::

    data.custom_cls.path=src/verl_ext/folded_sft_dataset.py
    data.custom_cls.name=FoldedSFTDataset

With this class ``data.ignore_input_ids_mismatch`` is unnecessary and should be removed,
so that a future format regression fails loudly instead of being suppressed.
"""

import logging
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FoldedSFTDataset(Dataset):
    """One supervised decision per row; prompt identical to the orchestrator's.

    Loss mask is 0 across the whole prompt (system turn, folded user turn, and the
    generation prompt including the empty think block) and 1 across the assistant
    target and its terminating ``<|im_end|>``.

    Simulated tool results live inside the folded user turn as ``- Result:`` prose, so
    they are masked structurally: there is no path by which sub-agent output can enter
    the loss. The ``<tool_call>`` block inside an action target *is* supervised, because
    issuing the call is the orchestrator's own decision.
    """

    def __init__(
        self,
        parquet_files,
        tokenizer,
        config=None,
        processor: Optional[object] = None,
        max_samples: int = -1,
    ):
        config = config or {}
        self.tokenizer = tokenizer
        self.processor = processor
        self.messages_key = config.get("messages_key", "messages")
        self.max_length = config.get("max_length", 1024)
        self.truncation = config.get("truncation", "error")
        self.pad_mode = config.get("pad_mode", "right")

        # `enable_thinking` decides whether Qwen3's generation prompt ends with the empty
        # `<think>\n\n</think>\n\n` block, which is exactly the 4-token gap this class exists
        # to close — so it must never be inferred from a loosely typed value. verl's own
        # sft_trainer_engine.yaml ships `enable_thinking_default: none`, which YAML parses as
        # the *string* "none", not None. A plain `config.get(..., False)` therefore returned a
        # truthy string in real training runs, the template took its thinking branch, and the
        # prompt silently lost the think block again while the pre-flight gate (which builds
        # its own config without this key) kept passing. Only a real boolean enables thinking.
        raw_thinking = config.get("enable_thinking_default", False)
        if isinstance(raw_thinking, bool):
            self.enable_thinking = raw_thinking
        elif isinstance(raw_thinking, str) and raw_thinking.strip().lower() in ("true", "false"):
            self.enable_thinking = raw_thinking.strip().lower() == "true"
        else:
            self.enable_thinking = False
            logger.warning(
                "enable_thinking_default=%r is not a boolean; using enable_thinking=False "
                "(no-thinking, matching the folded data and thinking_mode: NO at inference).",
                raw_thinking,
            )
        logger.info("FoldedSFTDataset: enable_thinking=%s, pad_mode=%s, truncation=%s",
                    self.enable_thinking, self.pad_mode, self.truncation)
        if self.truncation not in ("error", "left", "right"):
            raise ValueError(f"Unknown truncation method {self.truncation}")
        if self.pad_mode not in ("right", "no_padding"):
            raise ValueError(f"Unknown pad mode {self.pad_mode}")

        if not isinstance(parquet_files, (list, tuple)):
            parquet_files = [parquet_files]
        frames = [pd.read_parquet(p) for p in parquet_files]
        self.dataframe = pd.concat(frames).reset_index(drop=True)
        if max_samples is not None and 0 < max_samples < len(self.dataframe):
            self.dataframe = self.dataframe.iloc[:max_samples]

        self.messages = [list(m) for m in self.dataframe[self.messages_key].tolist()]
        for i, msgs in enumerate(self.messages):
            roles = [m["role"] for m in msgs]
            if roles != ["system", "user", "assistant"]:
                raise ValueError(
                    f"row {i}: FoldedSFTDataset expects [system, user, assistant] rows, got {roles}. "
                    "Build the parquet with `build_sft_parquet.py --format folded`."
                )

    def __len__(self):
        return len(self.messages)

    def _eos_id(self):
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError("tokenizer has no eos_token_id; cannot terminate the target")
        return eos

    def __getitem__(self, item):
        messages = [dict(m) for m in self.messages[item]]

        # Byte-identical to VLLMProvider._render_messages for Qwen3 (vllm_provider.py:338-343).
        prompt_ids = self.tokenizer.apply_chat_template(
            messages[:2],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        response_ids = self.tokenizer(
            messages[2]["content"], add_special_tokens=False
        )["input_ids"] + [self._eos_id()]

        input_ids = torch.tensor(list(prompt_ids) + list(response_ids), dtype=torch.long)
        loss_mask = torch.zeros_like(input_ids)
        loss_mask[len(prompt_ids):] = 1
        attention_mask = torch.ones_like(input_ids)

        sequence_length = input_ids.shape[0]
        if sequence_length > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            else:  # right — drops the tail of the target, never the prompt
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            sequence_length = input_ids.shape[0]

        if self.pad_mode == "right" and sequence_length < self.max_length:
            pad_id = self.tokenizer.pad_token_id
            pad_id = pad_id if pad_id is not None else 0
            pad_len = self.max_length - sequence_length
            input_ids = torch.cat((input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype)))
            loss_mask = torch.cat((loss_mask, torch.zeros(pad_len, dtype=loss_mask.dtype)))
            attention_mask = torch.cat((attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)))

        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)

        res = {"input_ids": input_ids, "position_ids": position_ids, "loss_mask": loss_mask}
        if self.pad_mode == "right":
            res["attention_mask"] = attention_mask
        return res
