"""GEPA adapter for the AgenticOrchestrator.

AgentGEPAAdapter connects GEPA's optimization loop to the msc-thesis
orchestrator. It implements the GEPAAdapter protocol with two optimizable
components: "system_prompt" and "planning_suffix".

Thinking mode is fixed at ORCHESTRATOR_ONLY to match the main experimental
condition and provide rich <think> traces for the reflector.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from agent_engine.core.orchestrator import AgenticOrchestrator
from agent_engine.core.state import ExecutionState
from agent_engine.core.tool import ToolRegistry
from agent_engine.datasets.base import DatasetExample
from agent_engine.datasets.evaluators.metrics import (
    evaluate_answer,
    is_math_answer,
    normalize_answer,
)
from agent_engine.models.base import BaseModelProvider
from gepa.core.adapter import EvaluationBatch

# Tokens that, when they appear at the start of a tool result, indicate the
# call failed. Detection is intentionally a lightweight string heuristic — the
# orchestrator surfaces ToolResult.error inline in the result string, so the
# prefix check is enough to flag failed retrievals / code executions for the
# reflector without changing the orchestrator's data shape.
_TOOL_ERROR_PREFIXES = ("error", "tool error", "exception", "traceback", "failed")


def _extract_thinking(text: str) -> str:
    """Return the content of the first <think>…</think> block, or ''."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


class AgentGEPAAdapter:
    """GEPAAdapter implementation wrapping AgenticOrchestrator.

    Args:
        model_provider: Shared vLLM provider (not re-loaded between candidates).
        tool_registry:  Pre-built tool registry.
        use_thinking:   Whether to enable orchestrator thinking (default True —
                        ORCHESTRATOR_ONLY mode).
        max_turns:      Maximum reasoning turns per question.
        tool_limits:    Per-tool call limits dict.
    """

    # GEPA's reflective_mutation accesses adapter.propose_new_texts directly
    # (no getattr/hasattr); setting it to None tells GEPA to use reflection_lm.
    propose_new_texts = None

    def __init__(
        self,
        model_provider: BaseModelProvider,
        tool_registry: ToolRegistry,
        use_thinking: bool = True,
        max_turns: int = 15,
        tool_limits: Optional[dict[str, int]] = None,
    ) -> None:
        self.model_provider = model_provider
        self.tool_registry = tool_registry
        self.use_thinking = use_thinking
        self.max_turns = max_turns
        self.tool_limits = tool_limits or {"web_search": 10}

    # ------------------------------------------------------------------ #
    # GEPAAdapter protocol                                                 #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        batch: list[DatasetExample],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Run the orchestrator on `batch` using `candidate`'s prompts.

        Stores ground_truth in each state's metadata so make_reflective_dataset
        can access it without needing the original examples.
        """
        orchestrator = AgenticOrchestrator(
            model_provider=self.model_provider,
            tool_registry=self.tool_registry,
            max_turns=self.max_turns,
            tool_limits=self.tool_limits,
            use_thinking=self.use_thinking,
            planning_suffix=candidate["planning_suffix"],
        )

        states: list[ExecutionState] = orchestrator.run_batch(
            questions=[ex.question for ex in batch],
            question_ids=[ex.question_id for ex in batch],
            system_prompts=[candidate["system_prompt"]] * len(batch),
            attachments=[ex.get_attachments() or None for ex in batch],
        )

        outputs: list[str] = []
        scores: list[float] = []
        trajectories: list[ExecutionState] | None = [] if capture_traces else None

        for state, example in zip(states, batch):
            prediction = state.answer or ""
            choices = example.metadata.get("choices")
            result = evaluate_answer(prediction, example.answer, choices=choices)
            outputs.append(prediction)
            scores.append(float(result["accuracy"]))
            # Stash everything make_reflective_dataset needs so it can build
            # rich, deterministic feedback without re-running the scorer.
            state.metadata["ground_truth"] = example.answer
            state.metadata["eval_result"] = result
            state.metadata["choices"] = choices
            if capture_traces:
                trajectories.append(state)  # type: ignore[union-attr]

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Build per-component reflective datasets from execution traces.

        Returns at most 12 records per component (6 correct, 6 wrong).
        """
        states: list[ExecutionState] = eval_batch.trajectories or []
        scores: list[float] = eval_batch.scores

        dataset: dict[str, list[dict]] = {}

        if "system_prompt" in components_to_update:
            dataset["system_prompt"] = self._system_prompt_records(states, scores)

        if "planning_suffix" in components_to_update:
            dataset["planning_suffix"] = self._planning_suffix_records(states, scores)

        return dataset

    # ------------------------------------------------------------------ #
    # Reflective dataset helpers                                           #
    # ------------------------------------------------------------------ #

    # Reflector prompt budget: instruction template + system_prompt + records.
    # With 12 verbose records the reflector regularly exceeded 32 K tokens on
    # thinking traces. 8 records + a hard cap on thinking length keeps the
    # reflective prompt comfortably under the reflector's max_model_len.
    _MAX_RECORDS = 8           # 4 correct + 4 wrong per reflective call
    _RESULT_SNIPPET_LEN = 300  # chars per tool result
    _THINKING_SNIPPET_LEN = 1500  # chars per <think> trace (truncated)

    # Heuristic threshold: if the prediction is more than this multiple of the
    # gold answer's word count, flag it as "verbose for a short answer". GAIA
    # gold answers are typically 1–3 tokens, so 4× catches paragraph-style
    # over-explanations without firing on borderline cases.
    _VERBOSITY_RATIO = 4

    def _diagnose(self, state: ExecutionState, score: float) -> str:
        """Build the Feedback string shown to the reflector.

        Feedback is fully deterministic (no extra LLM call): the score comes
        from ``evaluate_answer`` (stashed in ``state.metadata['eval_result']``
        during :meth:`evaluate`), and every other signal is read off the
        ExecutionState the orchestrator already produced. This matches GEPA's
        paper notion of an environment-derived feedback function (μ_f) for QA
        benchmarks, where the "environment" is the scorer plus the tool stack.

        For correct cases the feedback also includes tool usage and turn count
        so the reflector can learn which trajectories worked. For wrong cases
        it includes a per-failure-mode hint (format mismatch, tool errors,
        parametric-memory hallucination, max-turns thrash) that the reflector
        can credit-assign to the system_prompt or planning_suffix.
        """
        pred = state.answer or ""
        gt = state.metadata.get("ground_truth", "") or ""
        eval_result = state.metadata.get("eval_result") or {}
        used = {k: v for k, v in state.tool_counts.items() if v > 0}
        is_mc = state.metadata.get("choices") is not None

        if score > 0:
            return (
                f"CORRECT. Predicted {pred!r}; "
                f"tools={used or 'none'}; turns={state.turn}."
            )

        pred_display = repr(pred) if pred else "(empty)"
        parts = [
            f"WRONG. Ground truth: {gt!r}. Predicted: {pred_display}.",
            f"  Scoring: em={eval_result.get('em', 0.0):.2f}, "
            f"f1={eval_result.get('f1', 0.0):.2f}.",
        ]

        n_gt = normalize_answer(gt)
        n_pred = normalize_answer(pred)
        if n_gt and n_pred and n_gt != n_pred:
            parts.append(f"  Normalised: {n_gt!r} vs {n_pred!r}.")

        # Failure-mode hints (skip format checks for multiple-choice — the
        # answer shape there is one letter and is_math_answer/verbosity would
        # both misfire).
        if not pred:
            parts.append("  No final answer was produced.")
        elif not is_mc:
            if is_math_answer(gt) and not is_math_answer(pred):
                parts.append(
                    "  Format mismatch: gold is numeric/symbolic; "
                    "prediction is prose."
                )
            if gt and len(pred.split()) > self._VERBOSITY_RATIO * max(
                1, len(gt.split())
            ):
                parts.append(
                    "  Prediction is much longer than the gold answer — "
                    "likely not in the expected short form."
                )
            if eval_result.get("f1", 0.0) >= 0.5:
                parts.append(
                    "  Token overlap is high (f1 ≥ 0.5) — likely a "
                    "formatting/precision error, not a content error."
                )

        if used:
            parts.append(f"  Tools used: {used}.")
        else:
            parts.append(
                "  No tools called — the model answered from parametric "
                "memory only."
            )

        errors = sum(
            1
            for a in state.action_history
            if str(a.get("result", ""))[:120]
            .lower()
            .lstrip()
            .startswith(_TOOL_ERROR_PREFIXES)
        )
        if errors:
            parts.append(
                f"  {errors}/{len(state.action_history)} tool calls "
                "returned an error."
            )

        if state.metadata.get("max_turns_reached"):
            parts.append(
                f"  Max turns ({self.max_turns}) reached without a final "
                "answer."
            )

        return "\n".join(parts)

    def _balanced_sample(
        self, states: list[ExecutionState], scores: list[float]
    ) -> list[tuple[ExecutionState, float]]:
        """Return up to MAX_RECORDS pairs balanced between correct and wrong."""
        correct = [(s, sc) for s, sc in zip(states, scores) if sc > 0]
        wrong = [(s, sc) for s, sc in zip(states, scores) if sc == 0]
        half = self._MAX_RECORDS // 2
        return correct[:half] + wrong[:half]

    def _system_prompt_records(
        self, states: list[ExecutionState], scores: list[float]
    ) -> list[dict]:
        records = []
        for state, score in self._balanced_sample(states, scores):
            first_thinking = (
                _extract_thinking(state.output_messages[0]["content"])
                if state.output_messages
                else ""
            )
            if len(first_thinking) > self._THINKING_SNIPPET_LEN:
                first_thinking = first_thinking[: self._THINKING_SNIPPET_LEN] + "…[truncated]"
            action_steps = [
                {
                    "tool": a["tool_name"],
                    "sub_goal": a.get("sub_goal", ""),
                    "result_snippet": str(a.get("result", ""))[: self._RESULT_SNIPPET_LEN],
                }
                for a in state.action_history
            ]
            records.append({
                "Inputs": {"question": state.question},
                "Generated Outputs": {
                    "predicted_answer": state.answer or "",
                    "thinking_before_first_tool": first_thinking,
                    "action_steps": action_steps,
                },
                "Feedback": self._diagnose(state, score),
            })
        return records

    def _planning_suffix_records(
        self, states: list[ExecutionState], scores: list[float]
    ) -> list[dict]:
        records = []
        for state, score in self._balanced_sample(states, scores):
            raw_plan = state.raw_query_analysis or state.query_analysis or ""
            tools_used = [tc["name"] for tc in state.tool_calls]
            records.append({
                "Inputs": {"question": state.question},
                "Generated Outputs": {
                    "raw_planning_output": raw_plan,
                    "tools_subsequently_used": tools_used,
                    "num_turns_taken": state.turn,
                },
                "Feedback": self._diagnose(state, score),
            })
        return records
