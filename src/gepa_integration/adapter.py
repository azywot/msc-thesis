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
from agent_engine.datasets.evaluators.metrics import evaluate_answer
from agent_engine.models.base import BaseModelProvider
from gepa.core.adapter import EvaluationBatch


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
            # Store ground truth for make_reflective_dataset
            state.metadata["ground_truth"] = example.answer
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

    _MAX_RECORDS = 12          # 6 correct + 6 wrong per reflective call
    _RESULT_SNIPPET_LEN = 300  # chars per tool result

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
            gt = state.metadata.get("ground_truth", "")
            first_thinking = (
                _extract_thinking(state.output_messages[0]["content"])
                if state.output_messages
                else ""
            )
            action_steps = [
                {
                    "tool": a["tool_name"],
                    "sub_goal": a.get("sub_goal", ""),
                    "result_snippet": str(a.get("result", ""))[: self._RESULT_SNIPPET_LEN],
                }
                for a in state.action_history
            ]
            if score > 0:
                feedback = "CORRECT"
            else:
                parts = [f"WRONG — ground truth: {gt}. Predicted: {state.answer or '(empty)'}."]
                if state.metadata.get("max_turns_reached"):
                    parts.append("Max turns reached without answer.")
                feedback = " ".join(parts)

            records.append({
                "Inputs": {"question": state.question},
                "Generated Outputs": {
                    "predicted_answer": state.answer or "",
                    "thinking_before_first_tool": first_thinking,
                    "action_steps": action_steps,
                },
                "Feedback": feedback,
            })
        return records

    def _planning_suffix_records(
        self, states: list[ExecutionState], scores: list[float]
    ) -> list[dict]:
        records = []
        for state, score in self._balanced_sample(states, scores):
            raw_plan = state.raw_query_analysis or state.query_analysis or ""
            tools_used = [tc["name"] for tc in state.tool_calls]
            if score > 0:
                feedback = "CORRECT — the planning analysis led to a successful solution."
            else:
                feedback = (
                    f"WRONG — the planning analysis was: '{state.query_analysis}'. "
                    "Consider whether the plan correctly identified the required steps and tools."
                )
            records.append({
                "Inputs": {"question": state.question},
                "Generated Outputs": {
                    "raw_planning_output": raw_plan,
                    "tools_subsequently_used": tools_used,
                    "num_turns_taken": state.turn,
                },
                "Feedback": feedback,
            })
        return records
