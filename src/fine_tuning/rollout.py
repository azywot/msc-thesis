"""OrchestratorRollout — AgentFlow LitAgent wrapping AgenticOrchestrator.

Runs the full msc-thesis orchestration loop as a VERL rollout worker.
Connects to the VERL-served vLLM endpoint for model generation and uses
WebSearchTool in direct mode for web search during training.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from filelock import FileLock

from agentflow import LitAgent, reward
from agentflow.types import NamedResources

from agent_engine.core.orchestrator import AgenticOrchestrator
from agent_engine.core.tool import ToolRegistry
from agent_engine.models.api_provider import OpenAIProvider
from agent_engine.models.base import ModelConfig, ModelFamily
from agent_engine.prompts.builder import PromptBuilder
from agent_engine.tools import WebSearchTool

from .reward import OrchestratorReward

_reward_fn_instance = OrchestratorReward()


@reward
async def _reward_fn(
    question: str,
    ground_truth: str,
    prediction: str,
    data_source: str,
    val: bool = False,
) -> float:
    """AgentFlow reward function — registered with VERL via @reward decorator."""
    return _reward_fn_instance(prediction, ground_truth, data_source)


def _build_tool_registry() -> ToolRegistry:
    """Build a minimal ToolRegistry for rollout workers.

    Uses WebSearchTool in direct mode (model_provider=None). Requires
    SERPER_API_KEY or TAVILY_API_KEY in the environment.
    """
    api_key = os.environ.get("SERPER_API_KEY") or os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "SERPER_API_KEY (or TAVILY_API_KEY) must be set for rollout workers."
        )
    provider = (
        "tavily"
        if os.environ.get("TAVILY_API_KEY") and not os.environ.get("SERPER_API_KEY")
        else "serper"
    )
    registry = ToolRegistry()
    registry.register(WebSearchTool(api_key=api_key, provider=provider, top_k=5))
    return registry


def _make_model_config(model_id: str, temperature: float, max_tokens: int = 2048) -> ModelConfig:
    """Build a ModelConfig for the VERL-served model (OpenAI-compatible API)."""
    return ModelConfig(
        name="verl_orchestrator",
        family=ModelFamily.QWEN3,
        path_or_id=model_id,
        role="orchestrator",
        temperature=temperature,
        max_tokens=max_tokens,
        backend="openai",
    )


class OrchestratorRollout(LitAgent):
    """AgentFlow LitAgent that runs AgenticOrchestrator as the rollout agent.

    Mirrors AgentFlow's Rollout class in train/rollout.py, replacing
    construct_solver() with AgenticOrchestrator.
    """

    def __init__(
        self,
        rollout_dir: str = "./rollout_data",
        rollout_n: int = 8,
        train_temperature: float = 0.7,
        test_temperature: float = 0.0,
        max_turns: int = 5,
        max_tokens: int = 2048,
    ):
        super().__init__()
        self.rollout_dir = Path(rollout_dir)
        self.rollout_n = rollout_n
        self.train_temperature = train_temperature
        self.test_temperature = test_temperature
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self._prompt_builder = PromptBuilder()
        self._tool_registry: Optional[ToolRegistry] = None

    # ------------------------------------------------------------------ #
    # LitAgent interface                                                   #
    # ------------------------------------------------------------------ #

    async def training_rollout_async(
        self, task: Any, rollout_id: str, resources: NamedResources, val: bool = False
    ) -> None:
        temperature = self.test_temperature if val else self.train_temperature
        endpoint = resources.get("main_llm").endpoint
        await self._run_episode(task, rollout_id, temperature, endpoint=endpoint, val=val)

    async def validation_rollout_async(
        self, task: Any, rollout_id: str, resources: NamedResources
    ) -> None:
        endpoint = resources.get("main_llm").endpoint
        await self._run_episode(
            task, rollout_id, self.test_temperature, endpoint=endpoint, val=True
        )

    # ------------------------------------------------------------------ #
    # Core episode logic                                                   #
    # ------------------------------------------------------------------ #

    async def _run_episode(
        self,
        task: Any,
        rollout_id: str,
        temperature: float,
        endpoint: str,
        val: bool,
    ) -> None:
        question_text = task.get("question", "")
        ground_truth = str(task.get("result", ""))
        extra = task.get("extra_info", {}) or {}
        data_source = str(extra.get("data_source", "nq"))
        idx = extra.get("idx", 0)

        # Append output format instruction (mirrors AgentFlow rollout)
        output_fmt = (
            " When ready, output the final answer enclosed in "
            "<answer> and </answer> tags."
        )
        prompt = question_text + output_fmt

        answer = "None"
        output_messages: list = []
        try:
            provider = self._build_provider(endpoint, temperature)
            tool_registry = self._get_or_build_tools()
            system_prompt = self._prompt_builder.build_system_prompt(
                dataset_name="gaia",
                tool_schemas=tool_registry.get_all_schemas(),
                direct_tool_call=True,
            )
            orchestrator = AgenticOrchestrator(
                model_provider=provider,
                tool_registry=tool_registry,
                max_turns=self.max_turns,
            )
            state = orchestrator.run(
                question=prompt,
                question_id=idx,
                system_prompt=system_prompt,
            )
            output_messages = state.output_messages
            answer = state.answer or "None"
        except Exception as exc:
            print(f"[OrchestratorRollout] Episode failed: {exc}")
            answer = "None"

        # Register reward with VERL via @reward decorator
        reward_value = await _reward_fn(
            question_text, ground_truth, answer, data_source, val
        )
        print(f"answer={answer!r}  gt={ground_truth!r}  reward={reward_value}")

        # Save rollout data to disk for debugging (mirrors AgentFlow)
        self._save_rollout(
            idx=idx,
            rollout_id=rollout_id,
            question=question_text,
            ground_truth=ground_truth,
            answer=answer,
            reward=reward_value,
            output_messages=output_messages,
            val=val,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _build_provider(self, endpoint: str, temperature: float) -> OpenAIProvider:
        """Build an OpenAIProvider pointing at the VERL vLLM endpoint."""
        config = _make_model_config(
            model_id=os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B"),
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        return OpenAIProvider(config, api_key="EMPTY", base_url=endpoint)

    def _get_or_build_tools(self) -> ToolRegistry:
        if self._tool_registry is None:
            self._tool_registry = _build_tool_registry()
        return self._tool_registry

    # ------------------------------------------------------------------ #
    # Rollout persistence                                                  #
    # ------------------------------------------------------------------ #

    def _save_rollout(
        self,
        *,
        idx: int,
        rollout_id: str,
        question: str,
        ground_truth: str,
        answer: str,
        reward: float,
        output_messages: list,
        val: bool,
    ) -> None:
        split = "val" if val else "train"
        save_dir = self.rollout_dir / split / f"idx_{idx}"
        save_dir.mkdir(parents=True, exist_ok=True)

        lock_path = self.rollout_dir / f".{split}.lock"
        with FileLock(str(lock_path), timeout=30):
            existing = sum(1 for _ in save_dir.glob("rollout_*.json"))
            assert existing < self.rollout_n, (
                f"Too many rollouts for idx {idx}: {existing} >= {self.rollout_n}"
            )

        record = {
            "idx": idx,
            "rollout_id": rollout_id,
            "question": question,
            "groundtruth": ground_truth,
            "answer_extracted": answer,
            "reward": reward,
            "output_messages": output_messages,
            "timestamp": datetime.now().isoformat(),
        }
        out = save_dir / f"rollout_{uuid.uuid4().hex[:8]}.json"
        with open(out, "w") as f:
            json.dump(record, f, indent=2, default=str)
