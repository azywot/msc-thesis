"""OrchestratorRollout — AgentFlow LitAgent wrapping AgenticOrchestrator.

Runs the full msc-thesis orchestration loop as a VERL rollout worker.
Connects to the VERL-served vLLM endpoint for model generation and routes
tool use through the same sub-agent interfaces used by thesis inference.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fine_tuning.agentflow import LitAgent, reward
from fine_tuning.agentflow.types import NamedResources, Rollout, Triplet

from agent_engine.core.orchestrator import AgenticOrchestrator
from agent_engine.core.tool import ToolRegistry
from agent_engine.models.api_provider import OpenAIProvider
from agent_engine.models.base import ModelConfig, ModelFamily
from agent_engine.prompts.builder import PromptBuilder
from agent_engine.tools import WebSearchTool, CodeGeneratorTool

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


def _build_tool_registry(subagent_endpoint: str, subagent_model: str) -> ToolRegistry:
    """Build the ToolRegistry for rollout workers using a frozen sub-agent server.

    Sub-agents connect to a SEPARATE, fixed vLLM endpoint (not the VERL training
    endpoint).  This keeps sub-agent weights frozen throughout training so the
    orchestrator is evaluated against a stable tool interface at every rollout —
    matching eval-time behaviour where sub-agents run the base model, not an
    evolving snapshot.

    - WebSearchTool:     query → API results → sub-agent LLM analyses them
    - CodeGeneratorTool: task description → sub-agent LLM writes code → exec

    Requires:
      - SERPER_API_KEY or TAVILY_API_KEY in the environment.
      - subagent_endpoint: URL of a separately-launched frozen vLLM server
        (e.g. "http://localhost:9998/v1").
    """
    if not subagent_endpoint:
        raise EnvironmentError(
            "SUBAGENT_ENDPOINT must be set to the URL of a frozen sub-agent vLLM "
            "server (e.g. http://localhost:9998/v1).  Start it separately with:\n"
            "  vllm serve <SUBAGENT_MODEL> --port 9998 --tensor-parallel-size 1 \\\n"
            "    --gpu-memory-utilization 0.15 --max-model-len 8192"
        )
    api_key = os.environ.get("SERPER_API_KEY") or os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "SERPER_API_KEY (or TAVILY_API_KEY) must be set for rollout workers."
        )
    search_provider = (
        "tavily"
        if os.environ.get("TAVILY_API_KEY") and not os.environ.get("SERPER_API_KEY")
        else "serper"
    )
    # Sub-agent provider: frozen separate endpoint, greedy decoding, thinking off.
    subagent_config = _make_model_config(
        model_id=subagent_model,
        temperature=0.0,
        max_tokens=2048,
    )
    subagent_provider = OpenAIProvider(subagent_config, api_key="EMPTY", base_url=subagent_endpoint)

    registry = ToolRegistry()
    max_search_content_chars = int(os.environ.get("MAX_SEARCH_CONTENT_CHARS", 14000))
    registry.register(WebSearchTool(
        api_key=api_key,
        provider=search_provider,
        top_k=5,
        model_provider=subagent_provider,
        max_search_content_chars=max_search_content_chars,
    ))
    registry.register(CodeGeneratorTool(model_provider=subagent_provider))
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


def _get_task_metadata(task: Any) -> tuple[str, str, str, int]:
    """Extract rollout fields from VERL task rows.

    The parquet schema stores ``data_source`` at the top level. Some older
    records may carry it in ``extra_info``, so keep that fallback for smoke
    fixtures and generated data while preferring the shipped schema.
    """
    question_text = str(task.get("question", ""))
    ground_truth = str(task.get("result", ""))
    extra = task.get("extra_info", {}) or {}
    if not isinstance(extra, dict):
        extra = {}
    data_source = str(task.get("data_source") or extra.get("data_source") or "nq").lower()
    idx = int(extra.get("idx", 0))
    return question_text, ground_truth, data_source, idx


def _prompt_dataset_for_data_source(data_source: str) -> str:
    """Map training data source to the closest inference prompt family."""
    return "deepmath" if data_source.lower() == "deepmath" else "gaia"


def _build_rollout_question(question_text: str) -> str:
    """Return the exact user question passed to the thesis orchestrator.

    Do not append AgentFlow's ``<answer>`` suffix here: thesis inference relies
    on dataset system prompts for final-answer formatting, so RL should see the
    same user-question surface.
    """
    return question_text


class _CapturingProvider:
    """Wraps an OpenAIProvider to record (prompt_ids, response_ids) per generate() call.

    Delegates all attribute access to the underlying provider so the orchestrator
    can use it as a drop-in replacement. The token IDs are available in
    `captured_turns` after the episode finishes, one entry per LLM call.
    """

    def __init__(self, provider: OpenAIProvider) -> None:
        object.__setattr__(self, "_provider", provider)
        object.__setattr__(self, "captured_turns", [])

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_provider"), name)

    def generate(self, prompts: list) -> list:
        results = object.__getattribute__(self, "_provider").generate(prompts)
        turns = object.__getattribute__(self, "captured_turns")
        for result in results:
            turns.append({
                "prompt_ids": list(result.prompt_token_ids) if result.prompt_token_ids else [],
                "response_ids": list(result.response_token_ids) if result.response_token_ids else [],
                "response_text": result.text or "",
            })
        return results


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
        use_thinking: bool = False,
        subagent_endpoint: str = "",
        subagent_model: str = "Qwen/Qwen3-1.7B",
    ):
        super().__init__()
        self.rollout_dir = Path(rollout_dir)
        self.rollout_n = rollout_n
        self.train_temperature = train_temperature
        self.test_temperature = test_temperature
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.use_thinking = use_thinking
        self.subagent_endpoint = subagent_endpoint
        self.subagent_model = subagent_model
        self._prompt_builder = PromptBuilder()
        self._tool_registry: Optional[ToolRegistry] = None

    # ------------------------------------------------------------------ #
    # LitAgent interface                                                   #
    # ------------------------------------------------------------------ #

    async def training_rollout_async(
        self, task: Any, rollout_id: str, resources: NamedResources, val: bool = False
    ) -> float:
        temperature = self.test_temperature if val else self.train_temperature
        endpoint = resources.get("main_llm").endpoint
        return await self._run_episode(task, rollout_id, temperature, endpoint=endpoint, val=val)

    async def validation_rollout_async(
        self, task: Any, rollout_id: str, resources: NamedResources
    ) -> float:
        endpoint = resources.get("main_llm").endpoint
        return await self._run_episode(
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
    ) -> Rollout:
        question_text, ground_truth, data_source, idx = _get_task_metadata(task)
        prompt = _build_rollout_question(question_text)

        answer = "None"
        output_messages: list = []
        capturing_provider: Optional[_CapturingProvider] = None
        try:
            base_provider = self._build_provider(endpoint, temperature)
            capturing_provider = _CapturingProvider(base_provider)
            tool_registry = self._get_or_build_tools()
            system_prompt = self._prompt_builder.build_system_prompt(
                dataset_name=_prompt_dataset_for_data_source(data_source),
                tool_schemas=tool_registry.get_all_schemas(),
                direct_tool_call=False,
            )
            orchestrator = AgenticOrchestrator(
                model_provider=capturing_provider,
                tool_registry=tool_registry,
                max_turns=self.max_turns,
                use_thinking=self.use_thinking,
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

        self._save_rollout(
            idx=idx,
            rollout_id=rollout_id,
            question=question_text,
            ground_truth=ground_truth,
            answer=answer,
            reward=reward_value,
            data_source=data_source,
            output_messages=output_messages,
            val=val,
        )

        # Build triplets from captured LLM turns (one per generate() call).
        # The VERL proxy injects prompt/response token IDs into vLLM responses;
        # _CapturingProvider records them here.  Assign reward to the last turn.
        triplets: Optional[list] = None
        if capturing_provider is not None:
            turns = capturing_provider.captured_turns
            if turns:
                triplets = [
                    Triplet(
                        prompt={"token_ids": t["prompt_ids"]},
                        response={"token_ids": t["response_ids"], "text": t.get("response_text", "")},
                        reward=reward_value if i == len(turns) - 1 else None,
                    )
                    for i, t in enumerate(turns)
                ]

        return Rollout(
            rollout_id=rollout_id,
            final_reward=reward_value,
            triplets=triplets,
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
            self._tool_registry = _build_tool_registry(
                self.subagent_endpoint, self.subagent_model
            )
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
        data_source: str,
        output_messages: list,
        val: bool,
    ) -> None:
        split = "val" if val else "train"
        unique_id = f"{data_source}_{idx}"
        save_dir = self.rollout_dir / split / f"idx_{unique_id}"
        save_dir.mkdir(parents=True, exist_ok=True)

        record = {
            "idx": unique_id,
            "rollout_id": rollout_id,
            "data_source": data_source,
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
