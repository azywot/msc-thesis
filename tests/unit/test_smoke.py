"""Smoke tests — verify the critical pipeline path runs end-to-end.

These tests use a minimal mock model (no GPU / API keys required) and a
tiny in-memory config.  They are intentionally small: if they pass, the
wiring between config → orchestrator → evaluation is intact.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest

pytest.importorskip("math_verify", reason="math_verify required for metrics")

from agent_engine.config.loader import load_experiment_config
from agent_engine.config.schema import ExperimentConfig, DatasetConfig, ToolsConfig
from agent_engine.core.orchestrator import AgenticOrchestrator
from agent_engine.core.tool import ToolRegistry
from agent_engine.models.base import BaseModelProvider, GenerationResult, ModelConfig, ModelFamily
from agent_engine.datasets.evaluators.metrics import evaluate_answer
from agent_engine.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Minimal mock model
# ---------------------------------------------------------------------------

class _MockProvider(BaseModelProvider):
    """Returns a fixed answer string; never calls any real inference."""

    def __init__(self, answer: str = "42"):
        config = ModelConfig(
            name="mock",
            family=ModelFamily.QWEN3,
            path_or_id="mock",
            role="orchestrator",
            seed=7,
        )
        super().__init__(config)
        self._answer = answer

    def generate(self, prompts: List[str]) -> List[GenerationResult]:
        return [
            GenerationResult(
                text=f"Final Answer: {self._answer}",
                finish_reason="stop",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
            for _ in prompts
        ]

    def apply_chat_template(self, messages, use_thinking=False, force_tool_call=False) -> str:
        return json.dumps({"messages": messages}, ensure_ascii=False)

    def cleanup(self):
        pass


# ---------------------------------------------------------------------------
# Smoke: seed propagation
# ---------------------------------------------------------------------------

class TestSeedPropagation:
    def test_experiment_seed_propagates_to_model_with_no_explicit_seed(self, tmp_path):
        """Seed in YAML must reach ModelConfig after loading."""
        yaml_text = """
name: smoke
seed: 99
models:
  orchestrator:
    name: mock
    family: qwen3
    path_or_id: mock
    role: orchestrator
dataset:
  name: gaia
  split: validation
"""
        cfg_path = tmp_path / "smoke.yaml"
        cfg_path.write_text(yaml_text)
        config = load_experiment_config(cfg_path)
        assert config.seed == 99
        assert config.models["orchestrator"].seed == 99

    def test_explicit_model_seed_is_not_overwritten(self, tmp_path):
        yaml_text = """
name: smoke
seed: 99
models:
  orchestrator:
    name: mock
    family: qwen3
    path_or_id: mock
    role: orchestrator
    seed: 7
dataset:
  name: gaia
  split: validation
"""
        cfg_path = tmp_path / "smoke.yaml"
        cfg_path.write_text(yaml_text)
        config = load_experiment_config(cfg_path)
        assert config.models["orchestrator"].seed == 7

    def test_set_seed_is_callable(self):
        set_seed(42)  # must not raise


# ---------------------------------------------------------------------------
# Smoke: orchestrator runs and extracts answer
# ---------------------------------------------------------------------------

class TestOrchestratorSmoke:
    def _run(self, answer: str = "42"):
        model = _MockProvider(answer)
        orchestrator = AgenticOrchestrator(
            model_provider=model,
            tool_registry=ToolRegistry(),
            max_turns=3,
            baseline=True,  # skip planning turn → simpler path
        )
        state = orchestrator.run(
            question="What is 6 times 7?",
            question_id=0,
            system_prompt="You are a helpful assistant.",
        )
        return state

    def test_orchestrator_finishes(self):
        state = self._run()
        assert state.finished is True

    def test_answer_is_extracted(self):
        state = self._run("smoke-answer")
        assert state.answer == "smoke-answer"

    def test_turn_counter_incremented(self):
        state = self._run()
        assert state.turn >= 1

    def test_output_messages_populated(self):
        state = self._run()
        assert len(state.output_messages) >= 1

    def test_batch_matches_single(self):
        """run_batch with one question must produce same answer as run."""
        model = _MockProvider("batch-ok")
        orch = AgenticOrchestrator(
            model_provider=model,
            tool_registry=ToolRegistry(),
            max_turns=3,
            baseline=True,
        )
        states = orch.run_batch(
            questions=["Q?"],
            question_ids=[0],
            system_prompts=["sys"],
        )
        assert len(states) == 1
        assert states[0].answer == "batch-ok"


# ---------------------------------------------------------------------------
# Smoke: evaluation round-trip
# ---------------------------------------------------------------------------

class TestEvaluationSmoke:
    def test_correct_plain_text(self):
        result = evaluate_answer("Paris", "Paris")
        assert result["correct"] is True
        assert result["accuracy"] == 1.0

    def test_wrong_plain_text(self):
        result = evaluate_answer("Berlin", "Paris")
        assert result["correct"] is False

    def test_result_has_all_keys(self):
        result = evaluate_answer("42", "42")
        assert {"correct", "accuracy", "em", "f1"} <= result.keys()

    def test_math_answer(self):
        result = evaluate_answer("42", "42")
        assert result["correct"] is True

    def test_containment_scores_correct(self):
        # ground truth contained in a longer prediction
        result = evaluate_answer("The answer is Paris, France", "Paris")
        assert result["correct"] is True
