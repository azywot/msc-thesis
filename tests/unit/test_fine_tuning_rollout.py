import sys
import types
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# Keep these unit tests independent of the AgentFlow training environment.
agentflow = types.ModuleType("agentflow")


class _LitAgent:
    pass


def _reward(fn):
    return fn


agentflow.LitAgent = _LitAgent
agentflow.reward = _reward

agentflow_types = types.ModuleType("agentflow.types")
agentflow_types.NamedResources = dict

sys.modules.setdefault("agentflow", agentflow)
sys.modules.setdefault("agentflow.types", agentflow_types)


from fine_tuning.rollout import (  # noqa: E402
    _build_rollout_question,
    _get_task_metadata,
    _prompt_dataset_for_data_source,
)


def test_task_metadata_prefers_top_level_data_source():
    task = {
        "question": "Compute 2+2",
        "result": "4",
        "data_source": "deepmath",
        "extra_info": {"idx": 7, "data_source": "nq"},
    }

    question, ground_truth, data_source, idx = _get_task_metadata(task)

    assert question == "Compute 2+2"
    assert ground_truth == "4"
    assert data_source == "deepmath"
    assert idx == 7


def test_task_metadata_falls_back_to_extra_info_data_source():
    task = {
        "question": "Who wrote Hamlet?",
        "result": "Shakespeare",
        "extra_info": {"idx": 3, "data_source": "hotpotqa"},
    }

    _, _, data_source, idx = _get_task_metadata(task)

    assert data_source == "hotpotqa"
    assert idx == 3


def test_prompt_dataset_uses_math_family_for_deepmath_only():
    assert _prompt_dataset_for_data_source("deepmath") == "deepmath"
    assert _prompt_dataset_for_data_source("nq") == "gaia"
    assert _prompt_dataset_for_data_source("hotpotqa") == "gaia"


def test_rollout_question_is_not_modified_with_agentflow_answer_suffix():
    question = "Find the value of x."

    rollout_question = _build_rollout_question(question)

    assert rollout_question == question
    assert "<answer>" not in rollout_question
