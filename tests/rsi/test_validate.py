import copy
import json

import pytest

from rsi.bundle import seed_bundle
from rsi.validate import normalize_example, validate_unit


@pytest.fixture
def seed_b(SCHEMAS):
    return seed_bundle("search", SCHEMAS)


def example_with_call(call, answer="It is \\boxed{x}."):
    return {
        "question": "Q?",
        "steps": [
            {
                "step": 1,
                "reasoning": "r",
                "sub_goal": "g",
                "tool_call": call,
                "tool_response": "resp",
            },
            {"step": "final", "answer": answer},
        ],
    }


def test_seed_parts_are_valid(seed_b, SCHEMAS):
    assert validate_unit("orchestrator", seed_b.orchestrator, seed_b, SCHEMAS) == []
    assert validate_unit("web_search", seed_b.subagents["web_search"], seed_b, SCHEMAS) == []
    assert (
        validate_unit("code_generator", seed_b.subagents["code_generator"], seed_b, SCHEMAS) == []
    )


def test_unknown_unit_rejected(seed_b, SCHEMAS):
    assert any("unknown unit" in e for e in validate_unit("planner", {}, seed_b, SCHEMAS))


def test_missing_placeholder_rejected(seed_b, SCHEMAS):
    errs = validate_unit("web_search", "Analyse {query}.", seed_b, SCHEMAS)
    assert any("formatted_results" in e for e in errs)


def test_unknown_placeholder_rejected(seed_b, SCHEMAS):
    errs = validate_unit("code_generator", "{context_block}{task}{language}", seed_b, SCHEMAS)
    assert any("language" in e for e in errs)


def test_example_with_code_arg_in_subagent_mode_rejected(seed_b, SCHEMAS):
    ex = example_with_call({"name": "code_generator", "arguments": {"code": "print(1)"}})
    assert any("code" in e for e in validate_unit("orchestrator", {"example": ex}, seed_b, SCHEMAS))


def test_example_calling_disabled_tool_rejected(seed_b, SCHEMAS):
    ex = example_with_call({"name": "text_inspector", "arguments": {"question": "x"}})
    errs = validate_unit("orchestrator", {"example": ex}, seed_b, SCHEMAS)
    assert any("text_inspector" in e for e in errs)


def test_example_without_boxed_answer_rejected(seed_b, SCHEMAS):
    ex = example_with_call({"name": "web_search", "arguments": {"query": "x"}}, answer="It is x.")
    assert any(
        "boxed" in e for e in validate_unit("orchestrator", {"example": ex}, seed_b, SCHEMAS)
    )


def test_renamed_tool_parameter_rejected(seed_b, SCHEMAS):
    td = copy.deepcopy(seed_b.orchestrator["tool_descriptions"])
    td["web_search"]["parameters"] = {"q": "x"}
    assert validate_unit("orchestrator", {"tool_descriptions": td}, seed_b, SCHEMAS)


def test_normalize_example_accepts_object_tool_calls():
    ex = normalize_example(example_with_call({"name": "web_search", "arguments": {"query": "x"}}))
    assert ex["steps"][0]["tool_call"] == '{"name": "web_search", "arguments": {"query": "x"}}'


def test_normalize_example_renumbers_and_keeps_string_calls():
    raw = example_with_call('{"name": "web_search", "arguments": {"query": "x"}}')
    raw["steps"][0]["step"] = 7
    ex = normalize_example(raw)
    assert ex["steps"][0]["step"] == 1
    assert ex["steps"][-1]["step"] == "final"
    assert isinstance(ex["steps"][0]["tool_call"], str)


def test_string_tool_call_is_accepted(seed_b, SCHEMAS):
    ex = example_with_call(json.dumps({"name": "web_search", "arguments": {"query": "x"}}))
    assert validate_unit("orchestrator", {"example": ex}, seed_b, SCHEMAS) == []


def test_example_missing_required_argument_rejected(seed_b, SCHEMAS):
    ex = example_with_call({"name": "web_search", "arguments": {}})
    assert any(
        "query" in e for e in validate_unit("orchestrator", {"example": ex}, seed_b, SCHEMAS)
    )


def test_example_unparseable_tool_call_rejected(seed_b, SCHEMAS):
    ex = example_with_call("not json")
    assert validate_unit("orchestrator", {"example": ex}, seed_b, SCHEMAS)


def test_example_without_non_final_step_rejected(seed_b, SCHEMAS):
    ex = {"question": "Q?", "steps": [{"step": "final", "answer": "\\boxed{1}"}]}
    assert validate_unit("orchestrator", {"example": ex}, seed_b, SCHEMAS)


def test_example_last_step_not_final_rejected(seed_b, SCHEMAS):
    ex = example_with_call({"name": "web_search", "arguments": {"query": "x"}})
    ex["steps"].pop()
    assert validate_unit("orchestrator", {"example": ex}, seed_b, SCHEMAS)


def test_unknown_orchestrator_key_rejected(seed_b, SCHEMAS):
    assert validate_unit("orchestrator", {"bogus": "x"}, seed_b, SCHEMAS)


def test_empty_string_parts(seed_b, SCHEMAS):
    assert validate_unit("orchestrator", {"guidance_opening": " "}, seed_b, SCHEMAS)
    assert (
        validate_unit(
            "orchestrator", {"turn1n_prompt": "", "guidance_closing": ""}, seed_b, SCHEMAS
        )
        == []
    )


def test_tool_descriptions_missing_tool_rejected(seed_b, SCHEMAS):
    td = copy.deepcopy(seed_b.orchestrator["tool_descriptions"])
    del td["code_generator"]
    assert validate_unit("orchestrator", {"tool_descriptions": td}, seed_b, SCHEMAS)


def test_tool_descriptions_empty_string_rejected(seed_b, SCHEMAS):
    td = copy.deepcopy(seed_b.orchestrator["tool_descriptions"])
    td["web_search"]["description"] = ""
    assert validate_unit("orchestrator", {"tool_descriptions": td}, seed_b, SCHEMAS)


def test_subagent_for_disabled_tool_rejected(seed_b, SCHEMAS):
    only_web = [s for s in SCHEMAS if s["function"]["name"] == "web_search"]
    errs = validate_unit("code_generator", "{context_block}{task}", seed_b, only_web)
    assert any("not enabled" in e for e in errs)


def test_non_dict_orchestrator_payload_rejected(seed_b, SCHEMAS):
    assert validate_unit("orchestrator", "text", seed_b, SCHEMAS)


@pytest.mark.parametrize("bad", [{"steps": None}, "str", None, ["x"], {"question": "Q?"}])
def test_malformed_example_is_rejected_not_raised(bad, seed_b, SCHEMAS):
    ex = normalize_example(bad)
    assert validate_unit("orchestrator", {"example": ex}, seed_b, SCHEMAS)


@pytest.mark.parametrize("name", [["x"], {"a": 1}, 3, None])
def test_non_string_tool_name_is_rejected_not_raised(name, seed_b, SCHEMAS):
    ex = example_with_call({"name": name, "arguments": {}})
    assert validate_unit("orchestrator", {"example": normalize_example(ex)}, seed_b, SCHEMAS)


def test_normalize_example_skips_non_dict_steps():
    ex = normalize_example({"question": "Q", "steps": ["junk", {"step": 5}]})
    assert ex["steps"][1]["step"] == 1
