"""Run the AgentFlow team under a prompt bundle and return scored trajectories."""

from dataclasses import asdict, dataclass, field

from agent_engine.analysis.failure_modes import classify_failure
from agent_engine.core.orchestrator import AgenticOrchestrator
from agent_engine.datasets.evaluators.metrics import evaluate_answer
from rsi.bundle import (
    acting_suffix,
    apply_subagent_templates,
    planning_suffix,
    render_system_prompt,
)


@dataclass
class TrajectoryRecord:
    question_id: int
    question: str
    gold: str
    aliases: list = field(default_factory=list)
    prediction: str = ""
    correct: bool = False
    failure_label: str = None
    plan: str = ""
    action_history: list = field(default_factory=list)
    subagent_calls: list = field(default_factory=list)
    turns: int = 0
    tool_counts: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class BundleExecutor:
    def __init__(
        self,
        model_provider,
        tool_registry,
        tool_call_format,
        max_turns=7,
        tool_limits=None,
        use_thinking=False,
    ):
        self.model_provider = model_provider
        self.tool_registry = tool_registry
        self.tool_call_format = tool_call_format
        self.max_turns = max_turns
        self.tool_limits = tool_limits
        self.use_thinking = use_thinking
        self.rollouts = 0

    def run(self, bundle, examples):
        examples = list(examples)
        if not examples:
            return []
        apply_subagent_templates(bundle, self.tool_registry)
        # render_system_prompt applies the bundle's tool descriptions itself.
        system_prompt = render_system_prompt(
            bundle, self.tool_registry.get_all_schemas(), self.tool_call_format
        )
        orchestrator = AgenticOrchestrator(
            model_provider=self.model_provider,
            tool_registry=self.tool_registry,
            max_turns=self.max_turns,
            tool_limits=self.tool_limits,
            use_thinking=self.use_thinking,
            planning_suffix=planning_suffix(bundle),
            acting_suffix=acting_suffix(bundle),
        )
        states = orchestrator.run_batch(
            questions=[ex.question for ex in examples],
            question_ids=[ex.question_id for ex in examples],
            system_prompts=[system_prompt] * len(examples),
            attachments=[ex.get_attachments() or None for ex in examples],
        )
        self.rollouts += len(examples)
        return [self._record(ex, st) for ex, st in zip(examples, states)]

    @staticmethod
    def _record(ex, state):
        prediction = state.answer or ""
        aliases = list(ex.metadata.get("answer_aliases") or [])
        correct = bool(evaluate_answer(prediction, ex.answer, answer_aliases=aliases)["correct"])
        label = None
        if not correct:
            label = classify_failure(
                {
                    "action_history": state.action_history,
                    "prediction": prediction,
                    "turns": state.turn,
                    "question": ex.question,
                }
            )
        return TrajectoryRecord(
            question_id=ex.question_id,
            question=ex.question,
            gold=ex.answer,
            aliases=aliases,
            prediction=prediction,
            correct=correct,
            failure_label=label,
            plan=state.query_analysis or "",
            action_history=list(state.action_history),
            subagent_calls=list(state.metadata.get("subagent_calls") or []),
            turns=state.turn,
            tool_counts=dict(state.tool_counts),
        )


def accuracy(records):
    if not records:
        return 0.0
    return sum(1 for r in records if r.correct) / len(records)
