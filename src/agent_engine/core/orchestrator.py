"""Core orchestrator for agentic reasoning with tool use.

This module implements the main reasoning loop that coordinates LLM generation
with tool execution.
"""

import json
import os
import re
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

from ..models.base import BaseModelProvider
from ..utils.logging import get_logger
from ..utils.parsing import extract_answer, parse_tool_call, strip_thinking_tags
from ..utils.reasoning_context import get_attachment_context_for_code
from .state import ExecutionState
from .tool import ToolRegistry, ToolResult

logger = get_logger(__name__)


def _accumulate_usage(state: "ExecutionState", usage: Optional[Dict[str, int]]) -> None:
    """Accumulate token counts into state.metadata['token_usage'].

    Called after every LLM generation: orchestrator turns, batched web/code
    sub-agents, and any tool returning ToolResult.usage (text_inspector,
    image_inspector, web_search in single-run). The final state carries
    cumulative prompt/completion/total tokens across all models used.
    """
    if not usage:
        return
    tu = state.metadata.setdefault(
        "token_usage",
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        tu[key] = tu.get(key, 0) + int(usage.get(key, 0))


# Tools whose pre-call reasoning is indexed into the mind map graph.
# Mirrors MAT: reasoning before search, code, and mind_map calls is inserted.
_MIND_MAP_INDEXED_TOOLS = frozenset({"web_search", "code_generator", "mind_map"})

_IMAGE_EXTS: frozenset = frozenset({".jpg", ".jpeg", ".png"})
_TEXT_EXTS: frozenset = frozenset({
    ".txt", ".md", ".log", ".json", ".jsonl", ".xml",
    ".csv", ".tsv", ".yaml", ".yml", ".docx", ".xlsx",
    ".jsonld", ".parquet", ".pdf", ".pdb", ".pptx", ".py",
})


# ---------------------------------------------------------------------------
# Typed job descriptors for batched tool execution
# ---------------------------------------------------------------------------

class _WebJob(NamedTuple):
    state: ExecutionState
    tool_call: Dict[str, Any]
    tool: Any
    query: str
    payload: Dict[str, Any]


class _CodeJob(NamedTuple):
    state: ExecutionState
    tool_call: Dict[str, Any]
    tool: Any
    prompt: str


class _ImmediateResult(NamedTuple):
    state: ExecutionState
    tool_call: Dict[str, Any]
    result: ToolResult


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class AgenticOrchestrator:
    """Coordinates LLM reasoning with tool use.

    The orchestrator manages the main agentic reasoning loop:
    1. Generate next step using the model
    2. Parse output for tool calls
    3. Execute tools if needed
    4. Update conversation history
    5. Repeat until finished or max turns reached
    """

    def __init__(
        self,
        model_provider: BaseModelProvider,
        tool_registry: ToolRegistry,
        max_turns: int = 15,
        tool_limits: Optional[Dict[str, int]] = None,
        use_thinking: bool = False,
        cache_manager=None,
    ):
        """Initialize the orchestrator.

        Args:
            model_provider: Orchestrator LLM used for reasoning and tool-call generation.
            tool_registry: Registered tools the orchestrator may invoke.
            max_turns: Maximum reasoning turns before forcing an answer.
            tool_limits: Per-tool call limits, e.g. ``{"web_search": 10}``.
                         Tools not listed here have no limit.
            use_thinking: Whether the orchestrator should emit ``<think>`` output.
                          Has no effect if the model doesn't support thinking.
            cache_manager: Optional :class:`~agent_engine.caching.CacheManager`
                           for atomic cache persistence during parallel runs.
        """
        self.model = model_provider
        self.tools = tool_registry
        self.cache_manager = cache_manager
        self.max_turns = max_turns
        self.tool_limits = tool_limits or {"web_search": 10}
        self.use_thinking = use_thinking and model_provider.config.supports_thinking

        logger.info(f"Orchestrator initialized with {len(self.tools)} tools")
        logger.info(f"Thinking mode: {'enabled' if self.use_thinking else 'disabled'}")

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def run(
        self,
        question: str,
        question_id: int,
        system_prompt: str,
        attachments: Optional[List[str]] = None,
    ) -> ExecutionState:
        """Execute the agentic reasoning loop for a single question.

        Args:
            question: The question text to answer.
            question_id: Unique identifier for the question (used for context
                         manager namespacing and logging).
            system_prompt: Fully-formatted system prompt (tool schemas included).
            attachments: Optional list of file paths attached to the question
                         (e.g. an image or CSV file).

        Returns:
            :class:`~agent_engine.core.state.ExecutionState` populated with the
            final answer, full message history, and tool usage stats.
        """
        state = ExecutionState(
            question_id=question_id,
            question=question,
            messages=self._build_initial_messages(question, system_prompt, attachments),
            attachments=attachments,
        )

        self._init_mind_map(state)
        logger.info(f"Starting execution for question {question_id}")

        # Turn 0: Planning (does not count toward turn budget)
        self._run_planning_turn([state])

        while state.turn < self.max_turns and not state.finished:
            state.turn += 1
            logger.info(f"Turn {state.turn}/{self.max_turns}")

            try:
                memory_prompt = self._build_memory_prompt(state, system_prompt)
                prompt = self.model.apply_chat_template(memory_prompt, use_thinking=self.use_thinking)
                gen_result = self.model.generate([prompt])[0]
                state.current_output = gen_result.text
                _accumulate_usage(state, gen_result.usage)
            except Exception as e:
                logger.exception("Generation error")
                state.metadata["error"] = str(e)
                break

            tool_call = parse_tool_call(gen_result.text)
            if tool_call:
                self._index_reasoning_in_mind_map(gen_result.text, tool_call["name"], state)
                tool_result = self._execute_tool(tool_call, state)
                _accumulate_usage(state, tool_result.usage)
                state.add_message("assistant", gen_result.text)
                clean_output = strip_thinking_tags(tool_result.output or "")
                state.add_message("tool", f"<tool_response>\n{clean_output}\n</tool_response>")
                state.action_history.append({
                    "tool_name": tool_call["name"],
                    "sub_goal": self._extract_sub_goal(gen_result.text),
                    "command": json.dumps(tool_call.get("arguments") or {}),
                    "result": clean_output,
                })
                state.tool_calls.append(tool_call)
                state.increment_tool_count(tool_call["name"])
                logger.info(f"Tool '{tool_call['name']}' executed. Success: {tool_result.success}")
            else:
                state.add_message("assistant", gen_result.text)
                state.finished = True
                state.answer = extract_answer(gen_result.text)
                logger.info(f"Execution finished. Answer: {state.answer}")

        if not state.finished:
            logger.warning(f"Max turns ({self.max_turns}) reached without finishing")
            state.metadata["max_turns_reached"] = True
            state.answer = extract_answer(state.current_output)

        return state

    def run_batch(
        self,
        *,
        questions: Sequence[str],
        question_ids: Sequence[int],
        system_prompts: Sequence[str],
        attachments: Optional[Sequence[Optional[List[str]]]] = None,
    ) -> List[ExecutionState]:
        """Execute agentic reasoning for a batch of questions.

        All unfinished states are advanced by one turn in a single
        ``model.generate()`` call.  LLM-backed tool sub-agents
        (``web_search``, ``code_generator``) are also batched within each turn,
        amortizing per-call overhead across all active questions.

        Args:
            questions: Question texts (must be the same length as *question_ids*
                       and *system_prompts*).
            question_ids: Unique identifiers matching each question.
            system_prompts: Pre-built system prompts matching each question.
            attachments: Optional per-question attachment file paths.
                         ``None`` or a list of ``None`` values means no attachments.

        Returns:
            List of :class:`~agent_engine.core.state.ExecutionState` instances in
            the same order as *questions*.
        """
        if not (len(questions) == len(question_ids) == len(system_prompts)):
            raise ValueError("questions, question_ids, system_prompts must have the same length")

        resolved_attachments: Sequence[Optional[List[str]]] = attachments or ([None] * len(questions))
        if len(resolved_attachments) != len(questions):
            raise ValueError("attachments must be None or have the same length as questions")

        states = [
            ExecutionState(
                question_id=qid,
                question=q,
                messages=self._build_initial_messages(q, sp, att),
                attachments=att,
            )
            for q, qid, sp, att in zip(questions, question_ids, system_prompts, resolved_attachments)
        ]

        for s in states:
            self._init_mind_map(s)

        logger.info(f"Starting batched execution for {len(states)} questions")

        # Turn 0: Planning (single batched call, does not count toward turn budget)
        self._run_planning_turn(states)

        while True:
            active = [s for s in states if not s.finished and s.turn < self.max_turns]
            if not active:
                break
            self._process_batch_turn(active)

        for s in states:
            if not s.finished:
                logger.warning(f"Max turns ({self.max_turns}) reached for question {s.question_id}")
                s.metadata["max_turns_reached"] = True
                s.answer = extract_answer(s.current_output)
                s.finished = True

        return states

    def cleanup(self):
        """Release model and tool resources."""
        logger.info("Cleaning up orchestrator resources")
        self.model.cleanup()
        self.tools.cleanup_all()

    # ------------------------------------------------------------------ #
    # Batch turn processing                                               #
    # ------------------------------------------------------------------ #

    def _process_batch_turn(self, active: List[ExecutionState]) -> None:
        """Execute one reasoning turn for all active states."""
        for s in active:
            s.turn += 1

        system_prompt = active[0].messages[0]["content"]
        prompts = [
            self.model.apply_chat_template(
                self._build_memory_prompt(s, system_prompt),
                use_thinking=self.use_thinking,
            )
            for s in active
        ]

        try:
            gen_results = self.model.generate(prompts)
        except Exception as e:
            logger.exception("Batched generation error")
            for s in active:
                s.metadata["error"] = str(e)
                s.finished = True
            return

        web_jobs: List[_WebJob] = []
        code_jobs: List[_CodeJob] = []
        immediate_results: List[_ImmediateResult] = []

        for s, gen_result in zip(active, gen_results):
            s.current_output = gen_result.text
            _accumulate_usage(s, gen_result.usage)
            tool_call = parse_tool_call(gen_result.text)

            if tool_call:
                s.add_message("assistant", gen_result.text)
                self._classify_tool_call(s, tool_call, gen_result.text, web_jobs, code_jobs, immediate_results)
            else:
                s.add_message("assistant", gen_result.text)
                s.finished = True
                s.answer = extract_answer(gen_result.text)

        self._apply_immediate_results(immediate_results)
        if web_jobs:
            self._flush_web_batch(web_jobs)
        if code_jobs:
            self._flush_code_batch(code_jobs)

    def _classify_tool_call(
        self,
        state: ExecutionState,
        tool_call: Dict[str, Any],
        output_text: str,
        web_jobs: List[_WebJob],
        code_jobs: List[_CodeJob],
        immediate_results: List[_ImmediateResult],
    ) -> None:
        """Route a tool call to the appropriate execution path."""
        tool_name = tool_call["name"]
        tool = self.tools.get(tool_name)
        args = tool_call.get("arguments") or {}

        # Index reasoning into the mind map graph before tool execution.
        # Handles web_search, code_generator, and mind_map (mirrors MAT).
        self._index_reasoning_in_mind_map(output_text, tool_name, state)

        if tool and tool_name == "web_search" and not getattr(tool, "direct_mode", True):
            self._schedule_web_job(state, tool_call, tool, args, web_jobs, immediate_results)

        elif tool and tool_name == "code_generator" and not getattr(tool, "direct_mode", True):
            self._schedule_code_job(state, tool_call, tool, args, code_jobs, immediate_results)

        elif tool and tool_name == "mind_map":
            tool.set_current_question(state.question_id)
            immediate_results.append(_ImmediateResult(state, tool_call, self._execute_tool(tool_call, state)))

        else:
            immediate_results.append(_ImmediateResult(state, tool_call, self._execute_tool(tool_call, state)))

    def _schedule_web_job(
        self,
        state: ExecutionState,
        tool_call: Dict[str, Any],
        tool: Any,
        args: Dict[str, Any],
        web_jobs: List[_WebJob],
        immediate_results: List[_ImmediateResult],
    ) -> None:
        query = args.get("query", "")
        if not query or not hasattr(tool, "build_analysis_prompt") or not hasattr(tool, "search_and_format"):
            logger.info("Tool call: %s, %s", tool_call["name"], tool_call.get("arguments", {}))
            tr = ToolResult(success=False, output="", metadata={}, error="Missing required web_search arguments")
            immediate_results.append(_ImmediateResult(state, tool_call, tr))
            return

        analysis_cache = self._get_analysis_cache(tool)
        if query in analysis_cache:
            logger.info("Tool call: %s, %s", tool_call["name"], tool_call.get("arguments", {}))
            tr = ToolResult(success=True, output=analysis_cache[query], metadata={"cached": True, "query": query, "mode": "sub-agent"})
            immediate_results.append(_ImmediateResult(state, tool_call, tr))
            return

        try:
            payload = tool.search_and_format(query)
            web_jobs.append(_WebJob(state, tool_call, tool, query, payload))
        except Exception as exc:
            immediate_results.append(_ImmediateResult(state, tool_call, ToolResult(success=False, output="", metadata={"query": query}, error=str(exc))))

    def _schedule_code_job(
        self,
        state: ExecutionState,
        tool_call: Dict[str, Any],
        tool: Any,
        args: Dict[str, Any],
        code_jobs: List[_CodeJob],
        immediate_results: List[_ImmediateResult],
    ) -> None:
        task = args.get("task", "")
        if not task or not hasattr(tool, "build_task_prompt") or not hasattr(tool, "execute_code"):
            logger.info("Tool call: %s, %s", tool_call["name"], tool_call.get("arguments", {}))
            tr = ToolResult(success=False, output="", metadata={}, error="Missing required code_generator arguments")
            immediate_results.append(_ImmediateResult(state, tool_call, tr))
            return

        try:
            att_ctx = get_attachment_context_for_code(state)
            prompt = tool.build_task_prompt(task, context=att_ctx)
            code_jobs.append(_CodeJob(state, tool_call, tool, prompt))
        except Exception as exc:
            immediate_results.append(_ImmediateResult(state, tool_call, ToolResult(success=False, output="", metadata={}, error=str(exc))))

    def _apply_immediate_results(self, results: List[_ImmediateResult]) -> None:
        """Commit tool responses and update usage tracking for immediate results."""
        for item in results:
            _accumulate_usage(item.state, item.result.usage)
            clean_output = strip_thinking_tags(item.result.output or "")
            item.state.add_message("tool", f"<tool_response>\n{clean_output}\n</tool_response>")
            item.state.action_history.append({
                "tool_name": item.tool_call["name"],
                "sub_goal": self._extract_sub_goal(item.state.current_output),
                "command": json.dumps(item.tool_call.get("arguments") or {}),
                "result": clean_output,
            })
            item.state.tool_calls.append(item.tool_call)
            item.state.increment_tool_count(item.tool_call["name"])

    # ------------------------------------------------------------------ #
    # Batched web search                                                  #
    # ------------------------------------------------------------------ #

    def _flush_web_batch(self, jobs: List[_WebJob]) -> None:
        """Fetch URLs across all web jobs then run batched LLM analysis."""
        for job in jobs:
            logger.info("Tool call: %s, %s", job.tool_call["name"], job.tool_call.get("arguments", {}))
        self._fetch_urls_for_web_jobs(jobs)

        groups: Dict[int, List[_WebJob]] = {}
        for job in jobs:
            provider_id = id(getattr(job.tool, "model_provider", None))
            groups.setdefault(provider_id, []).append(job)

        for group in groups.values():
            self._run_web_analysis_batch(group)

    def _fetch_urls_for_web_jobs(self, jobs: List[_WebJob]) -> None:
        """Batch-fetch all URLs needed across all web search jobs."""
        all_urls: set = set()
        url_snippets: Dict[str, str] = {}
        for job in jobs:
            for url in job.payload.get("urls_to_fetch", []):
                all_urls.add(url)
                snippet = job.payload.get("url_snippets", {}).get(url)
                if snippet:
                    url_snippets[url] = snippet

        if not all_urls:
            return

        from ..external.url_fetcher import fetch_page_content
        logger.info(f"Batch fetching {len(all_urls)} URLs across {len(jobs)} web_search calls")
        try:
            use_jina = getattr(jobs[0].tool, "use_jina", False)
            fetched = fetch_page_content(list(all_urls), use_jina=use_jina, snippets=url_snippets)
            for job in jobs:
                if hasattr(job.tool, "url_cache"):
                    job.tool.url_cache.update(fetched)
            logger.info(f"Successfully fetched {len(fetched)} URLs")
            if self.cache_manager and fetched:
                self.cache_manager.save_url_cache()
        except Exception:
            logger.exception("Error during batch URL fetching")

    def _run_web_analysis_batch(self, jobs: List[_WebJob]) -> None:
        """Run LLM analysis for a group of web search jobs sharing a provider."""
        provider = getattr(jobs[0].tool, "model_provider", None)
        analysis_cache = self._get_analysis_cache(jobs[0].tool)

        prompts = [
            job.tool.build_analysis_prompt(
                job.query,
                job.tool._format_results(job.payload.get("results", []), job.query),
            )
            for job in jobs
        ]
        gen_outputs = provider.generate(prompts) if provider else []

        for job, out in zip(jobs, gen_outputs):
            _accumulate_usage(job.state, out.usage)
            text = strip_thinking_tags(out.text)
            analysis_cache[job.query] = text
            job.state.add_message("tool", f"<tool_response>\n{text}\n</tool_response>")
            job.state.action_history.append({
                "tool_name": job.tool_call["name"],
                "sub_goal": self._extract_sub_goal(job.state.current_output),
                "command": json.dumps(job.tool_call.get("arguments") or {}),
                "result": text,
            })
            job.state.tool_calls.append(job.tool_call)
            job.state.increment_tool_count(job.tool_call["name"])

    # ------------------------------------------------------------------ #
    # Batched code generation                                             #
    # ------------------------------------------------------------------ #

    def _flush_code_batch(self, jobs: List[_CodeJob]) -> None:
        """Run batched LLM code writing then execute each script."""
        groups: Dict[int, List[_CodeJob]] = {}
        for job in jobs:
            provider_id = id(getattr(job.tool, "model_provider", None))
            groups.setdefault(provider_id, []).append(job)

        for group in groups.values():
            self._run_code_generation_batch(group)

    def _run_code_generation_batch(self, jobs: List[_CodeJob]) -> None:
        """LLM code-write + execute for a group sharing a provider."""
        provider = getattr(jobs[0].tool, "model_provider", None)
        gen_outputs = provider.generate([job.prompt for job in jobs]) if provider else []

        for job, out in zip(jobs, gen_outputs):
            _accumulate_usage(job.state, out.usage)
            try:
                text = strip_thinking_tags(out.text)
                code = job.tool.extract_code_from_llm_response(text)
                logger.info("Tool call: %s, %s", job.tool_call["name"], job.tool_call.get("arguments", {}))
                tr = job.tool.execute_code(code)
            except Exception as exc:
                tr = ToolResult(success=False, output="", metadata={}, error=str(exc))
            clean_output = strip_thinking_tags(tr.output or "")
            job.state.add_message("tool", f"<tool_response>\n{clean_output}\n</tool_response>")
            job.state.action_history.append({
                "tool_name": job.tool_call["name"],
                "sub_goal": self._extract_sub_goal(job.state.current_output),
                "command": json.dumps(job.tool_call.get("arguments") or {}),
                "result": clean_output,
            })
            job.state.tool_calls.append(job.tool_call)
            job.state.increment_tool_count(job.tool_call["name"])

    # ------------------------------------------------------------------ #
    # Message construction                                                #
    # ------------------------------------------------------------------ #

    def _build_initial_messages(
        self,
        question: str,
        system_prompt: str,
        attachments: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Build the initial [system, user] message pair."""
        user_content = question
        for att_path in (attachments or []):
            if att_path:
                user_content += self._format_attachment_note(att_path)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _format_attachment_note(self, att_path: str) -> str:
        """Build the MAT-style [Attachment] block appended to the user message."""
        clean_path = att_path.strip().split("?", 1)[0].split("#", 1)[0]
        ext = os.path.splitext(clean_path)[1].lower()
        fname = os.path.basename(clean_path)

        lines = [
            "\n\n[Attachment]",
            f"- There is an attached file for this question: {fname}",
        ]

        if ext in _IMAGE_EXTS and self.tools.get("image_inspector") is not None:
            lines.append("- To inspect the image, call the tool `image_inspector` with a question about the image.")
        elif ext in _TEXT_EXTS and self.tools.get("text_inspector") is not None:
            lines.append("- To read the file, call the tool `text_inspector` (optionally with a question).")
        else:
            lines.append("- The attachment type is not supported by the available inspectors in this run.")

        lines.append("- Important: do NOT guess or provide file paths; inspectors use the attached file automatically.")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ #
    # Structured memory (AgentFlow-inspired)                              #
    # ------------------------------------------------------------------ #

    def _build_memory_prompt(
        self, state: ExecutionState, system_prompt: str
    ) -> List[Dict[str, str]]:
        """Build a fresh [system, user] prompt with structured memory.

        Mirrors AgentFlow's planner prompt: the model receives the original
        query, a query analysis, and a structured record of previous steps
        instead of the full growing conversation.
        """
        # Reconstruct original user content from messages[1]
        user_content = state.messages[1]["content"] if len(state.messages) > 1 else state.question

        parts = [user_content]

        if state.query_analysis:
            parts.append(f"\n**Query Analysis:**\n{state.query_analysis}")

        if state.action_history:
            parts.append(f"\n**Previous Steps:**\n{self._format_action_history(state.action_history)}")

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(parts)},
        ]

    @staticmethod
    def _format_action_history(action_history: List[Dict[str, str]]) -> str:
        """Format action history into AgentFlow-style ``Action Step N`` blocks."""
        lines = []
        for i, action in enumerate(action_history, 1):
            lines.append(f"Action Step {i}:")
            lines.append(f"  - Tool: {action.get('tool_name', 'unknown')}")
            lines.append(f"  - Sub-goal: {action.get('sub_goal', '')}")
            lines.append(f"  - Command: {action.get('command', '')}")
            lines.append(f"  - Result: {action.get('result', '')}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _extract_sub_goal(output_text: str) -> str:
        """Extract the reasoning sub-goal from assistant output.

        Strips <think> and <tool_call> blocks, returning the remaining text
        truncated to 500 chars.
        """
        text = strip_thinking_tags(output_text)
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
        text = text.strip()
        return text[:500] if len(text) > 500 else text

    def _run_planning_turn(self, states: List[ExecutionState]) -> None:
        """Execute Turn 0 (planning) for all states in a single batched call.

        The model analyzes each question without calling tools. The output is
        stored in ``state.query_analysis``. If the model produces a tool call
        or final answer, those are handled as edge cases.

        """
        if len(self.tools) == 0:
            # NOTE: double check if it should look like this, or if we should just omit the query analysis step entirely when no tools are available
            planning_suffix = (
                "\n\nBefore answering, analyze this query to determine the approach needed.\n"
                "Instructions:\n"
                "1. Identify the main objectives in the query.\n"
                "2. Break down the problem into sub-tasks.\n"
                "3. Consider what knowledge or reasoning steps are required.\n"
                "Be brief and precise. Do NOT provide the final answer yet."
            )
        else:
            planning_suffix = (
                "\n\nBefore using any tools, analyze this query to determine the approach needed.\n"
                "Instructions:\n"
                "1. Identify the main objectives in the query.\n"
                "2. List the necessary skills and tools.\n"
                "3. For each tool, explain how it helps address the query.\n"
                "4. Note any additional considerations.\n\n"
                "Be brief and precise. Do NOT call any tools yet."
            )

        prompts = []
        for s in states:
            planning_messages = list(s.messages)  # shallow copy
            planning_messages[-1] = {
                "role": planning_messages[-1]["role"],
                "content": planning_messages[-1]["content"] + planning_suffix,
            }
            prompts.append(
                self.model.apply_chat_template(planning_messages, use_thinking=self.use_thinking)
            )

        try:
            gen_results = self.model.generate(prompts)
        except Exception as e:
            logger.exception("Planning turn generation error")
            for s in states:
                s.metadata["planning_error"] = str(e)
            return

        for s, gen_result in zip(states, gen_results):
            _accumulate_usage(s, gen_result.usage)
            text = gen_result.text

            # Edge case: model produced a tool call — use text before it as analysis
            tool_call = parse_tool_call(text)
            if tool_call:
                analysis = self._extract_sub_goal(text)
                s.query_analysis = analysis
                logger.info(
                    "Planning turn for Q%s produced tool call (discarded); analysis: %.100s...",
                    s.question_id, analysis,
                )
            # Edge case: model produced a final answer
            elif "\\boxed{" in text or "\\boxed " in text:
                s.query_analysis = strip_thinking_tags(text)
                s.finished = True
                s.answer = extract_answer(text)
                logger.info("Planning turn for Q%s produced final answer: %s", s.question_id, s.answer)
            else:
                s.query_analysis = strip_thinking_tags(text)
                logger.info("Planning turn for Q%s complete: %.100s...", s.question_id, s.query_analysis)

            # Log planning output to messages for debugging
            s.add_message("assistant", text)

    # ------------------------------------------------------------------ #
    # Tool execution                                                      #
    # ------------------------------------------------------------------ #

    def _execute_tool(self, tool_call: Dict[str, Any], state: ExecutionState) -> ToolResult:
        """Dispatch a single tool call, injecting attachment paths where needed."""
        tool_name = tool_call["name"]
        tool = self.tools.get(tool_name)

        if not tool:
            logger.error(f"Tool not found: {tool_name}")
            return ToolResult(success=False, output="", metadata={}, error=f"Tool '{tool_name}' not found")

        if not self._check_tool_limit(tool_name, state):
            logger.warning(f"Tool limit exceeded for: {tool_name}")
            return ToolResult(success=False, output=f"Tool usage limit reached for {tool_name}", metadata={}, error="Limit exceeded")

        try:
            arguments = dict(tool_call.get("arguments") or {})
            inject_error = self._inject_attachment_path(tool_name, state, arguments)
            if inject_error:
                return ToolResult(success=False, output="", metadata={}, error=inject_error)

            logger.info("Tool call: %s, %s", tool_name, arguments)
            return tool.execute(**arguments)
        except Exception as e:
            logger.exception("Tool execution error")
            return ToolResult(success=False, output="", metadata={}, error=str(e))

    def _inject_attachment_path(
        self,
        tool_name: str,
        state: ExecutionState,
        arguments: Dict[str, Any],
    ) -> Optional[str]:
        """Inject `full_file_path` for file-inspector tools.

        Returns an error message string if the required attachment is missing,
        or None on success.
        """
        if tool_name == "image_inspector":
            path = self._get_first_attachment_with_exts(state, _IMAGE_EXTS)
            if not path:
                return "No supported image attachment found (provide image via attachments)"
            arguments["full_file_path"] = path

        elif tool_name == "text_inspector":
            path = self._get_first_attachment_with_exts(state, _TEXT_EXTS)
            if not path:
                return "No supported text attachment found (provide file via attachments)"
            arguments["full_file_path"] = path

        return None

    def _check_tool_limit(self, tool_name: str, state: ExecutionState) -> bool:
        if tool_name not in self.tool_limits:
            return True
        return state.get_tool_count(tool_name) < self.tool_limits[tool_name]

    # ------------------------------------------------------------------ #
    # Attachment helpers                                                  #
    # ------------------------------------------------------------------ #

    def _get_first_attachment_with_exts(self, state: ExecutionState, exts: frozenset) -> Optional[str]:
        """Return the first attachment whose extension is in *exts*, or None."""
        for path in (state.attachments or []):
            if not path or not isinstance(path, str):
                continue
            clean = path.strip().split("?", 1)[0].split("#", 1)[0]
            if os.path.splitext(clean)[1].lower() in exts:
                return path
        return None

    # ------------------------------------------------------------------ #
    # Mind map helpers                                             #
    # ------------------------------------------------------------------ #

    def _init_mind_map(self, state: ExecutionState) -> None:
        """Set the active question context on the mind map tool (both modes)."""
        tool = self.tools.get("mind_map")
        if tool is not None:
            tool.set_current_question(state.question_id)

    def _index_reasoning_in_mind_map(
        self, output_text: str, tool_name: str, state: ExecutionState
    ) -> None:
        """Index pre-tool reasoning into the mind map graph (non-direct mode).

        Mirrors MAT: reasoning produced before web_search, code_generator, and
        mind_map calls is inserted into the GraphRAG knowledge base so the
        mind_map query has richer context to draw from.
        """
        if tool_name not in _MIND_MAP_INDEXED_TOOLS:
            return
        tool = self.tools.get("mind_map")
        if tool is None or getattr(tool, "direct_mode", True):
            return
        tool.set_current_question(state.question_id)
        reasoning = re.sub(r"<tool_call>.*?</tool_call>", "", output_text, flags=re.DOTALL).strip()
        if reasoning:
            tool.add_entry(reasoning, state.question_id)

    # ------------------------------------------------------------------ #
    # Utility                                                             #
    # ------------------------------------------------------------------ #

    def _get_mind_map_for_state(self, state: ExecutionState):
        """Return the mind map's GraphRAG for this question if available."""
        ctx_tool = self.tools.get("mind_map")
        if not ctx_tool or not getattr(ctx_tool, "use_graphrag", False):
            return None
        return getattr(ctx_tool, "graphrag_instances", {}).get(state.question_id)

    @staticmethod
    def _get_analysis_cache(tool: Any) -> Dict[str, str]:
        """Lazily create and return the per-tool web-analysis cache."""
        cache = getattr(tool, "_analysis_cache", None)
        if cache is None:
            cache = {}
            setattr(tool, "_analysis_cache", cache)
        return cache
