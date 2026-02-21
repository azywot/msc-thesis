"""Core orchestrator for agentic reasoning with tool use.

This module implements the main reasoning loop that coordinates LLM generation
with tool execution.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..models.base import BaseModelProvider
from ..utils.logging import get_logger
from ..utils.parsing import extract_answer, parse_qwen3_tool_call, subagent_output_for_orchestrator
from .state import ExecutionState
from .tool import BaseTool, ToolRegistry, ToolResult

logger = get_logger(__name__)


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
    ):
        """Initialize orchestrator.

        Args:
            model_provider: Model provider for generation
            tool_registry: Registry of available tools
            max_turns: Maximum reasoning turns per question
            tool_limits: Optional per-tool usage limits
            use_thinking: Whether to use thinking mode for generation
        """
        self.model = model_provider
        self.tools = tool_registry
        self.max_turns = max_turns
        self.tool_limits = tool_limits or {'web_search': 10}
        self.use_thinking = use_thinking and model_provider.config.supports_thinking

        logger.info(f"Orchestrator initialized with {len(self.tools)} tools")
        logger.info(f"Thinking mode: {'enabled' if self.use_thinking else 'disabled'}")

    def run(
        self,
        question: str,
        question_id: int,
        system_prompt: str,
        attachments: Optional[List[str]] = None
    ) -> ExecutionState:
        """Execute agentic reasoning loop for a question.

        Args:
            question: Question text
            question_id: Unique question identifier
            system_prompt: System prompt with instructions and tool schemas
            attachments: Optional list of file attachments

        Returns:
            ExecutionState with complete execution history
        """
        # Initialize state
        state = ExecutionState(
            question_id=question_id,
            question=question,
            messages=self._build_initial_messages(question, system_prompt, attachments)
        )

        logger.info(f"Starting execution for question {question_id}")

        # Main reasoning loop
        while state.turn < self.max_turns and not state.finished:
            state.turn += 1
            logger.info(f"Turn {state.turn}/{self.max_turns}")

            # Generate next step
            try:
                prompt = self.model.apply_chat_template(
                    state.messages,
                    use_thinking=self.use_thinking
                )
                result = self.model.generate([prompt])[0]
                state.current_output = result.text

                logger.debug(f"Generated output: {result.text[:200]}...")

            except Exception as e:
                logger.exception("Generation error")
                state.metadata["error"] = str(e)
                break

            # Check for tool call
            tool_call = parse_qwen3_tool_call(result.text)

            if tool_call:
                # Tool call detected
                tool_name = tool_call["name"]
                logger.info(f"Tool call detected: {tool_name}")

                # Execute tool
                tool_result = self._execute_tool(tool_call, state)

                # Update conversation
                state.add_message("assistant", result.text)
                state.add_message("tool", f"<tool_response>\n{tool_result.output}\n</tool_response>")

                # Track usage
                state.tool_calls.append(tool_call)
                state.increment_tool_count(tool_name)

                logger.info(f"Tool executed. Success: {tool_result.success}")

            else:
                # No tool call - finish
                state.add_message("assistant", result.text)
                state.finished = True
                state.answer = extract_answer(result.text)

                logger.info(f"Execution finished. Answer: {state.answer}")

        # Handle max turns reached
        if not state.finished:
            logger.warning(f"Max turns ({self.max_turns}) reached without finishing")
            state.metadata["max_turns_reached"] = True
            # Try to extract answer from last output
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

        This mirrors the multi-sequence batching approach from the legacy
        `multi-agent-tools` runner:
        - At each turn, generate for *all unfinished* states in one call
          to `model.generate(prompts)`.
        - Then execute tools per state and continue until all are finished
          or max turns is reached.

        Notes:
        - Tool execution is still per-state (not batched).
        - Providers that don't have true batching (OpenAI/Anthropic) will
          still benefit from shared control flow but will not speed up.
        """
        if not (len(questions) == len(question_ids) == len(system_prompts)):
            raise ValueError("questions, question_ids, system_prompts must have the same length")

        if attachments is None:
            attachments = [None] * len(questions)
        if len(attachments) != len(questions):
            raise ValueError("attachments must be None or have the same length as questions")

        states: List[ExecutionState] = []
        for q, qid, sp, att in zip(questions, question_ids, system_prompts, attachments):
            states.append(
                ExecutionState(
                    question_id=qid,
                    question=q,
                    messages=self._build_initial_messages(q, sp, att),
                )
            )

        logger.info(f"Starting batched execution for {len(states)} questions")

        # Run until all states are finished or max turns reached.
        while True:
            active = [s for s in states if (not s.finished and s.turn < self.max_turns)]
            if not active:
                break

            # Increment turn per active state (matches sequential semantics).
            for s in active:
                s.turn += 1

            # Build prompts for active states.
            prompts: List[str] = []
            for s in active:
                prompts.append(self.model.apply_chat_template(s.messages, use_thinking=self.use_thinking))

            # Generate in a single call for the batch.
            try:
                results = self.model.generate(prompts)
            except Exception as e:
                logger.exception("Batched generation error")
                for s in active:
                    s.metadata["error"] = str(e)
                    s.finished = True
                break

            # Process outputs per state.
            # Tool batching jobs (only for LLM-backed sub-agent tools).
            web_jobs: List[Tuple[ExecutionState, Dict[str, Any], Any, str, Dict[str, Any]]] = []  # Added payload
            code_jobs: List[Tuple[ExecutionState, Dict[str, Any], Any, str]] = []

            immediate_tool_results: List[Tuple[ExecutionState, Dict[str, Any], ToolResult]] = []

            for s, result in zip(active, results):
                s.current_output = result.text

                tool_call = parse_qwen3_tool_call(result.text)
                if tool_call:
                    tool_name = tool_call["name"]
                    tool = self.tools.get(tool_name)

                    # Always add the assistant message first.
                    s.add_message("assistant", result.text)

                    # Schedule batching only for sub-agent LLM steps:
                    # - web_search: batch the summarization LLM call (Serper call stays per job)
                    # - code_generator: batch the code-writing LLM call (Python execution stays per job)
                    if tool and tool_name == "web_search" and getattr(tool, "direct_mode", True) is False:
                        args = tool_call.get("arguments", {}) or {}
                        query = args.get("query", "")
                        if query and hasattr(tool, "build_analysis_prompt") and hasattr(tool, "search_and_format"):
                            # Keep raw Serper results in `search_cache`.
                            # Cache analyzed summaries separately so we don't corrupt the cache format.
                            analysis_cache = getattr(tool, "_analysis_cache", None)
                            if analysis_cache is None:
                                analysis_cache = {}
                                setattr(tool, "_analysis_cache", analysis_cache)

                            if query in analysis_cache:
                                tr = ToolResult(
                                    success=True,
                                    output=analysis_cache[query],
                                    metadata={"cached": True, "query": query, "mode": "sub-agent"},
                                )
                                immediate_tool_results.append((s, tool_call, tr))
                            else:
                                # Run Serper (or reuse cached raw results) and collect URLs (don't fetch yet).
                                try:
                                    payload = tool.search_and_format(query)
                                    web_jobs.append((s, tool_call, tool, query, payload))
                                except Exception as exc:
                                    tr = ToolResult(success=False, output="", metadata={"query": query}, error=str(exc))
                                    immediate_tool_results.append((s, tool_call, tr))
                        else:
                            tr = ToolResult(success=False, output="", metadata={}, error="Missing required web_search arguments")
                            immediate_tool_results.append((s, tool_call, tr))

                    elif tool and tool_name == "code_generator" and getattr(tool, "direct_mode", True) is False:
                        args = tool_call.get("arguments", {}) or {}
                        task = args.get("task", "")
                        if task and hasattr(tool, "build_task_prompt") and hasattr(tool, "execute_code"):
                            try:
                                prompt = tool.build_task_prompt(task)
                                code_jobs.append((s, tool_call, tool, prompt))
                            except Exception as exc:
                                tr = ToolResult(success=False, output="", metadata={}, error=str(exc))
                                immediate_tool_results.append((s, tool_call, tr))
                        else:
                            tr = ToolResult(success=False, output="", metadata={}, error="Missing required code_generator arguments")
                            immediate_tool_results.append((s, tool_call, tr))

                    else:
                        # Default: execute tool immediately (non-batched).
                        tr = self._execute_tool(tool_call, s)
                        immediate_tool_results.append((s, tool_call, tr))
                else:
                    s.add_message("assistant", result.text)
                    s.finished = True
                    s.answer = extract_answer(result.text)

            # Apply immediate tool results (also increments usage tracking).
            for s, tool_call, tr in immediate_tool_results:
                tool_name = tool_call["name"]
                s.add_message("tool", f"<tool_response>\n{tr.output}\n</tool_response>")
                s.tool_calls.append(tool_call)
                s.increment_tool_count(tool_name)

            # Batch URL fetching for web_search jobs (matches old repo behavior).
            if web_jobs:
                # Collect all URLs that need fetching across all web jobs
                all_urls_to_fetch = set()
                url_snippets = {}
                for job in web_jobs:
                    payload = job[4]
                    urls = payload.get("urls_to_fetch", [])
                    snippets = payload.get("url_snippets", {})
                    for url in urls:
                        all_urls_to_fetch.add(url)
                        if url in snippets:
                            url_snippets[url] = snippets[url]
                
                # Batch-fetch all URLs if needed
                if all_urls_to_fetch:
                    from ..external.url_fetcher import fetch_page_content
                    logger.info(f"Batch fetching {len(all_urls_to_fetch)} URLs across all web_search calls")
                    try:
                        tool0 = web_jobs[0][2]
                        use_jina = getattr(tool0, "use_jina", False)
                        fetched = fetch_page_content(
                            list(all_urls_to_fetch),
                            use_jina=use_jina,
                            snippets=url_snippets
                        )
                        # Update URL cache for all web_search tools (they share the same cache)
                        for job in web_jobs:
                            tool = job[2]
                            if hasattr(tool, "url_cache"):
                                tool.url_cache.update(fetched)
                        logger.info(f"Successfully fetched {len(fetched)} URLs")
                    except Exception:
                        logger.exception("Error during batch URL fetching")

            # Batched web_search LLM analysis (group by provider instance).
            if web_jobs:
                groups: Dict[int, List[Tuple[ExecutionState, Dict[str, Any], Any, str, str]]] = {}
                for job in web_jobs:
                    tool = job[2]
                    query = job[3]
                    payload = job[4]
                    # Regenerate formatted_results with fetched content (URLs are now in cache)
                    results = payload.get("results", [])
                    formatted_results = tool._format_results(results, query)
                    prompt = tool.build_analysis_prompt(query, formatted_results)
                    provider_id = id(getattr(tool, "model_provider", None))
                    groups.setdefault(provider_id, []).append((job[0], job[1], tool, query, prompt))

                for _, jobs in groups.items():
                    tool0 = jobs[0][2]
                    provider = getattr(tool0, "model_provider", None)
                    prompts = [j[4] for j in jobs]
                    outs = provider.generate(prompts) if provider else []
                    for (s, tool_call, tool, query, _), out in zip(jobs, outs):
                        text = subagent_output_for_orchestrator(out.text)
                        # Cache analyzed summaries separately.
                        try:
                            analysis_cache = getattr(tool, "_analysis_cache", None)
                            if analysis_cache is None:
                                analysis_cache = {}
                                setattr(tool, "_analysis_cache", analysis_cache)
                            analysis_cache[query] = text
                        except Exception:
                            pass
                        tr = ToolResult(
                            success=True,
                            output=text,
                            metadata={"cached": False, "query": query, "mode": "sub-agent"},
                        )
                        s.add_message("tool", f"<tool_response>\n{tr.output}\n</tool_response>")
                        s.tool_calls.append(tool_call)
                        s.increment_tool_count(tool_call["name"])

            # Batched code_generator LLM code writing (group by provider instance).
            if code_jobs:
                groups2: Dict[int, List[Tuple[ExecutionState, Dict[str, Any], Any, str]]] = {}
                for job in code_jobs:
                    tool = job[2]
                    provider_id = id(getattr(tool, "model_provider", None))
                    groups2.setdefault(provider_id, []).append(job)

                for _, jobs in groups2.items():
                    tool0 = jobs[0][2]
                    provider = getattr(tool0, "model_provider", None)
                    prompts = [j[3] for j in jobs]
                    outs = provider.generate(prompts) if provider else []
                    for (s, tool_call, tool, _), out in zip(jobs, outs):
                        # Extract code, then execute it (per-item)
                        try:
                            text = subagent_output_for_orchestrator(out.text)
                            code = tool.extract_code_from_llm_response(text)
                            tr = tool.execute_code(code)
                        except Exception as exc:
                            tr = ToolResult(success=False, output="", metadata={}, error=str(exc))
                        s.add_message("tool", f"<tool_response>\n{tr.output}\n</tool_response>")
                        s.tool_calls.append(tool_call)
                        s.increment_tool_count(tool_call["name"])

        # Post-process: handle max turns reached.
        for s in states:
            if not s.finished:
                logger.warning(f"Max turns ({self.max_turns}) reached for question {s.question_id}")
                s.metadata["max_turns_reached"] = True
                s.answer = extract_answer(s.current_output)
                s.finished = True

        return states

    def _build_initial_messages(
        self,
        question: str,
        system_prompt: str,
        attachments: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """Build initial conversation messages.

        Args:
            question: Question text
            system_prompt: System prompt
            attachments: Optional file attachments

        Returns:
            List of initial messages
        """
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Add question
        question_text = question
        if attachments:
            attachment_note = "\n\nAttached files: " + ", ".join(attachments)
            question_text += attachment_note

        messages.append({"role": "user", "content": question_text})

        return messages

    def _execute_tool(
        self,
        tool_call: Dict[str, str],
        state: ExecutionState
    ) -> ToolResult:
        """Execute a tool and handle errors.

        Args:
            tool_call: Tool call dictionary with 'name' and 'arguments'
            state: Current execution state

        Returns:
            ToolResult with execution outcome
        """
        tool_name = tool_call["name"]
        tool = self.tools.get(tool_name)

        if not tool:
            logger.error(f"Tool not found: {tool_name}")
            return ToolResult(
                success=False,
                output="",
                metadata={},
                error=f"Tool '{tool_name}' not found"
            )

        # Check limits
        if not self._check_tool_limit(tool_name, state):
            logger.warning(f"Tool limit exceeded for: {tool_name}")
            return ToolResult(
                success=False,
                output=f"Tool usage limit reached for {tool_name}",
                metadata={},
                error="Limit exceeded"
            )

        # Execute
        try:
            arguments = tool_call.get("arguments", {})
            logger.debug(f"Executing {tool_name} with args: {arguments}")
            return tool.execute(**arguments)
        except Exception as e:
            logger.exception("Tool execution error")
            return ToolResult(
                success=False,
                output="",
                metadata={},
                error=str(e)
            )

    def _check_tool_limit(self, tool_name: str, state: ExecutionState) -> bool:
        """Check if tool usage is within limits.

        Args:
            tool_name: Name of the tool
            state: Current execution state

        Returns:
            True if within limits, False otherwise
        """
        if tool_name not in self.tool_limits:
            return True

        limit = self.tool_limits[tool_name]
        current_count = state.get_tool_count(tool_name)
        return current_count < limit

    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up orchestrator resources")
        self.model.cleanup()
        self.tools.cleanup_all()
