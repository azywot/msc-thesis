"""Code execution tool for Python code generation and execution.

This tool executes Python code in a sandboxed subprocess with timeout.

Supports two modes:
1. Direct mode: Executes provided Python code directly
2. Sub-agent mode: LLM generates code from task description, then executes
"""

import ast
import os
import re
import signal
import subprocess
import time
from typing import Any, Dict, List, Optional

from ..core.tool import BaseTool, ToolResult
from ..utils.logging import get_logger
from ..utils.parsing import strip_thinking_tags

logger = get_logger(__name__)


class CodeGeneratorTool(BaseTool):
    """Execute Python code in a sandboxed subprocess.

    Supports direct mode (execute provided code) or sub-agent mode
    (LLM generates code from task description).
    """

    def __init__(
        self,
        timeout_seconds: int = 60,
        max_output_chars: int = 8000,
        temp_dir: str = "./cache/code_temp",
        model_provider = None,  # Optional: for sub-agent mode
        use_thinking: bool = False  # Whether sub-agent uses thinking
    ):
        """Initialize code execution tool.

        Args:
            timeout_seconds: Maximum execution time
            max_output_chars: Maximum output length
            temp_dir: Directory for temporary files
            model_provider: Optional model provider for sub-agent mode
            use_thinking: Whether sub-agent uses thinking mode
        """
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.temp_dir = temp_dir
        self.model_provider = model_provider
        self.use_thinking = use_thinking
        self.direct_mode = model_provider is None

        # Create temp directory
        os.makedirs(self.temp_dir, exist_ok=True)

    @property
    def name(self) -> str:
        return "code_generator"

    @property
    def description(self) -> str:
        return "Execute Python code to perform calculations, data analysis, or other computational tasks"

    def get_schema(self) -> Dict[str, Any]:
        """Return Qwen3 JSON Schema."""
        if self.direct_mode:
            # Direct mode: expects actual Python code
            return {
                "type": "function",
                "function": {
                    "name": "code_generator",
                    "description": (
                        "Execute Python code to perform calculations, data processing, or solve computational problems. "
                        "Provide the Python code directly; the system will execute it and return stdout/stderr."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": (
                                    "Python code to execute. It must be a standalone script and must print its output. "
                                    "Do not include markdown fences."
                                ),
                            }
                        },
                        "required": ["code"]
                    }
                }
            }
        else:
            # Sub-agent mode: expects task description
            return {
                "type": "function",
                "function": {
                    "name": "code_generator",
                    "description": (
                        "Generate and execute Python code to perform calculations, data processing, or solve computational problems. "
                        "The code will be executed and the output will be returned."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "A clear description of what the code should do. Be specific about inputs, expected outputs, and any constraints."
                            }
                        },
                        "required": ["task"]
                    }
                }
            }

    def execute(self, code: str = None, task: str = None, context: str = "") -> ToolResult:
        """Execute Python code (direct) or generate then execute (sub-agent).

        Args:
            code: Python code string to execute (direct mode)
            task: Task description for code generation (sub-agent mode)
            context: Previous reasoning for sub-agent mode

        Returns:
            ToolResult with execution output or error
        """
        logger.info(f"Executing code ({'sub-agent' if not self.direct_mode else 'direct'} mode)")

        if not self.direct_mode:
            if not task:
                return ToolResult(
                    success=False,
                    output="",
                    metadata={},
                    error="Task description required for sub-agent mode",
                )
            code = self.generate_code(task, context)
            if not code:
                return ToolResult(
                    success=False,
                    output="",
                    metadata={},
                    error="Failed to generate code from task description",
                )

        return self.execute_code(code)

    def build_task_prompt(self, task: str, context: str = "") -> str:
        """Build the sub-agent LLM prompt for code generation.

        This is used by both single and batched sub-agent execution.
        context: Previous reasoning and optional attachment info (MAT-style).
        """
        context_block = f"Context:\n\n{context}\n\n" if context else ""
        prompt = (
            "You are a code generator. Generate ONLY executable Python code, with NO explanations, "
            "NO comments about what the code does, and NO additional text.\n\n"
            f"{context_block}Problem: {task}\n\n"
            "Requirements:\n"
            "- Output ONLY the Python code\n"
            "- The code must be executable as a standalone script\n"
            "- The code must print its output directly\n"
            "- NO explanatory text before or after the code\n"
            "- NO comments like \"Here's the code\" or \"This will output\"\n\n"
            "Python code:"
        )
        prompt_messages = [{"role": "user", "content": prompt}]
        return self.model_provider.apply_chat_template(prompt_messages, use_thinking=self.use_thinking)

    def extract_code_from_llm_response(self, response_text: str) -> str:
        """Extract Python code from the LLM response (robust to markdown fences)."""
        return self._extract_code_from_response(response_text)

    def generate_code(self, task: str, context: str = "") -> str:
        """Generate Python code from a task (sub-agent mode).

        In batched mode, the orchestrator should call `build_task_prompt(...)`,
        batch `model_provider.generate(prompts)`, then call
        `extract_code_from_llm_response(...)`.
        context: Previous reasoning.
        """
        prompt = self.build_task_prompt(task, context)
        result = self.model_provider.generate([prompt])[0]

        output = strip_thinking_tags(result.text)
        return self.extract_code_from_llm_response(output)

    def execute_code(self, code: Optional[str]) -> ToolResult:
        """Execute a concrete Python code string (shared by direct + sub-agent).

        Temp files written to ``self.temp_dir`` are intentionally kept on disk
        for post-hoc debugging. Remove them manually after a run if disk space
        is a concern.
        """
        if not code or not code.strip():
            return ToolResult(
                success=False,
                output="",
                metadata={},
                error="Empty code provided",
            )

        code = code.strip("\ufeff \t\r\n")

        try:
            ast.parse(code)
        except SyntaxError as exc:
            logger.error(f"Syntax error in code: {exc}")
            return ToolResult(
                success=False,
                output="",
                metadata={},
                error=f"Invalid Python syntax: {exc}",
            )

        run_id = os.getenv("SLURM_JOB_ID", f"{os.getpid()}_{int(time.time())}")
        temp_file_path = os.path.join(self.temp_dir, f"temp_code_{run_id}.py")

        try:
            with open(temp_file_path, "w") as f:
                f.write(code)
        except Exception as e:
            logger.error(f"Failed to write code to file: {e}")
            return ToolResult(
                success=False,
                output="",
                metadata={},
                error=f"Failed to write code: {e}",
            )

        logger.info("Code written to: %s", temp_file_path)
        code_preview = code if len(code) <= 3000 else code[:3000] + "\n... [truncated]"
        logger.info("Generated code:\n%s", code_preview)
        try:
            process = subprocess.Popen(
                ["python", temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                close_fds=True,
            )

            try:
                stdout, stderr = process.communicate(timeout=self.timeout_seconds)

                stdout = self._trim_output(stdout)
                stderr = self._trim_output(stderr)

                if process.returncode == 0:
                    logger.info("Code executed successfully")
                    return ToolResult(
                        success=True,
                        output=stdout or "(No output)",
                        metadata={"return_code": 0},
                    )

                # Execution failed
                error_msg = self._extract_error(stderr)
                logger.warning(f"Code execution failed: {error_msg}")

                if stdout and stdout.strip():
                    output = (
                        f"{stdout.strip()}\n\n"
                        "[Note: Code execution encountered an error. Output above may be incomplete.]"
                    )
                    return ToolResult(
                        success=False,
                        output=output,
                        metadata={"return_code": process.returncode},
                        error=error_msg,
                    )

                return ToolResult(
                    success=False,
                    output="",
                    metadata={"return_code": process.returncode},
                    error=error_msg,
                )

            except subprocess.TimeoutExpired:
                logger.warning(f"Code execution timeout after {self.timeout_seconds}s")
                try:
                    if hasattr(os, "setsid"):
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=5)
                except Exception:
                    pass

                return ToolResult(
                    success=False,
                    output="",
                    metadata={"timeout": self.timeout_seconds},
                    error=f"Code execution timeout after {self.timeout_seconds} seconds",
                )

        except Exception as e:
            logger.error(f"Code execution error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                output="",
                metadata={},
                error=str(e),
            )

    def _trim_output(self, text: str) -> str:
        """Trim output to maximum length."""
        if not text:
            return ""
        if len(text) > self.max_output_chars:
            return text[:self.max_output_chars] + "\n...[truncated]"
        return text

    def _extract_error(self, stderr: str) -> str:
        """Extract concise error message from stderr."""
        if not stderr:
            return "Unknown error"

        # Try to find the actual error line (last line with a colon)
        lines = stderr.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line and ":" in line and not line.startswith("File "):
                return line[:150]

        # Return first 150 chars of stderr
        return stderr[:150]

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response, handling markdown fences."""
        code_block_pattern = r'```(?:python)?\n(.*?)\n```'
        matches = re.findall(code_block_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        return response.strip()

    def validate_args(self, **kwargs) -> bool:
        """Validate code execution arguments."""
        if self.direct_mode:
            code = kwargs.get('code', '')
            return isinstance(code, str) and len(code.strip()) > 0
        else:
            task = kwargs.get('task', '')
            return isinstance(task, str) and len(task.strip()) > 0

