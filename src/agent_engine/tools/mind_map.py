"""Mind map tool for memory management.

This tool maintains a text-based mind map of the reasoning process
for memory and context management across turns.
"""

import os
import re
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.tool import BaseTool, ToolResult
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Constants for persistent text file format
ENTRY_BEGIN = "=== ENTRY ==="
ENTRY_END = "=== END ==="

# Try to import GraphRAG
try:
    from .graph_rag import MindMapGraphRAG
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False
    MindMapGraphRAG = None
    logger.warning("GraphRAG not available - mind map will use simple keyword search")


class MindMapTool(BaseTool):
    """Mind map tool for tracking reasoning context.

    Maintains a simple text-based memory of key information,
    useful for long conversations with many turns.

    Two modes:
    - Direct mode: Persistent text file with op-based read/write operations
    - Non-direct mode (sub-agent): Query-only interface with GraphRAG support
    """

    def __init__(
        self,
        max_entries: int = 50,
        direct_mode: bool = True,
        storage_path: Optional[str] = None,
        use_graphrag: bool = True,
        model_provider: Optional[Any] = None,
    ):
        """Initialize mind map tool.

        Args:
            max_entries: Maximum number of entries to maintain
            direct_mode: If True, use persistent text files with op-based interface.
                        If False, use query-only interface with GraphRAG.
            storage_path: Base path for storing mind map files.
                          Per-question dirs are created as ``<storage_path>/question_<id>``,
                          matching the multi-agent-tools naming convention.
            use_graphrag: Whether to use GraphRAG in non-direct mode (requires nano_graphrag)
            model_provider: BaseModelProvider used for GraphRAG entity extraction.
                            Should be the planner (or a dedicated mind_map) model.
                            If None, nano_graphrag falls back to its default (OpenAI).
        """
        self.max_entries = max_entries
        self.direct_mode = direct_mode
        self.storage_path = storage_path or "./mind_map_storage"
        self.use_graphrag = use_graphrag and not direct_mode and GRAPHRAG_AVAILABLE
        self.model_provider = model_provider
        self.entries: Dict[str, list] = {}  # question_id -> list of entries (in-memory/fallback mode)
        self.current_question_id: Optional[int] = None
        self.graphrag_instances: Dict[int, Any] = {}  # question_id -> GraphRAG instance

        if self.use_graphrag:
            if model_provider is not None:
                logger.info("Mind map initialized with GraphRAG + local model provider")
            else:
                logger.warning("Mind map GraphRAG enabled but no model_provider given — will attempt OpenAI fallback")
        elif not direct_mode and not GRAPHRAG_AVAILABLE:
            logger.warning("GraphRAG not available - falling back to simple keyword search")

    @property
    def name(self) -> str:
        return "mind_map"

    @property
    def description(self) -> str:
        if self.direct_mode:
            return "Persistent notes - write to save information, read to retrieve it"
        else:
            return "Query the mind map of your previous reasoning to recall key information"

    def get_schema(self) -> Dict[str, Any]:
        """Return Qwen3 JSON Schema.

        Schema differs by mode:
        - Direct mode: op-based read/write operations
        - Non-direct mode: query-only interface
        """
        if self.direct_mode:
            # Direct mode: persistent text file with op="write"/"read"
            return {
                "type": "function",
                "function": {
                    "name": "mind_map",
                    "description": "Persistent notes for this question. Use op='write' to save key facts/decisions, and op='read' to retrieve them later (optionally by query).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "op": {
                                "type": "string",
                                "enum": ["write", "read"],
                                "description": "Operation: 'write' to store notes, 'read' to retrieve notes.",
                            },
                            "content": {
                                "type": "string",
                                "description": "Text to store (required for op='write').",
                            },
                            "query": {
                                "type": "string",
                                "description": "Optional search query for op='read'. If omitted, return latest notes.",
                            },
                        },
                        "required": ["op"],
                    }
                }
            }
        else:
            # Non-direct mode: query-only interface
            return {
                "type": "function",
                "function": {
                    "name": "mind_map",
                    "description": "Query the reasoning memory to retrieve relevant information from your previous thoughts and reasoning steps. Use this to recall context, avoid repetition, or build on previous conclusions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to search for in your reasoning memory. Ask specific questions about what you've reasoned about before."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }

    def execute(self, op: Optional[str] = None, query: Optional[str] = None, content: Optional[str] = None) -> ToolResult:
        """Execute mind map operation.

        Direct mode args:
            op: Operation to perform ("write" or "read")
            content: Text to store (required for op='write')
            query: Optional search query (for op='read')

        Non-direct mode args:
            query: Search query for mind map

        Returns:
            ToolResult with operation result
        """
        if self.direct_mode:
            return self._execute_direct_mode(op, content, query)
        else:
            return self._execute_query_mode(query)

    def _execute_direct_mode(self, op: Optional[str], content: Optional[str], query: Optional[str]) -> ToolResult:
        """Execute mind map in direct mode (persistent text files)."""
        if self.current_question_id is None:
            return ToolResult(
                success=False,
                output="Mind map not initialized for current question",
                metadata={},
                error="No active question"
            )

        # Normalize operation
        op_norm = (op or "").strip().lower()

        if op_norm == "write":
            return self._write_entry(content)
        elif op_norm == "read":
            return self._read_entries(query)
        else:
            return ToolResult(
                success=False,
                output="",
                metadata={"op": op},
                error=f"Unknown operation '{op}'. Use 'write' or 'read'."
            )

    def _execute_query_mode(self, query: Optional[str]) -> ToolResult:
        """Execute mind map in non-direct mode (query-only interface)."""
        if query is None:
            return ToolResult(
                success=False,
                output="",
                metadata={},
                error="Query parameter is required"
            )

        logger.info(f"Mind map query: {query}")

        if self.current_question_id is None:
            return ToolResult(
                success=False,
                output="Mind map not initialized for current question",
                metadata={"query": query},
                error="No active question"
            )

        # Use GraphRAG if available and enabled
        if self.use_graphrag:
            return self._query_with_graphrag(query)

        # Fall back to simple keyword search
        return self._query_with_keyword_search(query)

    def _query_with_graphrag(self, query: str) -> ToolResult:
        """Query using GraphRAG with timeout and length validation (mirrors MAT safe_mind_map_query)."""
        # Get or create GraphRAG instance for this question.
        # Working dir matches MAT: <storage_path>/question_<id> (no nested graphrag/ subdir)
        if self.current_question_id not in self.graphrag_instances:
            working_dir = Path(self.storage_path) / f"question_{self.current_question_id}"
            self.graphrag_instances[self.current_question_id] = MindMapGraphRAG(
                working_dir=str(working_dir),
                model_provider=self.model_provider,
            )

        graph = self.graphrag_instances[self.current_question_id]

        try:

            def _timeout_handler(signum, frame):
                raise TimeoutError("Mind map query timed out")

            if hasattr(signal, "SIGALRM"):
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(30)

            try:
                result = graph.graph_retrieval(query)
            finally:
                if hasattr(signal, "SIGALRM"):
                    signal.alarm(0)

            if result is None:
                return self._query_with_keyword_search(query)

            result = str(result).strip()
            original_length = len(result)

            # Reject suspiciously large results (likely raw graph data, not a summary)
            MAX_ACCEPTABLE = 5000
            if original_length > MAX_ACCEPTABLE:
                logger.warning(
                    f"GraphRAG result too long ({original_length} chars), falling back to keyword search"
                )
                return self._query_with_keyword_search(query)

            # Truncate if needed
            MAX_RESULT = 2000
            if original_length > MAX_RESULT:
                result = result[:MAX_RESULT] + "... [truncated]"

            return ToolResult(
                success=True,
                output=result,
                metadata={"query": query, "mode": "graphrag"},
            )
        except Exception as e:
            logger.error(f"GraphRAG query failed: {e}", exc_info=True)
            return self._query_with_keyword_search(query)

    def _query_with_keyword_search(self, query: str) -> ToolResult:
        """Query using simple keyword search (fallback)."""
        # Get entries for current question
        entries = self.entries.get(self.current_question_id, [])

        if not entries:
            return ToolResult(
                success=True,
                output="Mind map is empty. No previous reasoning recorded yet.",
                metadata={"query": query, "num_entries": 0, "mode": "keyword"}
            )

        # Simple keyword search
        query_lower = query.lower()
        relevant_entries = [
            entry for entry in entries
            if query_lower in entry.lower()
        ]

        if not relevant_entries:
            # Return all entries if no specific match
            output = "No specific matches found. Here's your full reasoning history:\n\n"
            output += "\n".join(f"- {entry}" for entry in entries[-10:])  # Last 10 entries
        else:
            output = f"Found {len(relevant_entries)} relevant entries:\n\n"
            output += "\n".join(f"- {entry}" for entry in relevant_entries)

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "query": query,
                "num_entries": len(entries),
                "num_relevant": len(relevant_entries),
                "mode": "keyword"
            }
        )

    def _get_question_dir(self, question_id: int) -> Path:
        """Get directory path for a question's mind map."""
        return Path(self.storage_path) / f"question_{question_id}"

    def _get_mind_map_file(self, question_id: int) -> Path:
        """Get file path for a question's mind map."""
        return self._get_question_dir(question_id) / "mind_map.txt"

    def _write_entry(self, content: Optional[str]) -> ToolResult:
        """Write an entry to persistent text file (direct mode)."""
        if not content or not content.strip():
            return ToolResult(
                success=True,
                output="mind_map write skipped: empty content.",
                metadata={"operation": "write"}
            )

        # Ensure directory exists
        question_dir = self._get_question_dir(self.current_question_id)
        question_dir.mkdir(parents=True, exist_ok=True)

        # Append entry to file
        mind_map_file = self._get_mind_map_file(self.current_question_id)
        block = f"{ENTRY_BEGIN}\n{content.rstrip()}\n{ENTRY_END}\n\n"

        with open(mind_map_file, "a", encoding="utf-8") as f:
            f.write(block)

        logger.info(f"Mind map write: appended {len(content)} chars to {mind_map_file}")

        return ToolResult(
            success=True,
            output=f"mind_map write ok: appended {len(content)} chars.",
            metadata={"operation": "write", "content_length": len(content)}
        )

    def _read_entries(self, query: Optional[str], max_chars: int = 4000, top_k: int = 5) -> ToolResult:
        """Read entries from persistent text file (direct mode)."""
        mind_map_file = self._get_mind_map_file(self.current_question_id)

        if not mind_map_file.exists():
            return ToolResult(
                success=True,
                output="mind_map is empty (no memory written yet).",
                metadata={"operation": "read", "num_entries": 0}
            )

        q = (query or "").strip()

        if not q:
            # No query: return tail of file
            output = self._read_text_tail(mind_map_file, max_chars)
            if not output or not output.strip():
                return ToolResult(
                    success=True,
                    output="mind_map is empty",
                    metadata={"operation": "read", "mode": "tail"}
                )
            return ToolResult(
                success=True,
                output=output,
                metadata={"operation": "read", "mode": "tail"}
            )

        # Query mode: parse and rank entries
        with open(mind_map_file, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        entries = self._parse_entries(text)
        if not entries:
            return ToolResult(
                success=True,
                output="mind_map has no parsable entries yet.",
                metadata={"operation": "read", "num_entries": 0}
            )

        # Score and rank entries
        q_tokens = self._tokenize(q)
        scored: List[Tuple[int, int]] = []  # (score, idx)
        for i, e in enumerate(entries):
            scored.append((self._score_entry(q_tokens, e), i))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Filter out zero-score entries unless everything is zero
        best_score = scored[0][0] if scored else 0
        picked = []
        for s, idx in scored:
            if s <= 0 and best_score > 0:
                continue
            picked.append(entries[idx])
            if len(picked) >= top_k:
                break

        output = "\n\n".join(picked).strip()
        if not output:
            # Fall back to tail
            output = self._read_text_tail(mind_map_file, max_chars)

        if len(output) > max_chars:
            output = output[:max_chars] + "\n...[truncated]"

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "operation": "read",
                "mode": "query",
                "query": query,
                "num_entries": len(entries),
                "num_returned": len(picked)
            }
        )

    def _read_text_tail(self, path: Path, max_chars: int) -> str:
        """Read tail of text file."""
        if max_chars <= 0:
            max_chars = 2000

        with open(path, "rb") as f:
            try:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                start = max(0, size - max_chars)
                f.seek(start)
                data = f.read()
            except Exception:
                f.seek(0)
                data = f.read()

        text = data.decode("utf-8", errors="replace").strip()
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text

    def _parse_entries(self, text: str) -> List[str]:
        """Parse ENTRY blocks from text."""
        if not text:
            return []

        # Capture blocks between ENTRY_BEGIN and ENTRY_END
        pattern = re.escape(ENTRY_BEGIN) + r"(.*?)" + re.escape(ENTRY_END)
        matches = list(re.finditer(pattern, text, flags=re.DOTALL))

        entries: List[str] = []
        for m in matches:
            raw_inner = (m.group(1) or "").strip("\n")
            raw_block = f"{ENTRY_BEGIN}\n{raw_inner}\n{ENTRY_END}"
            entries.append(raw_block)

        return entries

    def _tokenize(self, s: str) -> List[str]:
        """Tokenize string for keyword search."""
        return re.findall(r"[a-z0-9]+", (s or "").lower())

    def _score_entry(self, query_tokens: List[str], entry: str) -> int:
        """Score entry by keyword overlap."""
        if not query_tokens:
            return 0

        text = entry.lower()
        score = 0
        for tok in query_tokens:
            if tok and tok in text:
                score += 1

        return score

    def add_entry(self, text: str, question_id: int):
        """Add an entry to the mind map.

        This is called externally by the orchestrator to record reasoning steps.
        Only used in non-direct mode.

        Args:
            text: Text to add to mind map
            question_id: Question identifier
        """
        # In direct mode, entries are written explicitly via tool calls
        if self.direct_mode:
            return

        # Skip very short texts and tool calls
        if len(text) <= 20 or "<tool_call>" in text:
            return

        # Add to GraphRAG if enabled
        if self.use_graphrag:
            if question_id not in self.graphrag_instances:
                working_dir = Path(self.storage_path) / f"question_{question_id}"
                self.graphrag_instances[question_id] = MindMapGraphRAG(
                    working_dir=str(working_dir),
                    model_provider=self.model_provider,
                )

            try:
                self.graphrag_instances[question_id].insert(text)
            except Exception as e:
                logger.error(f"Failed to insert into GraphRAG: {e}", exc_info=True)

        # Also maintain in-memory entries as fallback
        if question_id not in self.entries:
            self.entries[question_id] = []

        self.entries[question_id].append(text[:500])  # Truncate long entries

        # Maintain max entries
        if len(self.entries[question_id]) > self.max_entries:
            self.entries[question_id] = self.entries[question_id][-self.max_entries:]

    def set_current_question(self, question_id: int):
        """Set the current question for mind map context.

        Args:
            question_id: Question identifier
        """
        self.current_question_id = question_id

    def clear_question(self, question_id: int):
        """Clear mind map for a specific question.

        Args:
            question_id: Question identifier
        """
        if question_id in self.entries:
            del self.entries[question_id]

    def cleanup(self):
        """Clear all mind map data."""
        self.entries.clear()
        self.graphrag_instances.clear()
        self.current_question_id = None
