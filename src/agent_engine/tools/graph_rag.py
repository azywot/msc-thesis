"""GraphRAG wrapper for mind map tool.

This module provides a wrapper around nano_graphrag for intelligent
retrieval in non-direct mode.  It mirrors the local-model configuration
used in multi-agent-tools: zero embeddings + a local LLM completion
function, so no OpenAI API key is required.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    from nano_graphrag import GraphRAG, QueryParam
    from nano_graphrag._utils import wrap_embedding_func_with_attrs
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False
    GraphRAG = None
    QueryParam = None
    wrap_embedding_func_with_attrs = None
    np = None  # type: ignore


# ---------------------------------------------------------------------------
# Zero-embedding function (mirrors MAT's _zero_embedding).
#
# nano_graphrag uses embeddings only for the initial vector-store lookup
# (find nearest entities to the query). With zero embeddings, that lookup
# is effectively uniform — but for a per-question mind map the graph is
# tiny (handful of entities), so the real signal comes from graph traversal
# and LLM synthesis.
#
# At this scale there are few entities anyway, so filtering
# by semantic similarity adds no value — the whole graph is retrieved. The
# LLM then synthesizes the answer from that context, so retrieval quality
# is unchanged. Real embeddings would require OpenAI; zeros avoid that.
# ---------------------------------------------------------------------------

def _make_zero_embedding():
    """Build a zero-embedding function compatible with nano_graphrag."""
    if not GRAPHRAG_AVAILABLE or wrap_embedding_func_with_attrs is None:
        return None

    @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
    async def _zero_embedding(texts):
        if not texts:
            return np.zeros((0, 1536), dtype=np.float32)
        return np.zeros((len(texts), 1536), dtype=np.float32)

    return _zero_embedding


_zero_embedding = _make_zero_embedding()


# ---------------------------------------------------------------------------
# Local-model completion function factory (mirrors MAT's _create_vllm_completion)
# ---------------------------------------------------------------------------

def _make_completion_func(model_provider):
    """Wrap a BaseModelProvider into an async completion function for nano_graphrag.

    nano_graphrag calls:  await func(prompt, system_prompt=None, history_messages=None, **kwargs)
    and expects a plain string back.
    """
    async def _complete(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages=None,
        **kwargs,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        prompt_text = model_provider.apply_chat_template(messages, use_thinking=False)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model_provider.generate([prompt_text])[0].text,
        )
        return result

    return _complete


# ---------------------------------------------------------------------------
# MindMapGraphRAG wrapper
# ---------------------------------------------------------------------------

class MindMapGraphRAG:
    """Wrapper for GraphRAG mind map functionality.

    Configures nano_graphrag to use a local LLM (via model_provider) and
    zero embeddings — matching the multi-agent-tools setup exactly.
    """

    def __init__(
        self,
        working_dir: str = "./local_mem",
        model_provider=None,
        ini_content: str = "",
    ):
        """Initialize GraphRAG mind map.

        Args:
            working_dir: Working directory for GraphRAG storage.
                         Should be ``<base_path>/question_<id>`` to match MAT naming.
            model_provider: BaseModelProvider instance used for entity extraction.
                            Required when no OpenAI key is available (the normal case).
            ini_content: Optional initial text to insert on first run.
        """
        if not GRAPHRAG_AVAILABLE:
            raise ImportError(
                "nano_graphrag is not installed. Install it with: "
                "pip install nano-graphrag==0.0.8.2"
            )

        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        kwargs: dict = {"working_dir": str(self.working_dir)}

        if model_provider is not None:
            completion = _make_completion_func(model_provider)
            kwargs.update(
                {
                    "best_model_func": completion,
                    "cheap_model_func": completion,
                    "best_model_max_async": 1,
                    "cheap_model_max_async": 1,
                    "enable_local": True,
                    "enable_naive_rag": False,
                    "embedding_func": _zero_embedding,
                }
            )

        self.graph_func = GraphRAG(**kwargs)

        if ini_content:
            self.graph_func.insert(ini_content)

    def insert(self, content: str):
        """Insert content into the graph."""
        if content and content.strip():
            self.graph_func.insert(content)

    def query(self, query: str, mode: str = "local") -> str:
        """Query the graph."""
        return self.graph_func.query(query, param=QueryParam(mode=mode))

    def graph_retrieval(self, query: str) -> str:
        """Retrieve from graph using local mode."""
        return self.query(query, mode="local")

    def process_community_report(self, json_path: Optional[str] = None) -> str:
        """Read and process community report JSON."""
        if json_path is None:
            json_path = self.working_dir / "kv_store_community_reports.json"
        else:
            json_path = Path(json_path)

        if not json_path.exists():
            return "No community reports available yet."

        with open(json_path, "r") as f:
            data = json.load(f)

        all_reports = []
        for community_id, community in data.items():
            report_string = community.get("report_string", "")
            if report_string:
                all_reports.append(f"Snippet {community_id}:\n{report_string}\n")

        if not all_reports:
            return "No community reports available."

        return "\n".join(all_reports)

    def __call__(self, query: str) -> str:
        """Callable interface for graph retrieval."""
        return self.graph_retrieval(query)
