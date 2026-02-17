"""GraphRAG wrapper for mind map tool.

This module provides a wrapper around nano_graphrag for intelligent
retrieval in non-direct mode.
"""

import json
from pathlib import Path
from typing import Optional

try:
    from nano_graphrag import GraphRAG, QueryParam
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False
    GraphRAG = None
    QueryParam = None


class MindMapGraphRAG:
    """Wrapper for GraphRAG mind map functionality.

    This class manages a GraphRAG instance for storing and querying
    reasoning history in an intelligent way.
    """

    def __init__(self, ini_content: str = "", working_dir: str = "./local_mem"):
        """Initialize GraphRAG mind map.

        Args:
            ini_content: Initial content to insert into the graph
            working_dir: Working directory for GraphRAG storage
        """
        if not GRAPHRAG_AVAILABLE:
            raise ImportError(
                "nano_graphrag is not installed. Install it with: "
                "pip install nano-graphrag==0.0.8.2"
            )

        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.graph_func = GraphRAG(working_dir=str(self.working_dir))

        # Insert initial content if provided
        if ini_content:
            self.graph_func.insert(ini_content)

    def insert(self, content: str):
        """Insert content into the graph.

        Args:
            content: Content to insert
        """
        if content and content.strip():
            self.graph_func.insert(content)

    def query(self, query: str, mode: str = "local") -> str:
        """Query the graph.

        Args:
            query: Query string
            mode: Query mode ("local" or "global")

        Returns:
            Query result as string
        """
        return self.graph_func.query(query, param=QueryParam(mode=mode))

    def graph_retrieval(self, query: str) -> str:
        """Retrieve from graph using local mode.

        Args:
            query: Query string

        Returns:
            Retrieval result
        """
        return self.query(query, mode="local")

    def graph_query(self, query: str) -> str:
        """Query graph and return community reports.

        Args:
            query: Query string

        Returns:
            Combined community reports
        """
        combined_report = self.process_community_report()
        return combined_report

    def process_community_report(self, json_path: Optional[str] = None) -> str:
        """Read and process community report JSON.

        Args:
            json_path: Path to community reports JSON file.
                      If None, uses default location in working_dir.

        Returns:
            Combined report string
        """
        if json_path is None:
            json_path = self.working_dir / "kv_store_community_reports.json"
        else:
            json_path = Path(json_path)

        if not json_path.exists():
            return "No community reports available yet."

        # Read JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Collect all report strings from each community
        all_reports = []
        for community_id, community in data.items():
            report_string = community.get("report_string", "")
            if report_string:
                all_reports.append(f"Snippet {community_id}:\n{report_string}\n")

        # Combine all reports
        if not all_reports:
            return "No community reports available."

        return "\n".join(all_reports)

    def __call__(self, query: str) -> str:
        """Callable interface for querying.

        Args:
            query: Query string

        Returns:
            Query result
        """
        return self.graph_retrieval(query)
