"""Analyze experiment results.

This script provides utilities for analyzing and visualizing experiment results.
"""

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any


def load_results(results_path: Path) -> List[Dict[str, Any]]:
    """Load results from JSON file.

    Args:
        results_path: Path to results JSON file

    Returns:
        List of result dictionaries
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_accuracy(results: List[Dict[str, Any]]) -> float:
    """Compute overall accuracy.

    Args:
        results: List of results

    Returns:
        Accuracy (0-1)
    """
    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    return correct / total if total > 0 else 0.0


def analyze_by_level(results: List[Dict[str, Any]]):
    """Analyze results by difficulty level (for GAIA).

    Args:
        results: List of results
    """
    levels = {}
    for result in results:
        level = result.get("metadata", {}).get("level", "unknown")
        if level not in levels:
            levels[level] = {"total": 0, "correct": 0}
        levels[level]["total"] += 1
        if result.get("correct", False):
            levels[level]["correct"] += 1

    print("\nAccuracy by Level:")
    print("-" * 40)
    for level in sorted(levels.keys()):
        stats = levels[level]
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  Level {level}: {acc:.1%} ({stats['correct']}/{stats['total']})")


def analyze_tool_usage(results: List[Dict[str, Any]]):
    """Analyze tool usage statistics.

    Args:
        results: List of results
    """
    tool_counts = Counter()
    total_turns = 0

    for result in results:
        turns = result.get("turns", 0)
        total_turns += turns

        for tool, count in result.get("tool_counts", {}).items():
            tool_counts[tool] += count

    print("\nTool Usage:")
    print("-" * 40)
    for tool, count in tool_counts.most_common():
        avg = count / len(results) if results else 0
        print(f"  {tool}: {count} total, {avg:.2f} avg/question")

    avg_turns = total_turns / len(results) if results else 0
    print(f"\nAverage turns per question: {avg_turns:.2f}")


def print_summary(results: List[Dict[str, Any]]):
    """Print overall summary.

    Args:
        results: List of results
    """
    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    accuracy = correct / total if total > 0 else 0

    print("="*60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*60)
    print(f"Total examples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("results", type=str,
                       help="Path to results JSON file")
    parser.add_argument("--by-level", action="store_true",
                       help="Show accuracy by difficulty level")
    parser.add_argument("--tools", action="store_true",
                       help="Show tool usage statistics")
    args = parser.parse_args()

    results_path = Path(args.results)

    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return

    print(f"Loading results from: {results_path}")
    results = load_results(results_path)

    print_summary(results)

    if args.by_level:
        analyze_by_level(results)

    if args.tools:
        analyze_tool_usage(results)


if __name__ == "__main__":
    main()
