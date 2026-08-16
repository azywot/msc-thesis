"""Tool-agnostic batching for sub-agent tools.

A sub-agent tool defers work: a cheap ``prepare`` step builds a job, all
prepared jobs are turned into prompts and generated in one batched LLM call,
then ``finalize`` turns each generation into a :class:`ToolResult`.  The
orchestrator previously hardcoded two copies of this shape -- ``_WebJob`` and
``_CodeJob``, with a matching pair of ``_flush_*``/``_run_*`` methods.  A new
batched sub-agent now implements this protocol instead of requiring surgery on
the core loop.

Behaviour notes carried over verbatim, each of which the obvious tidy-up would
break:

* **Flush order is by ``batch_priority`` ascending** (web 10, code 20).  This
  was hardcoded as "web then code" and it is observable: web analysis populates
  caches that a later code job in the same turn can read.
* **Only the generation's usage is accumulated**, never the finalize
  ``ToolResult``'s.  The immediate path does the opposite.  Unifying them
  double-counts sub-agent tokens.
* **Grouping is by ``id(tool.model_provider)``** -- identity, not equality, and
  in first-seen order.
* **``finalize`` owns its own output cleaning.**  ``flush_batches`` commits
  ``result.output`` untouched.  The two original paths differed here: the web
  path committed already-stripped analysis text, while the code path stripped
  the *executed* result.  Stripping centrally would strip the web path twice,
  and ``strip_thinking_tags`` is not idempotent on text with two orphaned
  ``</think>`` markers.
* **A tool with no provider silently drops its jobs.**  ``generate`` is skipped
  and ``zip`` yields nothing, so nothing is committed.  Preserved as-is.
"""

from typing import Any, Dict, List, NamedTuple, Protocol, runtime_checkable

from .tool import BaseTool, ToolResult


class BatchJob(NamedTuple):
    """One deferred unit of work.

    ``payload`` carries whatever the tool needs between ``prepare`` and
    ``finalize`` (the web tool keeps its query and search results there; the
    code tool keeps its built prompt).
    """

    state: Any
    tool_call: Dict[str, Any]
    tool: BaseTool
    payload: Dict[str, Any]


@runtime_checkable
class BatchedTool(Protocol):
    """A tool whose work is deferred and batched across states."""

    batch_priority: int

    def prepare(self, state, tool_call, args) -> Any:
        """Return a :class:`BatchJob` to defer, or a ``ToolResult`` to
        short-circuit (bad arguments, a cache hit, a failure while preparing)."""

    def batch_prompt(self, job: BatchJob) -> str:
        """The prompt for this job's batched generation."""

    def finalize(self, job: BatchJob, generation) -> ToolResult:
        """Turn one generation into the result to commit, output already clean."""


def flush_batches(jobs_by_tool, commit, accumulate_usage) -> None:
    """Run every deferred job, in the order the previous implementation used.

    ``commit(state, tool_call, text)`` and ``accumulate_usage(state, usage)``
    are injected so this module never imports the orchestrator.
    """
    ordered = sorted(
        jobs_by_tool.items(),
        key=lambda kv: getattr(kv[1][0].tool, "batch_priority", 100),
    )

    for _tool_name, jobs in ordered:
        if not jobs:
            continue

        # Optional hook, run once over ALL of this tool's jobs before grouping:
        # the web tool fetches every URL across every job in one pass here.
        pre = getattr(jobs[0].tool, "pre_batch", None)
        if pre is not None:
            pre(jobs)

        groups: Dict[int, List[BatchJob]] = {}
        for job in jobs:
            groups.setdefault(id(getattr(job.tool, "model_provider", None)), []).append(job)

        for group in groups.values():
            provider = getattr(group[0].tool, "model_provider", None)
            # Per-job tool lookup, not group[0].tool: a group is keyed by
            # provider identity, so in principle it can mix tool instances.
            prompts = [job.tool.batch_prompt(job) for job in group]
            outputs = provider.generate(prompts) if provider else []
            for job, out in zip(group, outputs):
                accumulate_usage(job.state, out.usage)
                result = job.tool.finalize(job, out)
                commit(job.state, job.tool_call, result.output or "")
