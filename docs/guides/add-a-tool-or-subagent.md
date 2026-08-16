# Add a tool or sub-agent

A tool is anything the orchestrator can call by emitting a tool call. Adding one
means writing a class and a factory. **You do not edit the orchestrator**, and
there is no dispatch chain to extend — if you find yourself adding a name check
to `core/`, stop and re-read this page.

> This guide was executed end to end against the code it describes, by building
> the `echo` tool below, running it in a real experiment, and deleting it
> afterwards. The transcript of what that run printed is at the bottom.

---

## 1. Write the tool class

Subclass `BaseTool` (`src/agent_engine/core/tool.py`). Four members are
required: `name`, `description`, `get_schema()`, `execute()`.

`src/agent_engine/tools/echo.py`:

```python
"""A minimal tool, kept only as a worked example."""

from ..core.tool import BaseTool, ToolResult
from .registry import register_tool


class EchoTool(BaseTool):
    def __init__(self, prefix="ECHO"):
        self.prefix = prefix

    @property
    def name(self):
        return "echo"

    @property
    def description(self):
        return "Echo a message back. Useful only as an example."

    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "echo",
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Text to echo"},
                    },
                    "required": ["message"],
                },
            },
        }

    def execute(self, message=None, **kwargs):
        if not message:
            return ToolResult(success=False, output="", error="Missing required echo arguments")
        return ToolResult(success=True, output=f"{self.prefix}: {message}", metadata={})
```

Notes that will save you an hour:

- `get_schema()` returns the **nested** `{"type": "function", "function": {...}}`
  shape. The tool's own `name` must match `function.name` and must match the
  name you register under; three-way agreement is asserted by
  `tests/unit/test_wiring_invariants.py`.
- `execute()` returns a `ToolResult`, never raises for ordinary bad input. A
  failed `ToolResult` becomes the tool's output text and the model gets a chance
  to recover; an exception ends the question.
- `execute()` should accept `**kwargs`. The model will eventually pass an
  argument you did not declare.

## 2. Register a factory

In the same file, after the class:

```python
@register_tool("echo")
def build_echo(deps) -> EchoTool:
    return EchoTool(prefix="ECHO")
```

`deps` is a `ToolDeps` (`tools/registry.py`) carrying everything a factory may
use:

| Member | Use |
|---|---|
| `deps.config` | The whole `ExperimentConfig`. |
| `deps.direct_mode` | `config.tools.direct_tool_call`. |
| `deps.provider_for("echo")` | The sub-agent model for this tool, or `None` in direct mode. |
| `deps.use_subagent_thinking` | Whether sub-agents emit `<think>`. |
| `deps.cache_manager`, `deps.api_keys` | Cache and API keys. |
| `deps.mind_map_storage_path` | Only `mind_map` uses this. |

A factory may return `None` to decline (`image_inspector` does this in direct
mode); the tool is then simply absent from the registry.

The decorator raises on a duplicate name, so a copy-pasted registration fails
loudly at import rather than silently shadowing another tool.

## 3. Make sure the module is imported

Registration happens as an import side effect. `runner/tools.py` imports the
`tools` package, so the tool must be reachable from
`src/agent_engine/tools/__init__.py`:

```python
from .echo import EchoTool  # noqa: F401
```

If you skip this, `build_tool("echo", deps)` returns `None` and the tool is
silently missing — the same behaviour as a typo'd name. Symptom: the model
never calls your tool and nothing in the log explains why.

## 4. Enable it in a config

```yaml
tools:
  enabled_tools: [echo]
  direct_tool_call: true
```

Verify without a GPU:

```bash
python -c "
from agent_engine.tools.registry import registered_tools, build_tool
print(registered_tools())
"
```

`echo` should appear in the list.

---

## Making it a sub-agent (batched)

Everything above gives you a tool that runs inline. A **sub-agent** additionally
calls an LLM to interpret its own output, and that call should be batched with
the other questions in the turn — otherwise a 200-question run makes 200
separate generation calls.

Implement the `BatchedTool` protocol (`core/batching.py`). It is
`runtime_checkable`, so there is nothing to register: implement the methods and
the orchestrator picks it up.

```python
class EchoTool(BaseTool):
    batch_priority = 30          # lower flushes first; web=10, code=20

    def __init__(self, model_provider=None, direct_mode=True):
        self.model_provider = model_provider
        self.direct_mode = direct_mode

    def prepare(self, state, tool_call, args):
        """Non-LLM work. Return a BatchJob to defer, or a ToolResult to finish now."""
        message = args.get("message")
        if not message:
            return ToolResult(success=False, output="", error="Missing required echo arguments")
        return BatchJob(state, tool_call, self, {"message": message})

    def pre_batch(self, jobs):
        """Optional. Cross-job work, once per turn — bulk I/O belongs here."""

    def batch_prompt(self, job):
        return self.model_provider.apply_chat_template(
            [{"role": "user", "content": f"Summarise: {job.payload['message']}"}],
            use_thinking=False,
        )

    def finalize(self, job, generation):
        return ToolResult(success=True, output=strip_thinking_tags(generation.text), metadata={})
```

Four things the existing tools learned the hard way:

1. **`batch_priority` is ordering, and ordering matters.** `web_search` (10)
   must flush before `code_generator` (20), because a web analysis populates a
   cache a code job in the same turn can read. Pick a number relative to that.
2. **Clean your output inside `finalize`.** `flush_batches` commits
   `result.output` untouched, on purpose: `strip_thinking_tags` is not
   idempotent on text with two orphaned `</think>` markers, so a central strip
   corrupts the web path.
3. **`direct_mode` must be an attribute**, because `_is_batched` reads
   `getattr(tool, "direct_mode", True)`. A batched tool in direct mode falls
   back to the immediate path automatically.
4. **`prepare` returning a `ToolResult` is the short-circuit**, used for missing
   arguments, cache hits, and failures. Only return a `BatchJob` when there is
   genuinely an LLM call to make.

Jobs are grouped by `id(model_provider)`, so two tools sharing a provider share
one generation call.

---

## 5. Test it

Batched tools need their own unit tests: the orchestrator trace fixture drives
*fake* tools, so it will stay green even if your `prepare`/`finalize` logic is
wrong. `tests/unit/test_batched_tools.py` is the model to copy.

```bash
pytest tests/unit/test_batched_tools.py tests/unit/test_wiring_invariants.py -q
```

---

## Walkthrough transcript

This guide was executed exactly as written. Building `echo.py` with the factory
but **skipping step 3**:

```
registered: ['code_generator', 'image_inspector', 'mind_map', 'text_inspector', 'web_search']
```

No error, no warning — just absent. After adding the import to
`tools/__init__.py`:

```
registered: ['code_generator', 'echo', 'image_inspector', 'mind_map', 'text_inspector', 'web_search']
built: echo | ECHO: hello
missing arg -> Missing required echo arguments
schema name matches: True
```

Then driven through the real `AgenticOrchestrator` with a scripted model
provider (no GPU needed — script the generations and the rest of the loop is
real):

```
answer         : done
tool calls     : ['echo']
tool_counts    : {'echo': 1}
action_history : echo | Echo the phrase | ECHO: hello
```

> **Gotcha the walkthrough exposed.** In AgentFlow mode the orchestrator runs a
> **planning turn before turn 1**, and any tool call in the planning output is
> parsed and *discarded* ("produced tool call (discarded)" in the log). A
> scripted provider therefore needs three responses — plan, tool call, answer —
> not two. Getting this wrong looks like the tool never being called: the answer
> comes back correct and `tool_calls` is empty. In baseline mode there is no
> planning turn and the first response is the tool call.

And the batched variant, confirming the protocol dispatch:

```
BatchedEcho satisfies protocol : True
plain EchoTool does not        : True
_is_batched(sub-agent mode)    : True
_is_batched(direct mode)       : False
_is_batched(plain tool)        : False
```

The tool was deleted afterwards, which is why `echo.py` is not in the tree. If
you add one and it does *not* appear in `registered_tools()`, the cause is
almost always step 3.
