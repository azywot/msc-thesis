# Examples — agent_engine (sub-agent mode)

Each script tests a **single tool** in sub-agent mode using the shared config at
`experiments/configs/gaia/test_subagent.yaml`.  They are designed to force the
model to actually call the tool under test rather than answering from its weights.

All scripts should be run from the `msc-thesis/` root directory.

---

## Prerequisites

```bash
# Required for web_search
export SERPER_API_KEY="<your-key>"

# Required for all examples (models loaded via vLLM)
# Make sure HF_HOME / TRANSFORMERS_CACHE points to a cache with Qwen/Qwen3-4B
```

---

## Files

| File | Tool tested | Question type |
|---|---|---|
| `example_web_search.py` | `web_search` | Recent real-world fact the model cannot know from training data |
| `example_code_generator.py` | `code_generator` | Multi-part maths problem requiring actual code execution |
| `example_text_inspector.py` | `text_inspector` | Five specific questions about a local document (`fixtures/sample_document.txt`) |
| `example_image_inspector.py` | `image_inspector` | Visual questions about a bar-chart PNG (generated on the fly) |
| `example_mind_map.py` | `mind_map` | Multi-step task requiring storing facts and then querying them back |

Support files:

| File | Purpose |
|---|---|
| `_common.py` | Shared helpers (model init, tool wiring, orchestrator factory, prompt builder) — mirrors `scripts/run_experiment.py` |
| `simple_example.py` | Original minimal example (web_search + code_generator, direct mode) |
| `fixtures/sample_document.txt` | Synthetic annual report used by the text-inspector example |
| `fixtures/make_test_image.py` | Standalone helper to regenerate the bar-chart PNG manually |

---

## Running the examples

```bash
# Web search
python examples/example_web_search.py

# Code generation
python examples/example_code_generator.py

# Text inspection (reads fixtures/sample_document.txt)
python examples/example_text_inspector.py

# Image inspection (generates fixtures/test_chart.png automatically)
python examples/example_image_inspector.py

# Mind map (store → query flow)
python examples/example_mind_map.py
```

Each script saves its output under `experiments/results/examples/<tool_name>/`:
- `result.json` — question, answer, turns used, tool call counts
- `trace.json` — full execution state (all messages, tool_calls) for debugging
- `example.log` — execution log including a readable trace (every message and tool call with arguments). For the code_generator example, the log also includes the generated code and the temp file path where it was written before execution.

---

## How it relates to the full experiment runner

The examples use `orchestrator.run()` (single question) instead of
`orchestrator.run_batch()`. Everything else — model caching, tool constructor
arguments, `thinking_mode`, `CacheManager`, `PromptBuilder` — is identical to
`scripts/run_experiment.py`. For a single question `run()` and `run_batch(batch_size=1)`
produce equivalent results.

The config field `tools.enabled_tools` in the YAML is **overridden** in each
example so that only the one tool under test is registered, even though the YAML
lists all five tools.
