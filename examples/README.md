# examples/

Single-tool sanity-check scripts. For prerequisites, run instructions, and
how these relate to the full experiment runner see the
[main README](../README.md#examples).

## Files

| Script | Tool tested |
|---|---|
| `example_web_search.py` | `web_search` |
| `example_code_generator.py` | `code_generator` |
| `example_text_inspector.py` | `text_inspector` |
| `example_image_inspector.py` | `image_inspector` |
| `example_context_manager.py` | `web_search` + `context_manager` |
| `_common.py` | Shared helpers (model init, tool wiring, orchestrator factory) |
| `fixtures/sample_document.txt` | Synthetic document used by `example_text_inspector.py` |
| `fixtures/make_test_image.py` | Helper to regenerate the test bar-chart PNG |

## Run

From the `msc-thesis/` root:

```bash
python examples/example_web_search.py
python examples/example_code_generator.py
python examples/example_text_inspector.py
python examples/example_image_inspector.py
python examples/example_context_manager.py
```

Output is saved to `experiments/results/examples/<tool_name>/`.
