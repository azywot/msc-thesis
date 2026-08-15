"""Root pytest configuration.

Some test modules exercise the RL fine-tuning stack and import packages that
live only in the ``cosmas-train`` environment (``agentops``, ``verl``).  In the
``agent_engine`` environment those imports raise ``ModuleNotFoundError`` during
*collection*, which aborts the entire run before a single test executes.

Skipping the affected modules when their dependency is absent keeps ``pytest``
from the repo root working in both environments.  This changes only which
modules are *collected*: every test that ran before still runs, and no test
outcome changes.  In the ``cosmas-train`` environment the dependency is present
and the module is collected as normal.
"""

import importlib.util

# Test-module filename fragment -> import that must be available to collect it.
_REQUIRES = {
    "test_fine_tuning_rollout": "agentops",
}


def pytest_ignore_collect(collection_path, config):
    """Skip collecting modules whose optional dependency is not installed."""
    name = collection_path.name
    for fragment, module in _REQUIRES.items():
        if fragment in name and importlib.util.find_spec(module) is None:
            return True
    return False
