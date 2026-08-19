"""Check that verl will accept the Prefix-RFT extensions at runtime.

Three contracts, one question: subclassing verl's classes and copying its methods is not
enough — verl's *runtime* imposes rules that no import error and no unit test reveals.
Each check below exists because it did not exist, and a GPU allocation was spent
discovering the rule the hard way.

  1. Ray method binding (job 25751449)
     WorkerGroup._bind_worker_method walks dir(cls) and binds ONLY the methods carrying
     the marker @register attaches. Overriding a registered method without re-applying
     the decorator does not merely fail to register the new one — it REMOVES the method
     from the worker's remote interface:
         AttributeError: 'RayWorkerGroup' object has no attribute 'init_model'
     A changed dispatch_mode is worse: it binds, runs on the wrong ranks, and never
     raises. Both are checked.

  2. Config dataclass conversion (job 25751544)
     init_model converts config subtrees into typed dataclasses (fsdp_workers.py:922),
     and those reject keys they do not declare. A config Hydra composes perfectly can
     still kill every worker at startup:
         TypeError: FSDPActorConfig.__init__() got an unexpected keyword argument
                    'prefix_entropy_keep_ratio'
     Extra keys directly on actor_rollout_ref are fine — verl never converts that level,
     which is exactly why prefix_entropy_keep_ratio lives there.

  3. Name resolution in copied bodies (job 25752267)
     A copied method carries its original's dependencies but not its original's imports.
     Python resolves globals at call time, so a missing one waits inside the function
     until its line runs — for _train_step, the first training step, after Ray, both
     GPUs and a full rollout phase have all succeeded:
         NameError: name 'compute_response_mask' is not defined

None of this needs Ray, a GPU or model weights. It is seconds of CPU against hours of
allocation. Companion to `launch_verl.py --dry-run`, which proves Hydra can COMPOSE the
overrides; this proves verl will ACCEPT the result.

Usage:
    python scripts/check_prefix_rft_runtime_contracts.py
    python scripts/check_prefix_rft_runtime_contracts.py --config <experiment config>.yaml
"""

from __future__ import annotations

import argparse
import ast
import builtins
import importlib
import subprocess
import sys
from pathlib import Path

DEFAULT_CONFIGS = (
    "experiments/configs/fine_tuning/config_prefix_rft_tiny8b.yaml",
    "experiments/configs/fine_tuning/config_prefix_rft_smoke8b.yaml",
    "experiments/configs/fine_tuning/config_prefix_rft.yaml",
)

# Subtree -> the dataclass verl converts it into. None means "verl infers the type".
CONVERSIONS = (
    ("actor_rollout_ref.actor", None),
    ("actor_rollout_ref.rollout", None),
    ("actor_rollout_ref.model", "HFModelConfig"),
)

# (module holding the copy, method name, "<edits module>.<extractor>")
COPIES = (
    ("verl_ext.prefix_rft.trainer", "_train_step", "trainer_edits.actual_prefix_rft_train_step"),
    ("verl_ext.prefix_rft.actor", "update_policy", "actor_edits.actual_prefix_rft_update_policy"),
    ("verl_ext.prefix_rft.daemon", "_async_set_up", "daemon_edits.actual_prefix_rft_async_set_up"),
)


# ── 1. Ray method binding ────────────────────────────────────────────────────────


def bound_methods(cls) -> dict:
    """The methods verl would bind, by the same rule verl uses."""
    from verl.single_controller.base.decorator import MAGIC_ATTR

    found = {}
    for name in dir(cls):
        try:
            method = getattr(cls, name)
        except Exception:
            continue
        if callable(method) and hasattr(method, MAGIC_ATTR):
            found[name] = getattr(method, MAGIC_ATTR)
    return found


def check_worker_binding() -> bool:
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

    from verl_ext.prefix_rft.worker import PrefixRFTWorker

    parent = bound_methods(AsyncActorRolloutRefWorker)
    child = bound_methods(PrefixRFTWorker)
    print(f"  verl exposes {len(parent)} registered methods; PrefixRFTWorker {len(child)}")

    ok = True
    missing = sorted(set(parent) - set(child))
    if missing:
        ok = False
        for name in missing:
            print(f"  LOST: {name} — overridden without @register?")
        print(
            "  Re-apply verl's decorator to the override, e.g.\n"
            "      @register(dispatch_mode=Dispatch.ONE_TO_ALL)\n"
            "      def init_model(self): ..."
        )

    for name in sorted(set(parent) & set(child)):
        if parent[name] != child[name]:
            ok = False
            print(f"  DISPATCH CHANGED: {name}\n      verl: {parent[name]}\n      ours: {child[name]}")
            print("  This binds and runs, but on the wrong ranks.")

    if ok and "init_model" in child:
        print(f"  init_model binds as {child['init_model']['dispatch_mode']}")
    elif "init_model" not in child:
        ok = False
        print("  init_model is not bound; trainer.init_workers() would crash.")
    return ok


# ── 2. Config dataclass conversion ───────────────────────────────────────────────


def overrides_for(config_path: str) -> list[str]:
    """Ask launch_verl.py for the exact override list it would pass."""
    result = subprocess.run(
        [sys.executable, "scripts/launch_verl.py", "--config", config_path, "--dry-run"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"FAILED: {config_path} does not even compose.\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    for line in result.stdout.splitlines():
        if " -m verl_ext.prefix_rft " in line or " -m fine_tuning.agentflow.verl " in line:
            parts = line.split()
            return [p for p in parts[parts.index("-m") + 2 :] if p not in ("--cfg", "job")]
    raise SystemExit(f"FAILED: could not find the launch command in the dry run of {config_path}")


def check_config(config_path: str) -> bool:
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf
    from verl.utils.config import omega_conf_to_dataclass

    overrides = overrides_for(config_path)
    with initialize_config_module(config_module="verl_ext.prefix_rft.config", version_base=None):
        cfg = compose(config_name="prefix_rft_trainer", overrides=overrides)

    ok = True
    for dotted, dataclass_name in CONVERSIONS:
        node = OmegaConf.select(cfg, dotted)
        if node is None:
            print(f"    {dotted}: absent, skipped")
            continue
        try:
            if dataclass_name == "HFModelConfig":
                from verl.workers.config import HFModelConfig

                omega_conf_to_dataclass(node, dataclass_type=HFModelConfig)
            else:
                omega_conf_to_dataclass(node)
            print(f"    {dotted}: converts cleanly")
        except Exception as exc:  # noqa: BLE001 - the message is the whole point
            ok = False
            print(f"    {dotted}: FAILED -> {type(exc).__name__}: {exc}")
            print(
                "      A key here is not declared by verl's dataclass. Move it up to\n"
                "      actor_rollout_ref (never converted), or drop it."
            )
    return ok


# ── 3. Name resolution in copied bodies ──────────────────────────────────────────


def global_names(source: str) -> set[str]:
    """Every name the function body reads that it does not itself bind.

    Deliberately conservative: comprehension and except-handler targets count as bound,
    and attribute access only counts its root (``torch`` in ``torch.zeros``).
    """
    func = ast.parse(source.strip()).body[0]
    bound: set[str] = set()
    read: set[str] = set()

    for arg in list(getattr(func.args, "args", [])) + list(getattr(func.args, "kwonlyargs", [])):
        bound.add(arg.arg)
    for extra in ("vararg", "kwarg"):
        node = getattr(func.args, extra, None)
        if node is not None:
            bound.add(node.arg)

    for node in ast.walk(func):
        if isinstance(node, ast.Name):
            (bound if isinstance(node.ctx, (ast.Store, ast.Del)) else read).add(node.id)
        elif isinstance(node, ast.alias):
            bound.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)

    return read - bound - set(dir(builtins))


def check_copied_names() -> bool:
    from verl_ext.prefix_rft import actor_edits, daemon_edits, trainer_edits

    modules = {
        "trainer_edits": trainer_edits,
        "actor_edits": actor_edits,
        "daemon_edits": daemon_edits,
    }

    ok = True
    for module_path, method, extractor in COPIES:
        edits_module, func_name = extractor.split(".")
        source = getattr(modules[edits_module], func_name)()
        module = importlib.import_module(module_path)
        names = global_names(source)
        missing = sorted(n for n in names if not hasattr(module, n))
        print(f"  {module_path}.{method}: {len(names)} global names, {len(missing)} unresolved")
        for name in missing:
            ok = False
            print(f"    UNRESOLVED: {name} — raises NameError the first time its line runs")
    if not ok:
        print(
            "  Add the import to the module holding the copy. Check what the ORIGINAL "
            "module imports it from, so the copy keeps using the same object."
        )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", dest="configs")
    args = parser.parse_args()
    configs = args.configs or list(DEFAULT_CONFIGS)

    print("── 1. Ray binds every method the driver calls ──────────────────────")
    ok_binding = check_worker_binding()

    print("\n── 2. verl's dataclasses accept every launch config ────────────────")
    ok_configs = True
    for config_path in configs:
        if not Path(config_path).exists():
            print(f"  {config_path}: MISSING")
            ok_configs = False
            continue
        print(f"  {config_path}")
        ok_configs &= check_config(config_path)

    print("\n── 3. Copied methods resolve every name they use ───────────────────")
    ok_names = check_copied_names()

    failed = [
        label
        for label, ok in (
            ("Ray method binding", ok_binding),
            ("config dataclass conversion", ok_configs),
            ("name resolution in copies", ok_names),
        )
        if not ok
    ]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}. verl would reject this at runtime.")
        return 1
    print("\nPASSED: all three runtime contracts hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
