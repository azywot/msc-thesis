"""Launch the VERL training server.

Mirrors AgentFlow's train/train_agent.py: reads train/config.yaml, sets
environment variables, and spawns `python -m agentflow.verl key=value ...`.

Usage:
    python scripts/launch_verl.py --config train/config.yaml
"""

import argparse
import os
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description="Launch VERL training server.")
    parser.add_argument("--config", type=str, default="train/config.yaml")
    args, unknown = parser.parse_known_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set environment variables from config.env
    for key, value in config.get("env", {}).items():
        os.environ[key] = str(value)
        print(f"  Exported {key}={value}")

    # Build: python -m agentflow.verl key=value key=value ...
    command = [sys.executable, "-m", "agentflow.verl"]
    for key, value in config.get("python_args", {}).items():
        if isinstance(value, str):
            expanded = os.path.expandvars(value)
            command.append(f"{key}={expanded}")
        else:
            command.append(f"{key}={value}")
    command.extend(unknown)

    print("Launching VERL server:")
    print(" ".join(str(x) for x in command))
    print("-" * 60)

    try:
        subprocess.run(command, check=True, env=os.environ)
    except subprocess.CalledProcessError as e:
        print(f"VERL server exited with code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
