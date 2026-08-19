"""Upload a local JSON file to W&B as a versioned artifact.

Standalone utility (not wired into the experiment pipeline): logs an
arbitrary JSON file to a W&B project as a ``dataset`` artifact, using the
``WANDB_API_KEY`` already configured in ``.env`` / the environment.

Usage:
    python scripts/upload_json_to_wandb.py path/to/file.json
    python scripts/upload_json_to_wandb.py path/to/file.json --project data-project --name prompts_export

Requires: wandb (already a project dependency; see pyproject.toml).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def upload_json_to_wandb(
    json_path: str,
    project: str = "data-project",
    artifact_name: str | None = None,
    entity: str | None = None,
) -> str:
    """Upload ``json_path`` to W&B as a ``dataset``-type artifact.

    Args:
        json_path: Path to the local JSON file to upload.
        project: W&B project to upload into.
        artifact_name: Artifact name (defaults to the file stem).
        entity: W&B entity/team (defaults to the account tied to the API key).

    Returns:
        The full W&B run URL for the upload.
    """
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    if path.suffix.lower() != ".json":
        raise ValueError(f"Expected a .json file, got: {path}")

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY is not set. Add it to .env or export it before running."
        )

    import wandb  # imported lazily so --help works without the dependency installed

    name = artifact_name or path.stem

    run = wandb.init(project=project, entity=entity, job_type="upload-json", name=f"upload-{name}")
    try:
        artifact = wandb.Artifact(name=name, type="dataset")
        artifact.add_file(str(path), name=path.name)
        run.log_artifact(artifact)
        artifact.wait()  # block until the upload finishes so we can confirm success
        run_url = run.url
    finally:
        run.finish()

    return run_url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", help="Path to the JSON file to upload")
    parser.add_argument("--project", default="data-project", help="W&B project (default: data-project)")
    parser.add_argument("--name", default=None, help="Artifact name (default: file stem)")
    parser.add_argument("--entity", default=None, help="W&B entity/team (default: your account)")
    args = parser.parse_args()

    url = upload_json_to_wandb(
        json_path=args.json_path,
        project=args.project,
        artifact_name=args.name,
        entity=args.entity,
    )
    print(f"Uploaded '{args.json_path}' -> {url}")


if __name__ == "__main__":
    main()
