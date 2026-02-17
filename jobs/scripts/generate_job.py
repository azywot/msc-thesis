"""Generate SLURM job file from experiment configuration.

This script reads an experiment YAML config and generates a corresponding
SLURM job file using Jinja2 templates.
"""

import argparse
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# Allow running without `pip install -e .` (e.g., on login node)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_engine.config import load_experiment_config


def generate_job(config_path: Path, output_path: Path = None, project_dir: Path = None):
    """Generate SLURM job file from experiment config.

    Args:
        config_path: Path to experiment YAML config
        output_path: Path for output job file (optional)
        project_dir: Project root directory (optional)
    """
    # Load experiment config (applies schema defaults for missing fields)
    config = load_experiment_config(config_path)

    # Extract job parameters
    experiment_name = config.name

    # Determine GPU requirements from model configs
    num_gpus = 1
    for model_cfg in config.models.values():
        tensor_parallel = getattr(model_cfg, "tensor_parallel_size", 1) or 1
        gpu_ids = getattr(model_cfg, "gpu_ids", None) or []
        num_gpus = max(num_gpus, tensor_parallel, len(gpu_ids))

    # Get other parameters
    seed = config.seed
    output_dir = str(config.output_dir)
    conda_env = "agent_engine"

    # Project directory
    if project_dir is None:
        project_dir = Path.cwd()

    # Output path
    if output_path is None:
        output_path = Path("jobs/generated") / f"{experiment_name}.job"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load Jinja2 template
    template_dir = Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("experiment.job.j2")

    # Render template
    job_content = template.render(
        experiment_name=experiment_name,
        num_gpus=num_gpus,
        seed=seed,
        config_path=str(config_path.absolute()),
        output_dir=output_dir,
        time_limit="04:00:00",
        project_dir=str(project_dir.absolute()),
        conda_env=conda_env,
    )

    # Write job file
    with open(output_path, 'w') as f:
        f.write(job_content)

    print(f"Generated SLURM job file: {output_path}")
    print(f"  Experiment: {experiment_name}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Config: {config_path}")
    print(f"\nTo submit: sbatch {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate SLURM job file from experiment config")
    parser.add_argument("config", type=str,
                       help="Path to experiment config YAML file")
    parser.add_argument("--output", "-o", type=str,
                       help="Output path for job file (default: jobs/generated/<name>.job)")
    parser.add_argument("--project-dir", type=str,
                       help="Project root directory (default: current directory)")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_path = Path(args.output) if args.output else None
    project_dir = Path(args.project_dir) if args.project_dir else None

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    generate_job(config_path, output_path, project_dir)


if __name__ == "__main__":
    main()
