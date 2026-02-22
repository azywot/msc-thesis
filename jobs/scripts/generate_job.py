"""Generate SLURM job file from experiment configuration.

This script reads an experiment YAML config and generates a corresponding
SLURM job file using Jinja2 templates.
"""

import argparse
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_engine.config import load_experiment_config


def generate_job(config_path: Path, output_path: Path = None, project_dir: Path = None):
    """Generate SLURM job file from experiment config."""
    config = load_experiment_config(config_path)
    slurm = config.slurm

    num_gpus = slurm.num_gpus if slurm.num_gpus is not None else 1

    if project_dir is None:
        project_dir = Path.cwd()

    if output_path is None:
        output_path = Path("jobs/generated") / f"{config.name}.job"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # SLURM stdout goes to out/experiments/<experiment_name>/
    slurm_output_dir = f"out/experiments/{config.name}"
    (project_dir / slurm_output_dir).mkdir(parents=True, exist_ok=True)

    template_dir = Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("experiment.job.j2")

    job_content = template.render(
        experiment_name=config.name,
        partition=slurm.partition,
        num_gpus=num_gpus,
        ntasks=slurm.ntasks,
        cpus_per_task=slurm.cpus_per_task,
        time=slurm.time,
        conda_env=slurm.conda_env,
        seed=config.seed,
        config_path=str(config_path.absolute()),
        output_dir=str(config.output_dir),
        project_dir=str(project_dir.absolute()),
        slurm_output_dir=slurm_output_dir,
    )

    with open(output_path, 'w') as f:
        f.write(job_content)

    print(f"Generated SLURM job file: {output_path}")
    print(f"  Experiment: {config.name}")
    print(f"  Partition:  {slurm.partition}")
    print(f"  GPUs:       {num_gpus}")
    print(f"  Time:       {slurm.time}")
    print(f"  Config:     {config_path}")
    print(f"\nTo submit: sbatch {output_path}")
    print(f"JOB_FILE={output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate SLURM job file from experiment config")
    parser.add_argument("config", type=str, help="Path to experiment config YAML file")
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
