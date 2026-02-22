# jobs/

SLURM job scripts for the Snellius cluster.

For setup instructions, job file descriptions, and how to submit experiments see the
[main README](../README.md#hpc--cluster-setup).

## Files

| File | Purpose |
|------|---------|
| `001_setup.job` | Create conda env + install project |
| `002_download_datasets.job` | Download benchmark datasets |
| `003_test_simple.job` | Smoke-test a single example |
| `004_export_env.job` | Export conda env YAMLs to `env_exports/` |
| `005_export_prompts.job` | Export prompt templates + tool schemas to JSON |
| `submit_job.sh` | Convenience wrapper — generates a job file from a config and submits it |
| `scripts/generate_job.py` | Generate a SLURM `.job` file from an experiment YAML |
| `templates/experiment.job.j2` | Jinja2 SLURM job template |
| `generated/` | Generated `.job` files (git-ignored) |
| `env_exports/` | Exported conda environment YAMLs |
