# fine_tuning - placeholder

> **This module is not implemented yet.** The files below are structural placeholders only.

## Intended direction - exact methodology TBD

RL-based fine-tuning of the orchestrator. The reward signal will be derived from task performance
(answer correctness) evaluated against the existing benchmark datasets.

## Structure

```
fine_tuning/
├── config.py    # Training hyperparameters and paths
├── reward.py    # Reward function(s)
├── rollout.py   # Episode collection - runs the agent, gathers trajectories
└── trainer.py   # Main training loop
```
