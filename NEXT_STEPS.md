# Next Steps: Fine-Tuning Infrastructure

This document outlines the plan for adding orchestrator fine-tuning capabilities to the repository.

## Overview

The current repository supports **inference-only** orchestration. To enable fine-tuning of the orchestrator's decision-making (e.g., via RL or supervised learning), we need to add training infrastructure while preserving the existing modular architecture.

## Recommended Repository Structure

```
msc-thesis/
├── src/agent_engine/
│   ├── core/
│   │   ├── orchestrator.py           # ✅ Existing: inference orchestrator
│   │   ├── orchestrator_trainable.py # 🆕 Trainable variant with policy hooks
│   │   └── state.py                  # ✅ Existing: execution state
│   │
│   ├── training/                      # 🆕 Fine-tuning module
│   │   ├── __init__.py
│   │   ├── collector.py               # Collect trajectories/episodes
│   │   ├── reward.py                  # Reward functions
│   │   ├── trainer.py                 # Training loop (RL/supervised)
│   │   └── policy.py                  # Policy network (if using RL)
│   │
│   └── ...
│
├── scripts/
│   ├── run_experiment.py              # ✅ Existing: inference
│   ├── collect_data.py                # 🆕 Collect training data
│   ├── train_orchestrator.py          # 🆕 Fine-tuning script
│   └── evaluate_policy.py             # 🆕 Evaluate trained policy
│
├── experiments/
│   ├── configs/
│   │   ├── gaia/                      # ✅ Existing: inference configs
│   │   └── training/                  # 🆕 Training configs
│   │       ├── rl_baseline.yaml
│   │       ├── supervised_baseline.yaml
│   │       └── dpo_baseline.yaml
│   │
│   ├── results/                       # ✅ Existing: inference results
│   └── training_runs/                 # 🆕 Training artifacts
│       └── rl_baseline_v1/
│           ├── checkpoints/
│           ├── logs/
│           └── config.yaml
│
└── data/
    └── training/                      # 🆕 Training data
        └── trajectories/
            ├── gaia_train_trajectories.jsonl
            └── gaia_val_trajectories.jsonl
```

## Key Components to Implement

### 1. Trajectory Collector (`src/agent_engine/training/collector.py`)

Collects reasoning trajectories for training.

```python
"""Trajectory collection for orchestrator training."""

from typing import List, Dict, Any
import json
from pathlib import Path

from ..core.orchestrator import AgenticOrchestrator
from ..core.state import ExecutionState


class TrajectoryCollector:
    """Collect reasoning trajectories for training.
    
    A trajectory contains:
    - Question and context
    - Sequence of states, actions (tool calls), and outcomes
    - Final answer and correctness
    - Intermediate rewards (if using dense rewards)
    """
    
    def __init__(self, orchestrator: AgenticOrchestrator, reward_fn=None):
        self.orchestrator = orchestrator
        self.reward_fn = reward_fn
        self.trajectories = []
    
    def collect_episode(
        self,
        question: str,
        question_id: int,
        system_prompt: str,
        ground_truth: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Run orchestrator and record all decisions/outcomes.
        
        Returns:
            Trajectory dict with states, actions, rewards, final outcome
        """
        # Run orchestrator
        state = self.orchestrator.run(question, question_id, system_prompt)
        
        # Evaluate final answer
        is_correct = self._evaluate_answer(state.answer, ground_truth, metadata)
        
        # Build trajectory
        trajectory = {
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "metadata": metadata or {},
            "states": self._extract_states(state),
            "actions": self._extract_actions(state),
            "final_answer": state.answer,
            "correct": is_correct,
            "total_turns": state.turn,
            "tool_counts": state.tool_counts,
        }
        
        # Compute rewards if reward function provided
        if self.reward_fn:
            trajectory["rewards"] = self.reward_fn.compute_trajectory_rewards(
                trajectory, state
            )
        
        self.trajectories.append(trajectory)
        return trajectory
    
    def _extract_states(self, state: ExecutionState) -> List[Dict[str, Any]]:
        """Extract state snapshots at each turn."""
        # Implementation: serialize conversation state at each turn
        pass
    
    def _extract_actions(self, state: ExecutionState) -> List[Dict[str, Any]]:
        """Extract actions (tool calls) from execution."""
        return state.tool_calls
    
    def _evaluate_answer(self, prediction: str, ground_truth: str, metadata: Dict) -> bool:
        """Evaluate if prediction matches ground truth."""
        # Use dataset-specific evaluator
        pass
    
    def save_trajectories(self, path: Path):
        """Save collected trajectories as JSONL."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            for traj in self.trajectories:
                f.write(json.dumps(traj) + '\n')
    
    def load_trajectories(self, path: Path):
        """Load trajectories from JSONL."""
        self.trajectories = []
        with open(path, 'r') as f:
            for line in f:
                self.trajectories.append(json.loads(line))
```

### 2. Reward Functions (`src/agent_engine/training/reward.py`)

Define reward signals for training.

```python
"""Reward functions for orchestrator training."""

from typing import Dict, Any, List
from abc import ABC, abstractmethod


class RewardFunction(ABC):
    """Base class for reward functions."""
    
    @abstractmethod
    def compute_trajectory_rewards(
        self,
        trajectory: Dict[str, Any],
        state: Any
    ) -> List[float]:
        """Compute per-step rewards for a trajectory.
        
        Args:
            trajectory: Trajectory dict with states/actions
            state: Final execution state
            
        Returns:
            List of rewards (one per action/turn)
        """
        pass


class SparseReward(RewardFunction):
    """Sparse reward: +1 if correct, 0 otherwise."""
    
    def __init__(self, correct_bonus: float = 1.0):
        self.correct_bonus = correct_bonus
    
    def compute_trajectory_rewards(self, trajectory, state):
        num_turns = trajectory["total_turns"]
        is_correct = trajectory["correct"]
        
        # All zeros except final turn
        rewards = [0.0] * (num_turns - 1)
        rewards.append(self.correct_bonus if is_correct else 0.0)
        
        return rewards


class DenseReward(RewardFunction):
    """Dense reward: intermediate rewards for good/bad tool calls."""
    
    def __init__(
        self,
        correct_bonus: float = 1.0,
        incorrect_penalty: float = 0.0,
        useful_tool_bonus: float = 0.1,
        redundant_tool_penalty: float = -0.05,
        max_turns_penalty: float = -0.2
    ):
        self.correct_bonus = correct_bonus
        self.incorrect_penalty = incorrect_penalty
        self.useful_tool_bonus = useful_tool_bonus
        self.redundant_tool_penalty = redundant_tool_penalty
        self.max_turns_penalty = max_turns_penalty
    
    def compute_trajectory_rewards(self, trajectory, state):
        num_turns = trajectory["total_turns"]
        is_correct = trajectory["correct"]
        actions = trajectory["actions"]
        
        rewards = []
        seen_tools = set()
        
        for turn_idx in range(num_turns):
            if turn_idx < len(actions):
                action = actions[turn_idx]
                tool_name = action["name"]
                
                # Reward for first use of each tool
                if tool_name not in seen_tools:
                    rewards.append(self.useful_tool_bonus)
                    seen_tools.add(tool_name)
                else:
                    # Penalty for redundant tool calls
                    rewards.append(self.redundant_tool_penalty)
            else:
                # No action (reasoning-only turn)
                rewards.append(0.0)
        
        # Final reward
        if is_correct:
            rewards[-1] += self.correct_bonus
        else:
            rewards[-1] += self.incorrect_penalty
        
        # Penalty if max turns reached
        if trajectory.get("metadata", {}).get("max_turns_reached"):
            rewards[-1] += self.max_turns_penalty
        
        return rewards


class ShapedReward(RewardFunction):
    """Shaped reward: guide agent toward correct behavior."""
    
    def __init__(self, correct_bonus: float = 1.0):
        self.correct_bonus = correct_bonus
    
    def compute_trajectory_rewards(self, trajectory, state):
        # TODO: implement reward shaping based on:
        # - Progress toward answer (heuristic)
        # - Tool call appropriateness (learned classifier?)
        # - Reasoning quality (if extractable)
        pass
```

### 3. Trainer (`src/agent_engine/training/trainer.py`)

Main training loop.

```python
"""Orchestrator training loop."""

from typing import Dict, Any, Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from ..utils.logging import get_logger

logger = get_logger(__name__)


class OrchestratorTrainer:
    """Train orchestrator policy (RL or supervised).
    
    Supports:
    - PPO (Proximal Policy Optimization)
    - DPO (Direct Preference Optimization)
    - Supervised learning from expert trajectories
    """
    
    def __init__(
        self,
        policy,
        reward_fn,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        self.policy = policy
        self.reward_fn = reward_fn
        self.config = config
        self.device = device
        
        self.optimizer = self._create_optimizer()
        self.step = 0
    
    def _create_optimizer(self):
        """Create optimizer based on config."""
        lr = self.config.get("learning_rate", 1e-4)
        return torch.optim.Adam(self.policy.parameters(), lr=lr)
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step.
        
        Args:
            batch: Batch of trajectories/experiences
            
        Returns:
            Dict of training metrics
        """
        algorithm = self.config.get("algorithm", "ppo")
        
        if algorithm == "ppo":
            return self._train_step_ppo(batch)
        elif algorithm == "dpo":
            return self._train_step_dpo(batch)
        elif algorithm == "supervised":
            return self._train_step_supervised(batch)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _train_step_ppo(self, batch):
        """PPO training step."""
        # TODO: implement PPO update
        # 1. Compute advantages
        # 2. Compute policy loss (clipped objective)
        # 3. Compute value loss
        # 4. Update policy
        pass
    
    def _train_step_dpo(self, batch):
        """DPO training step."""
        # TODO: implement DPO update
        # 1. Get preferred/dispreferred trajectory pairs
        # 2. Compute log-likelihood ratio
        # 3. Update policy to prefer better trajectories
        pass
    
    def _train_step_supervised(self, batch):
        """Supervised learning step."""
        # TODO: implement supervised update
        # 1. Get expert trajectories
        # 2. Compute cross-entropy loss
        # 3. Update policy
        pass
    
    def save_checkpoint(self, path: Path):
        """Save training checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'step': self.step,
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        logger.info(f"Loaded checkpoint from {path} (step {self.step})")
```

### 4. Trainable Orchestrator (`src/agent_engine/core/orchestrator_trainable.py`)

Orchestrator variant that uses a learned policy.

```python
"""Trainable orchestrator with policy network."""

from typing import Optional
from .orchestrator import AgenticOrchestrator


class TrainableOrchestrator(AgenticOrchestrator):
    """Orchestrator with learnable action selection policy.
    
    Differences from base orchestrator:
    - Can use neural policy for action selection
    - Exposes state representations for policy input
    - Supports exploration (epsilon-greedy, etc.)
    """
    
    def __init__(
        self,
        model_provider,
        tool_registry,
        policy_network=None,
        exploration_rate: float = 0.0,
        **kwargs
    ):
        super().__init__(model_provider, tool_registry, **kwargs)
        self.policy = policy_network
        self.exploration_rate = exploration_rate
    
    def select_next_action(self, state):
        """Select action using learned policy (if available).
        
        Falls back to base LLM generation if no policy or during exploration.
        """
        if self.policy and random.random() > self.exploration_rate:
            # Use learned policy
            state_repr = self._get_state_representation(state)
            action = self.policy.predict(state_repr)
            return action
        else:
            # Use base LLM generation (exploration or no policy)
            return self._generate_with_llm(state)
    
    def _get_state_representation(self, state):
        """Extract features from state for policy input.
        
        Possible features:
        - Conversation history (embedded)
        - Current turn number
        - Tool usage counts
        - Available tools
        """
        # TODO: implement state featurization
        pass
    
    def _generate_with_llm(self, state):
        """Base LLM generation (from parent orchestrator)."""
        # Delegate to parent implementation
        pass
```

## Training Configuration Schema

Add to `src/agent_engine/config/schema.py`:

```python
@dataclass
class TrainingConfig:
    """Training configuration for orchestrator fine-tuning."""
    
    # Algorithm
    algorithm: str = "ppo"  # ppo, dpo, supervised
    
    # Training loop
    num_episodes: int = 10000
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    
    # Reward
    reward_type: str = "sparse"  # sparse, dense, shaped
    correct_bonus: float = 1.0
    incorrect_penalty: float = 0.0
    
    # Checkpointing
    save_every: int = 100
    keep_best: int = 5
    checkpoint_dir: Path = Path("./experiments/training_runs")
    
    # Exploration (for RL)
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
```

## Example Training Config

**`experiments/configs/training/rl_baseline.yaml`:**

```yaml
name: "orchestrator_rl_v1"
description: "PPO training for orchestrator on GAIA"

# Base orchestrator settings
orchestrator:
  type: "trainable"
  max_turns: 15
  tool_limits:
    web_search: 10

# Training settings
training:
  algorithm: "ppo"
  num_episodes: 10000
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 10
  
  reward:
    type: "dense"  # sparse, dense, shaped
    correct_bonus: 1.0
    incorrect_penalty: 0.0
    useful_tool_bonus: 0.1
    redundant_tool_penalty: -0.05
  
  exploration:
    initial_rate: 0.3
    decay: 0.995
    min_rate: 0.05
  
  checkpoints:
    save_every: 100
    keep_best: 5
    dir: "./experiments/training_runs/rl_baseline_v1"

# Dataset for training
dataset:
  name: "gaia"
  split: "train"
  data_dir: "./data"

# Models (same as inference)
models:
  orchestrator:
    name: "Qwen3-32B"
    family: "qwen3"
    path_or_id: "Qwen/Qwen3-32B"
    role: "orchestrator"
    tensor_parallel_size: 2
    gpu_ids: [0, 1]

# Tools
tools:
  enabled_tools: [web_search, code_generator]
  direct_tool_call: true
```

## Training Scripts

### `scripts/collect_data.py`

```python
"""Collect training trajectories from orchestrator."""

import argparse
from pathlib import Path

from agent_engine.config import load_experiment_config
from agent_engine.training.collector import TrajectoryCollector
from agent_engine.core.orchestrator import AgenticOrchestrator
# ... setup code ...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-episodes", type=int, default=1000)
    args = parser.parse_args()
    
    # Load config and setup orchestrator
    config = load_experiment_config(Path(args.config))
    orchestrator = setup_orchestrator(config)
    
    # Collect trajectories
    collector = TrajectoryCollector(orchestrator)
    dataset = load_dataset(config.dataset)
    
    for i, example in enumerate(dataset):
        if i >= args.num_episodes:
            break
        collector.collect_episode(
            question=example.question,
            question_id=example.id,
            system_prompt=build_system_prompt(...),
            ground_truth=example.answer
        )
    
    # Save
    collector.save_trajectories(Path(args.output))
    print(f"Saved {len(collector.trajectories)} trajectories to {args.output}")

if __name__ == "__main__":
    main()
```

### `scripts/train_orchestrator.py`

```python
"""Train orchestrator policy."""

import argparse
from pathlib import Path

from agent_engine.config import load_experiment_config
from agent_engine.training.trainer import OrchestratorTrainer
from agent_engine.training.policy import OrchestratorPolicy
from agent_engine.training.reward import SparseReward, DenseReward
# ... setup code ...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trajectories", required=True)
    args = parser.parse_args()
    
    # Load config
    config = load_experiment_config(Path(args.config))
    
    # Setup policy and reward
    policy = OrchestratorPolicy(config)
    if config.training.reward.type == "sparse":
        reward_fn = SparseReward(config.training.reward.correct_bonus)
    elif config.training.reward.type == "dense":
        reward_fn = DenseReward(...)
    
    # Setup trainer
    trainer = OrchestratorTrainer(policy, reward_fn, config.training)
    
    # Load trajectories
    trajectories = load_trajectories(Path(args.trajectories))
    
    # Training loop
    for epoch in range(config.training.num_epochs):
        for batch in create_batches(trajectories, config.training.batch_size):
            metrics = trainer.train_step(batch)
            log_metrics(metrics)
        
        # Save checkpoint
        if epoch % config.training.save_every == 0:
            trainer.save_checkpoint(
                config.training.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            )

if __name__ == "__main__":
    main()
```

## Implementation Roadmap

### Phase 1: Data Collection Infrastructure
- [ ] Implement `TrajectoryCollector`
- [ ] Implement reward functions (sparse, dense)
- [ ] Create `scripts/collect_data.py`
- [ ] Collect baseline trajectories on GAIA train set

### Phase 2: Training Infrastructure
- [ ] Implement `OrchestratorPolicy` network
- [ ] Implement `OrchestratorTrainer` (start with supervised learning)
- [ ] Create `scripts/train_orchestrator.py`
- [ ] Test training loop on small dataset

### Phase 3: Trainable Orchestrator
- [ ] Implement `TrainableOrchestrator`
- [ ] Add policy integration hooks
- [ ] Test trained policy in inference mode

### Phase 4: Advanced Algorithms
- [ ] Implement PPO training
- [ ] Implement DPO training
- [ ] Add exploration strategies
- [ ] Hyperparameter tuning

### Phase 5: Evaluation & Analysis
- [ ] Create `scripts/evaluate_policy.py`
- [ ] Compare trained vs base orchestrator
- [ ] Analyze learned behaviors
- [ ] Write up results

## Estimated Effort

- **New files**: 8–10 files (~1500–2000 lines)
- **Modified files**: 3–4 (runner scripts, config schema)
- **Time estimate**: 2–4 weeks for full implementation
- **Dependencies**: May need PyTorch, transformers for policy network

## Design Principles

1. **Separation**: Training code isolated in `src/agent_engine/training/`
2. **Reuse**: Existing orchestrator/tools/models work as-is
3. **Flexibility**: Easy to try different RL algorithms
4. **Tracking**: Training runs separate from inference runs
5. **Reproducibility**: Training configs + checkpoints versioned

## Notes & Considerations

- **Policy architecture**: Consider starting with simple MLP over state features, then potentially upgrade to transformer-based policy
- **State representation**: Need to carefully design state features for policy input (conversation embeddings, tool usage, etc.)
- **Reward engineering**: Start with sparse rewards, then experiment with dense/shaped rewards
- **Computational cost**: RL training will be expensive (many episodes needed)
- **Baseline comparison**: Always compare against frozen LLM orchestrator
- **Evaluation**: Need holdout test set separate from training data

## References

- PPO: Schulman et al., 2017
- DPO: Rafailov et al., 2023
- ReAct: Yao et al., 2023
- AgentTuning: Zeng et al., 2023

---

**Last Updated**: 2026-02-17  
**Status**: Planning phase - ready to implement when needed
