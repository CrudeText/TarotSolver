"""
Custom PyTorch training helpers built on top of the TarotEnv* environments.

First version: PPO-style actor–critic training for 4-player Tarot using the
TarotEnv4P environment and the TarotActorCritic network.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import random
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from . import __version__
from .env_game import StepResult, TarotEnv4P
from .models import TarotActorCritic, PolicyConfig


@dataclass
class Transition:
    """One environment step transition suitable for PPO-style algorithms."""

    obs: Sequence[float]
    action: int
    reward: float
    value: float
    log_prob: float
    done: bool
    legal_actions_mask: Sequence[bool]


@dataclass
class PPOConfig:
    """Hyperparameters for PPO training."""

    obs_dim: int = 412
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    batch_size: int = 2048
    minibatch_size: int = 256
    update_epochs: int = 4
    max_grad_norm: float = 0.5


def _mask_logits(
    logits: torch.Tensor,
    legal_actions_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply a boolean legal-actions mask to logits."""
    illegal = ~legal_actions_mask
    logits = logits.clone()
    logits[illegal] = -1e9
    return logits


def _pad_observation(obs: Sequence[float], target_dim: int) -> List[float]:
    """
    Pad or truncate an observation to ``target_dim``.

    Bidding observations are length 116, while play observations are length 412.
    The network expects a fixed size (default 412), so we:
      - pad with zeros when obs is shorter;
      - truncate if it is (unexpectedly) longer.
    """
    if len(obs) == target_dim:
        return list(obs)
    if len(obs) > target_dim:
        return list(obs)[:target_dim]
    padded = list(obs) + [0.0] * (target_dim - len(obs))
    return padded


class TarotPPOTrainer:
    """
    PPO trainer for a single-seat TarotEnv4P.

    This is intentionally minimal and CPU-first; it can later be extended to
    multi-seat self-play, GPU training, and personality conditioning.
    """

    def __init__(
        self,
        env: TarotEnv4P,
        cfg: PPOConfig | None = None,
        policy_cfg: PolicyConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.env = env
        self.cfg = cfg or PPOConfig()
        self.policy_cfg = policy_cfg or PolicyConfig(obs_dim=self.cfg.obs_dim)
        self.device = device or torch.device("cpu")

        self.model = TarotActorCritic(
            obs_dim=self.policy_cfg.obs_dim,
            hidden_dim=self.policy_cfg.hidden_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)

    # ---- Checkpointing ----

    def save_checkpoint(self, directory: str) -> None:
        """
        Save model weights and configs to ``directory``.

        Layout:
          - policy.pt      : model state_dict
          - config.json    : policy + PPO config and version metadata
        """
        import json

        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), out_dir / "policy.pt")

        meta = {
            "version": __version__,
            "policy_config": asdict(self.policy_cfg),
            "ppo_config": asdict(self.cfg),
        }
        with (out_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load_model_from_checkpoint(
        directory: str,
        device: torch.device | None = None,
    ) -> Tuple[TarotActorCritic, PolicyConfig]:
        """
        Load a TarotActorCritic model from a checkpoint directory created by
        ``save_checkpoint``.
        """
        import json

        device = device or torch.device("cpu")
        ckpt_dir = Path(directory)
        with (ckpt_dir / "config.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)

        policy_cfg_dict = meta.get("policy_config", {})
        policy_cfg = PolicyConfig(**policy_cfg_dict)
        model = TarotActorCritic(
            obs_dim=policy_cfg.obs_dim,
            hidden_dim=policy_cfg.hidden_dim,
        ).to(device)

        state_dict = torch.load(ckpt_dir / "policy.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return model, policy_cfg

    def _collect_rollouts(self, seed: int | None = None) -> Tuple[Transition, ...]:
        """Collect one batch of rollouts."""
        rng = random.Random(seed)
        _ = rng  # currently unused; kept for future stochasticity hooks

        transitions: List[Transition] = []

        step: StepResult = self.env.reset()
        while len(transitions) < self.cfg.batch_size:
            obs_vec = _pad_observation(step.obs, self.policy_cfg.obs_dim)
            obs = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_np = np.array(step.legal_actions_mask, dtype=bool)
            mask = torch.from_numpy(mask_np).to(self.device).unsqueeze(0)

            logits, value = self.model(obs)
            masked_logits = _mask_logits(logits, mask)
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_step = self.env.step(int(action.item()))

            transitions.append(
                Transition(
                    obs=obs_vec,
                    action=int(action.item()),
                    reward=float(next_step.reward),
                    value=float(value.item()),
                    log_prob=float(log_prob.item()),
                    done=bool(next_step.done),
                    legal_actions_mask=list(step.legal_actions_mask),
                )
            )

            step = next_step
            if step.done:
                step = self.env.reset()

        return tuple(transitions)

    def _compute_advantages(
        self,
        transitions: Sequence[Transition],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns from a sequence of transitions.
        """
        rewards = [t.reward for t in transitions]
        values = [t.value for t in transitions]
        dones = [t.done for t in transitions]

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values + [0.0], dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        advantages = torch.zeros(len(transitions), dtype=torch.float32, device=self.device)
        gae = 0.0
        for t in reversed(range(len(transitions))):
            delta = rewards_t[t] + self.cfg.gamma * values_t[t + 1] * (1.0 - dones_t[t]) - values_t[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - dones_t[t]) * gae
            advantages[t] = gae

        returns = advantages + values_t[:-1]
        return advantages, returns

    def update(self, seed: int | None = None) -> dict:
        """
        Run one PPO update cycle: collect rollouts, compute losses, and step the optimizer.
        """
        transitions = self._collect_rollouts(seed=seed)
        advantages, returns = self._compute_advantages(transitions)

        obs_batch = torch.tensor(
            [t.obs for t in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        mask_batch = torch.tensor(
            [t.legal_actions_mask for t in transitions],
            dtype=torch.bool,
            device=self.device,
        )
        actions_batch = torch.tensor(
            [t.action for t in transitions],
            dtype=torch.long,
            device=self.device,
        )
        old_log_probs_batch = torch.tensor(
            [t.log_prob for t in transitions],
            dtype=torch.float32,
            device=self.device,
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = len(transitions)
        idxs = np.arange(batch_size)

        stats: dict = {}

        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_idx = idxs[start:end]

                mb_obs = obs_batch[mb_idx]
                mb_mask = mask_batch[mb_idx]
                mb_actions = actions_batch[mb_idx]
                mb_old_log_probs = old_log_probs_batch[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                logits, values = self.model(mb_obs)
                masked_logits = _mask_logits(logits, mb_mask)
                dist = Categorical(logits=masked_logits)
                log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, mb_returns)

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                stats = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                }

        return stats


__all__ = ["Transition", "PPOConfig", "TarotPPOTrainer"]

