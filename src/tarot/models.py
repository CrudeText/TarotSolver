"""
Neural network models for Tarot policies and value functions.

First version: a simple MLP-based actor–critic that operates on the flat
observations defined in ``tarot.env`` and produces:

- logits over the global action space (size NUM_ACTIONS)
- a scalar state-value estimate
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from .env import NUM_ACTIONS


class TarotMLP(nn.Module):
    """Shared MLP backbone used by actor and critic."""

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class TarotActorCritic(nn.Module):
    """
    Combined policy + value network.

    - Input: flat observation tensor of shape (batch, obs_dim)
    - Output:
        - logits: (batch, NUM_ACTIONS)
        - value:  (batch,)
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.backbone = TarotMLP(obs_dim, hidden_dim=hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, NUM_ACTIONS)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        x = self.backbone(obs)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


@dataclass
class PolicyConfig:
    """Metadata describing a saved policy architecture."""

    arch_name: str = "tarot_mlp_v1"
    obs_dim: int = 412
    hidden_dim: int = 256


__all__ = ["TarotActorCritic", "TarotMLP", "PolicyConfig"]

