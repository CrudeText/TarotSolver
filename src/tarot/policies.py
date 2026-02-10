"""
Helpers for turning trained checkpoints into Policy objects.

The main entry point is ``load_policy_from_checkpoint``, which loads a
``TarotActorCritic`` model trained via ``TarotPPOTrainer`` and wraps it in an
object implementing the ``Policy`` protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, TYPE_CHECKING

import numpy as np
import torch
from torch.distributions import Categorical

from .agents import Policy, RandomAgent
from .models import TarotActorCritic, PolicyConfig
from .training import TarotPPOTrainer, _mask_logits, _pad_observation

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checking only
    from .tournament import Agent


@dataclass
class NNPolicy(Policy):
    """
    Policy wrapper around a trained TarotActorCritic network.

    By default actions are sampled stochastically; set ``deterministic=True``
    to always pick the argmax action instead.
    """

    model: TarotActorCritic
    policy_cfg: PolicyConfig
    device: torch.device
    deterministic: bool = False

    def act(self, obs: Sequence[float], legal_actions_mask: Iterable[bool]) -> int:  # type: ignore[override]
        obs_vec = _pad_observation(obs, self.policy_cfg.obs_dim)
        obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_np = np.array(list(legal_actions_mask), dtype=bool)
        if mask_np.shape[0] == 0:
            raise ValueError("Empty legal_actions_mask in NNPolicy.act")
        mask_t = torch.from_numpy(mask_np).to(self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.model(obs_t)
            masked_logits = _mask_logits(logits, mask_t)
            if self.deterministic:
                action = torch.argmax(masked_logits, dim=-1)
            else:
                dist = Categorical(logits=masked_logits)
                action = dist.sample()

        return int(action.item())


def load_policy_from_checkpoint(
    directory: str,
    device: torch.device | None = None,
    deterministic: bool = False,
) -> NNPolicy:
    """
    Load a trained policy from ``directory`` created by TarotPPOTrainer.
    """
    device = device or torch.device("cpu")
    model, policy_cfg = TarotPPOTrainer.load_model_from_checkpoint(directory, device=device)
    return NNPolicy(model=model, policy_cfg=policy_cfg, device=device, deterministic=deterministic)


def policy_for_agent(
    agent: "Agent",
    device: torch.device | None = None,
    deterministic: bool = False,
) -> Policy:
    """
    Construct a Policy for a tournament Agent.

    - If the agent has a ``checkpoint_path``, load the corresponding neural policy.
    - Otherwise, fall back to a seeded RandomAgent (baseline behaviour).
    """
    device = device or torch.device("cpu")
    if agent.checkpoint_path:
        return load_policy_from_checkpoint(
            agent.checkpoint_path,
            device=device,
            deterministic=deterministic,
        )
    # Stable seed derived from agent id for reproducibility across runs.
    seed = abs(hash(agent.id)) % (2**32)
    return RandomAgent(seed=seed)


__all__ = ["NNPolicy", "load_policy_from_checkpoint", "policy_for_agent"]

