"""
Simple baseline agents and the generic policy interface.

Phase 3 starts with a RandomAgent that:
- Works with any of the TarotEnv* environments.
- Chooses uniformly among actions marked as legal by the env-provided mask.

The small ``Policy`` protocol defines the contract used throughout tournaments
and training: ``act(obs, legal_actions_mask) -> action_index``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Protocol, Sequence


class Policy(Protocol):
    """Stateless or stateful decision policy working on flat observations."""

    def act(self, obs: Sequence[float], legal_actions_mask: Iterable[bool]) -> int:
        """
        Choose an action index given an observation and a boolean legal-action mask.

        Implementations must only return indices where ``legal_actions_mask[i]`` is
        true; callers are free to validate or fall back to a default if needed.
        """


@dataclass
class RandomAgent:
    """
    Baseline policy that samples uniformly among legal actions.

    Usage:
        agent = RandomAgent(seed=42)
        action = agent.act(obs, legal_actions_mask)
    """

    seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def act(self, obs: Sequence[float], legal_actions_mask: Iterable[bool]) -> int:
        """Pick a random legal action given an observation and a boolean mask."""
        legal_indices: List[int] = [i for i, ok in enumerate(legal_actions_mask) if ok]
        if not legal_indices:
            raise ValueError("No legal actions available for RandomAgent")
        return self._rng.choice(legal_indices)


__all__ = ["Policy", "RandomAgent"]

