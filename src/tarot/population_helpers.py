"""
Population augmentation helpers for the League GUI.

Generate random agents, mutate from a base selection, or clone agents
to build or expand a population.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .ga import GAConfig, mutate_agent
from .tournament import Agent, AgentId, Population

DEFAULT_TRAIT_NAMES = ("aggressiveness", "defensiveness")


def generate_random_agents(
    n: int,
    player_counts: List[int],
    rng: random.Random,
    *,
    trait_names: Tuple[str, ...] = DEFAULT_TRAIT_NAMES,
    trait_bounds: Tuple[float, float] = (0.0, 1.0),
    id_prefix: str = "rand",
) -> List[Agent]:
    """
    Create N agents with random traits and no checkpoint.

    Args:
        n: Number of agents to create.
        player_counts: Player counts supported (e.g. [4] or [3, 4, 5]).
        rng: Random generator.
        trait_names: Trait keys to randomize.
        trait_bounds: (min, max) for each trait value.
        id_prefix: Prefix for agent IDs (rand0, rand1, ...).

    Returns:
        List of new agents.
    """
    lo, hi = trait_bounds
    agents: List[Agent] = []
    for i in range(n):
        traits = {k: lo + (hi - lo) * rng.random() for k in trait_names}
        agent = Agent(
            id=f"{id_prefix}{i}",
            name=f"{id_prefix}_{i}",
            player_counts=list(player_counts),
            traits=traits,
            checkpoint_path=None,
            arch_name=None,
        )
        agents.append(agent)
    return agents


def mutate_from_base(
    base_agents: List[Agent],
    n: int,
    mutation_prob: float,
    mutation_std: float,
    rng: random.Random,
    *,
    existing_ids: set[AgentId] | None = None,
    id_prefix: str = "mut",
) -> List[Agent]:
    """
    Create N mutated children from a base selection of agents.

    Parents are chosen uniformly at random from base_agents. Each child
    is a mutated copy of its parent.

    Args:
        base_agents: Agents to use as parents.
        n: Number of children to create.
        mutation_prob: Probability of mutating each trait.
        mutation_std: Std for Gaussian perturbation of traits.
        rng: Random generator.
        existing_ids: IDs to avoid when assigning child IDs.
        id_prefix: Prefix for child IDs (mut0, mut1, ...).

    Returns:
        List of new mutated agents.
    """
    if not base_agents:
        return []

    existing_ids = existing_ids or set()
    cfg = GAConfig(
        population_size=n,
        elite_fraction=0.0,
        mutation_prob=mutation_prob,
        mutation_std=mutation_std,
    )
    children: List[Agent] = []
    for i in range(n):
        parent = rng.choice(base_agents)
        new_id = f"{id_prefix}{i}"
        while new_id in existing_ids:
            i += 1
            new_id = f"{id_prefix}{i}"
        existing_ids.add(new_id)
        child = mutate_agent(parent, new_id, cfg, rng)
        children.append(child)
    return children


def clone_agents(
    agents: List[Agent],
    n_per_agent: int,
    rng: random.Random,
    *,
    existing_ids: set[AgentId] | None = None,
    id_prefix: str = "clone",
) -> List[Agent]:
    """
    Clone agents (same traits, new IDs). Checkpoint path is shared, not copied.

    Args:
        agents: Agents to clone.
        n_per_agent: Number of clones per agent (total = len(agents) * n_per_agent).
        rng: Random generator (for ID uniqueness fallback).
        existing_ids: IDs to avoid when assigning clone IDs.
        id_prefix: Prefix for clone IDs.

    Returns:
        List of cloned agents.
    """
    if not agents or n_per_agent <= 0:
        return []

    existing_ids = set(existing_ids or [])
    clones: List[Agent] = []
    counter = 0
    for agent in agents:
        for _ in range(n_per_agent):
            new_id = f"{id_prefix}{counter}"
            while new_id in existing_ids:
                counter += 1
                new_id = f"{id_prefix}{counter}"
            existing_ids.add(new_id)
            counter += 1
            clone = Agent(
                id=new_id,
                name=agent.name,
                player_counts=list(agent.player_counts),
                elo_3p=agent.elo_3p,
                elo_4p=agent.elo_4p,
                elo_5p=agent.elo_5p,
                elo_global=agent.elo_global,
                generation=agent.generation,
                traits=dict(agent.traits),
                checkpoint_path=agent.checkpoint_path,
                arch_name=agent.arch_name,
                parents=list(agent.parents),
                can_use_as_ga_parent=agent.can_use_as_ga_parent,
                fixed_elo=agent.fixed_elo,
                clone_only=agent.clone_only,
                play_in_league=agent.play_in_league,
                matches_played=0,
                total_match_score=0.0,
            )
            clones.append(clone)
    return clones


__all__ = [
    "generate_random_agents",
    "mutate_from_base",
    "clone_agents",
    "DEFAULT_TRAIT_NAMES",
]
