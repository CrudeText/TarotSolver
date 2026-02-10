"""
Genetic algorithm helpers for evolving a population of Agents.

This module focuses on:
- Fitness computation from ELO and match statistics.
- Selection (with elites).
- Mutation of traits / metadata (model-agnostic).
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Dict, List, Tuple

from .tournament import Agent, AgentId, Population


@dataclass
class GAConfig:
    population_size: int
    elite_fraction: float = 0.1
    mutation_prob: float = 0.5
    mutation_std: float = 0.1  # for traits in [0, 1]


def compute_fitness(
    agent: Agent,
    weight_global_elo: float = 1.0,
    weight_avg_score: float = 0.0,
) -> float:
    """
    Default fitness: primarily global ELO, optionally nudged by average match score.

    avg_match_score is total_match_score / matches_played if matches_played > 0, else 0.
    """
    if agent.matches_played > 0:
        avg_score = agent.total_match_score / agent.matches_played
    else:
        avg_score = 0.0
    return weight_global_elo * agent.elo_global + weight_avg_score * avg_score


def _sorted_agents_by_fitness(
    pop: Population,
    fitness_fn: Callable[[Agent], float],
) -> List[Tuple[Agent, float]]:
    scored = [(agent, fitness_fn(agent)) for agent in pop.agents.values()]
    scored.sort(key=lambda af: af[1], reverse=True)
    return scored


def select_elites(
    pop: Population,
    cfg: GAConfig,
    fitness_fn: Callable[[Agent], float],
) -> List[Agent]:
    scored = _sorted_agents_by_fitness(pop, fitness_fn)
    elite_count = max(1, int(cfg.population_size * cfg.elite_fraction))
    return [a for a, _ in scored[:elite_count]]


def _roulette_select(
    scored_agents: List[Tuple[Agent, float]],
    num: int,
    rng: random.Random,
) -> List[Agent]:
    # Shift fitnesses to be non-negative
    fitnesses = [max(0.0, f) for _, f in scored_agents]
    total = sum(fitnesses)
    if total == 0.0:
        # All equal, fall back to uniform
        return [rng.choice([a for a, _ in scored_agents]) for _ in range(num)]

    selected: List[Agent] = []
    for _ in range(num):
        r = rng.random() * total
        acc = 0.0
        for (agent, f) in scored_agents:
            acc += max(0.0, f)
            if acc >= r:
                selected.append(agent)
                break
    return selected


def mutate_agent(
    parent: Agent,
    new_id: AgentId,
    cfg: GAConfig,
    rng: random.Random,
) -> Agent:
    """
    Create a mutated child from a parent.

    For now we mutate only traits slightly; model weights and hyperparameters
    remain unchanged (they are referenced via checkpoint_path / arch_name).
    """
    child = Agent(
        id=new_id,
        name=parent.name,
        player_counts=list(parent.player_counts),
        elo_3p=parent.elo_3p,
        elo_4p=parent.elo_4p,
        elo_5p=parent.elo_5p,
        elo_global=parent.elo_global,
        generation=parent.generation + 1,
        traits=dict(parent.traits),
        checkpoint_path=parent.checkpoint_path,
        arch_name=parent.arch_name,
        parents=[parent.id],
    )

    # Reset match stats for new generation
    child.matches_played = 0
    child.total_match_score = 0.0

    # Mutate traits with some probability
    for k, v in list(child.traits.items()):
        if rng.random() < cfg.mutation_prob:
            delta = rng.gauss(0.0, cfg.mutation_std)
            new_v = min(1.0, max(0.0, v + delta))
            child.traits[k] = new_v

    return child


def next_generation(
    pop: Population,
    cfg: GAConfig,
    rng: random.Random | None = None,
    fitness_fn: Callable[[Agent], float] = compute_fitness,
) -> Population:
    """
    Build the next generation from the current population using:
      - Elites (top fraction by fitness) copied unchanged.
      - Remaining slots filled by mutated children of parents selected
        via roulette selection on fitness.

    NOTE: This function assumes that ELOs and match stats have already been
    updated (e.g. by running tournaments) before it is called.
    """
    rng = rng or random.Random()

    scored = _sorted_agents_by_fitness(pop, fitness_fn)
    elites = select_elites(pop, cfg, fitness_fn)
    elite_ids = {a.id for a in elites}

    new_pop = Population()
    # Copy elites as-is
    for agent in elites:
        new_pop.add(agent)

    # Select parents for children
    remaining = cfg.population_size - len(elites)
    if remaining <= 0:
        return new_pop

    parents = _roulette_select(scored, remaining, rng)
    # Simple child ID scheme: parent_id + "-cN"
    child_counts: Dict[AgentId, int] = {}
    for parent in parents:
        count = child_counts.get(parent.id, 0) + 1
        child_counts[parent.id] = count
        new_id = f"{parent.id}-c{count}"
        # Ensure we don't collide with existing IDs
        while new_id in pop.agents or new_id in new_pop.agents:
            count += 1
            child_counts[parent.id] = count
            new_id = f"{parent.id}-c{count}"
        child = mutate_agent(parent, new_id, cfg, rng)
        new_pop.add(child)

    return new_pop


__all__ = [
    "GAConfig",
    "compute_fitness",
    "select_elites",
    "mutate_agent",
    "next_generation",
]

