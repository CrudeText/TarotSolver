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
    elite_clone_fraction: float = 0.0  # fraction of offspring slots filled by cloning elites
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
    *,
    ga_parents_only: bool = False,
) -> List[Tuple[Agent, float]]:
    agents = pop.agents.values()
    if ga_parents_only:
        agents = [a for a in agents if a.can_use_as_ga_parent]
    scored = [(agent, fitness_fn(agent)) for agent in agents]
    scored.sort(key=lambda af: af[1], reverse=True)
    return scored


def select_elites(
    pop: Population,
    cfg: GAConfig,
    fitness_fn: Callable[[Agent], float],
    *,
    ga_parents_only: bool = True,
) -> List[Agent]:
    scored = _sorted_agents_by_fitness(pop, fitness_fn, ga_parents_only=ga_parents_only)
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
        can_use_as_ga_parent=parent.can_use_as_ga_parent,
        fixed_elo=parent.fixed_elo,
        clone_only=parent.clone_only,
        play_in_league=parent.play_in_league,
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
      - Reference agents (can_use_as_ga_parent=False) are copied unchanged.
      - Elites (top fraction by fitness among eligible agents) copied unchanged.
      - Remaining slots filled by mutated children of parents selected
        via roulette selection on fitness (only from eligible agents).

    NOTE: This function assumes that ELOs and match stats have already been
    updated (e.g. by running tournaments) before it is called.
    """
    rng = rng or random.Random()

    reference_agents = [a for a in pop.agents.values() if not a.can_use_as_ga_parent]
    eligible_agents = [a for a in pop.agents.values() if a.can_use_as_ga_parent]

    new_pop = Population()
    # Copy reference agents as-is (they participate in tournaments but not in evolution)
    for agent in reference_agents:
        new_pop.add(agent)

    slots_for_evolved = max(0, cfg.population_size - len(reference_agents))
    if slots_for_evolved == 0:
        return new_pop

    scored = _sorted_agents_by_fitness(pop, fitness_fn, ga_parents_only=True)
    elite_count = max(1, int(slots_for_evolved * cfg.elite_fraction))
    elites = [a for a, _ in scored[:elite_count]]

    for agent in elites:
        new_pop.add(agent)

    remaining = slots_for_evolved - len(elites)
    if remaining <= 0:
        return new_pop

    clone_fraction = max(0.0, min(1.0, cfg.elite_clone_fraction))
    clone_slots = int(remaining * clone_fraction)
    mutate_slots = remaining - clone_slots

    clone_counter = 0
    for _ in range(clone_slots):
        parent = rng.choice(elites)
        new_id = f"{parent.id}-clone{clone_counter}"
        while new_id in pop.agents or new_id in new_pop.agents:
            clone_counter += 1
            new_id = f"{parent.id}-clone{clone_counter}"
        clone_counter += 1
        clone = Agent(
            id=new_id,
            name=parent.name,
            player_counts=list(parent.player_counts),
            elo_3p=parent.elo_3p,
            elo_4p=parent.elo_4p,
            elo_5p=parent.elo_5p,
            elo_global=parent.elo_global,
            generation=parent.generation,
            traits=dict(parent.traits),
            checkpoint_path=parent.checkpoint_path,
            arch_name=parent.arch_name,
            parents=list(parent.parents),
            can_use_as_ga_parent=parent.can_use_as_ga_parent,
            fixed_elo=parent.fixed_elo,
            clone_only=parent.clone_only,
            play_in_league=parent.play_in_league,
            matches_played=0,
            total_match_score=0.0,
        )
        new_pop.add(clone)

    parents = _roulette_select(scored, mutate_slots, rng)
    child_counts: Dict[AgentId, int] = {}
    for parent in parents:
        count = child_counts.get(parent.id, 0) + 1
        child_counts[parent.id] = count
        new_id = f"{parent.id}-c{count}"
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

