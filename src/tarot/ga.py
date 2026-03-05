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
    # Legacy (used only when loading old configs; ignored by next_generation logic)
    elite_fraction: float = 0.1
    elite_clone_fraction: float = 0.0
    # Count-based: sexual_offspring + mutate + clone MUST equal GA slots (population_size - non-GA agents)
    sexual_offspring_count: int | None = None
    mutate_count: int | None = None
    clone_count: int | None = None
    # Sexual reproduction (gearbox)
    sexual_parent_with_replacement: bool = True
    sexual_parent_fitness_weighted: bool = True
    sexual_trait_combination: str = "average"  # "average" | "crossover"
    mutation_prob: float = 0.5
    mutation_std: float = 0.1  # for traits in [0, 1]


def compute_fitness(
    agent: Agent,
    *,
    fitness_elo_a: float = 1.0,
    fitness_elo_b: float = 1.0,
    fitness_avg_c: float = 0.0,
    fitness_avg_d: float = 1.0,
    # Legacy names for backward compatibility when loading old configs
    weight_global_elo: float | None = None,
    weight_avg_score: float | None = None,
) -> float:
    """
    Fitness = a*ELO^b + c*avg_score^d.

    avg_match_score is total_match_score / matches_played if matches_played > 0, else 0.
    If weight_global_elo / weight_avg_score are provided (legacy), they map to a=weight_global_elo, b=1, c=weight_avg_score, d=1.
    """
    if weight_global_elo is not None:
        fitness_elo_a = weight_global_elo
        fitness_elo_b = 1.0
    if weight_avg_score is not None:
        fitness_avg_c = weight_avg_score
        fitness_avg_d = 1.0
    if agent.matches_played > 0:
        avg_score = agent.total_match_score / agent.matches_played
    else:
        avg_score = 0.0
    elo = max(0.0, agent.elo_global)
    score = max(0.0, avg_score)
    return fitness_elo_a * (elo ** fitness_elo_b) + fitness_avg_c * (score ** fitness_avg_d)


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


def _select_parents_from_pool(
    scored_elite: List[Tuple[Agent, float]],
    num_picks: int,
    rng: random.Random,
    with_replacement: bool,
    fitness_weighted: bool,
) -> List[Agent]:
    """Select num_picks agents from scored_elite (e.g. 2 parents per sexual offspring)."""
    if not scored_elite or num_picks <= 0:
        return []
    pool: List[Tuple[Agent, float]] = list(scored_elite)
    selected: List[Agent] = []
    for _ in range(num_picks):
        if not pool:
            break
        weights = [max(0.0, f) for _, f in pool]
        total = sum(weights)
        if total <= 0.0 or not fitness_weighted:
            i = rng.randint(0, len(pool) - 1)
        else:
            r = rng.random() * total
            acc = 0.0
            i = 0
            for j, (_, f) in enumerate(pool):
                acc += max(0.0, f)
                if acc >= r:
                    i = j
                    break
        agent = pool[i][0]
        selected.append(agent)
        if not with_replacement:
            pool.pop(i)
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


def combine_agents(
    parent1: Agent,
    parent2: Agent,
    new_id: AgentId,
    combination: str,
    rng: random.Random,
) -> Agent:
    """
    Create one offspring from two parents by combining traits.
    combination: "average" (per-trait mean) or "crossover" (per-trait random choice from one parent).
    Checkpoint/arch/name taken from parent1.
    """
    all_keys = set(parent1.traits) | set(parent2.traits)
    traits: Dict[str, float] = {}
    for k in all_keys:
        v1 = parent1.traits.get(k, 0.5)
        v2 = parent2.traits.get(k, 0.5)
        if combination == "average":
            traits[k] = min(1.0, max(0.0, (v1 + v2) / 2.0))
        else:  # crossover
            traits[k] = v1 if rng.random() < 0.5 else v2
    gen = max(parent1.generation, parent2.generation) + 1
    child = Agent(
        id=new_id,
        name=parent1.name,
        player_counts=list(parent1.player_counts),
        elo_3p=parent1.elo_3p,
        elo_4p=parent1.elo_4p,
        elo_5p=parent1.elo_5p,
        elo_global=parent1.elo_global,
        generation=gen,
        traits=traits,
        checkpoint_path=parent1.checkpoint_path,
        arch_name=parent1.arch_name,
        parents=[parent1.id, parent2.id],
        can_use_as_ga_parent=parent1.can_use_as_ga_parent,
        fixed_elo=parent1.fixed_elo,
        clone_only=parent1.clone_only,
        play_in_league=parent1.play_in_league,
        matches_played=0,
        total_match_score=0.0,
    )
    return child


def next_generation(
    pop: Population,
    cfg: GAConfig,
    rng: random.Random | None = None,
    fitness_fn: Callable[[Agent], float] = compute_fitness,
) -> Population:
    """
    Build the next generation from the current population.

    Deterministic banded selection over GA parents (count-based only):
      - Reference agents (can_use_as_ga_parent == False) are copied as-is into the next
        population and never counted in GA slots.
      - GA parents are sorted by fitness (desc). Let:
            n = cfg.clone_count
            m = cfg.mutate_count
            u = cfg.sexual_offspring_count
        and slots_for_evolved = cfg.population_size - len(reference_agents).
      - The following hard constraints are enforced (no clamping):
            n + m + u == slots_for_evolved
            slots_for_evolved <= number of GA parents
      - Clone band: top n GA parents are copied unchanged into the next population.
      - Mutate band: next m GA parents are replaced by mutated children
        (one child per parent via mutate_agent).
      - Sexual band: bottom u GA parents are deleted; their slots are filled by u
        sexual offspring whose parents are sampled from the elite parent pool =
        top (n + m) GA parents, using GAConfig gearbox settings.

    NOTE: ELOs and match stats must already be updated (e.g. by tournaments) before calling.
    """
    rng = rng or random.Random()

    reference_agents = [a for a in pop.agents.values() if not a.can_use_as_ga_parent]
    new_pop = Population()
    for agent in reference_agents:
        new_pop.add(agent)

    slots_for_evolved = max(0, cfg.population_size - len(reference_agents))
    if slots_for_evolved == 0:
        return new_pop

    # GA parents only (used for selection and reproduction)
    scored = _sorted_agents_by_fitness(pop, fitness_fn, ga_parents_only=True)
    ga_parents = [a for a, _ in scored]
    if not ga_parents and slots_for_evolved > 0:
        raise ValueError("GAConfig requires GA parents but none are available (can_use_as_ga_parent == False for all agents).")

    # Validate counts: clone + mutate + sexual must exactly fill GA slots and not exceed GA parent count.
    if cfg.clone_count is None or cfg.mutate_count is None or cfg.sexual_offspring_count is None:
        raise ValueError("GAConfig must define clone_count, mutate_count, and sexual_offspring_count for count-based GA.")

    clone_n = int(cfg.clone_count)
    mutate_n = int(cfg.mutate_count)
    sexual_n = int(cfg.sexual_offspring_count)
    if clone_n < 0 or mutate_n < 0 or sexual_n < 0:
        raise ValueError("GAConfig counts must be non-negative.")

    total = clone_n + mutate_n + sexual_n
    if total != slots_for_evolved:
        raise ValueError(
            f"GAConfig counts inconsistent with GA slots: clone({clone_n}) + mutate({mutate_n}) + sexual({sexual_n}) = {total}, "
            f"but slots_for_evolved = {slots_for_evolved}."
        )
    if total > len(ga_parents):
        raise ValueError(
            f"GAConfig requires {total} GA slots but only {len(ga_parents)} GA parents are available."
        )

    # Deterministic bands over sorted GA parents
    # Band order (by descending fitness): [clones][mutate][sexual-delete]
    band_parents = ga_parents[:total]
    clone_band = band_parents[:clone_n]
    mutate_band = band_parents[clone_n : clone_n + mutate_n]
    # sexual-delete band is band_parents[clone_n + mutate_n :], but we only need its size (sexual_n)

    # 1) Clone band: carry over elites unchanged (no duplication)
    for parent in clone_band:
        new_pop.add(parent)

    # 2) Mutate band: each parent produces exactly one mutated child; parents themselves are not copied
    child_counts: Dict[AgentId, int] = {}
    for parent in mutate_band:
        count = child_counts.get(parent.id, 0) + 1
        child_counts[parent.id] = count
        new_id = f"{parent.id}-c{count}"
        while new_id in pop.agents or new_id in new_pop.agents:
            count += 1
            child_counts[parent.id] = count
            new_id = f"{parent.id}-c{count}"
        child = mutate_agent(parent, new_id, cfg, rng)
        new_pop.add(child)

    # 3) Sexual band: bottom u GA parents are conceptually deleted; we fill their slots with sexual offspring
    combo = (cfg.sexual_trait_combination or "average").lower()
    if combo not in ("average", "crossover"):
        combo = "average"
    elite_for_sexual = band_parents[: clone_n + mutate_n]  # parents allowed for sexual reproduction
    scored_elite: List[Tuple[Agent, float]] = [(a, f) for (a, f) in scored if a in elite_for_sexual]
    sex_counter = 0
    for _ in range(sexual_n):
        parents = _select_parents_from_pool(
            scored_elite,
            2,
            rng,
            with_replacement=cfg.sexual_parent_with_replacement,
            fitness_weighted=cfg.sexual_parent_fitness_weighted,
        )
        if len(parents) < 2:
            break
        new_id = f"sex-{sex_counter}"
        while new_id in pop.agents or new_id in new_pop.agents:
            sex_counter += 1
            new_id = f"sex-{sex_counter}"
        sex_counter += 1
        offspring = combine_agents(parents[0], parents[1], new_id, combo, rng)
        new_pop.add(offspring)

    return new_pop


__all__ = [
    "GAConfig",
    "combine_agents",
    "compute_fitness",
    "mutate_agent",
    "next_generation",
    "select_elites",
]

