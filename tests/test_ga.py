"""Tests for genetic algorithm helpers."""

import random

from tarot.ga import GAConfig, compute_fitness, select_elites, mutate_agent, next_generation
from tarot.tournament import Agent, Population


def _make_dummy_population() -> Population:
    pop = Population()
    for i in range(4):
        a = Agent(
            id=f"A{i}",
            name=f"A{i}",
            player_counts=[4],
            elo_3p=1500.0,
            elo_4p=1500.0 + i * 50,
            elo_5p=1500.0,
            elo_global=1500.0 + i * 50,
            generation=0,
            traits={"aggressiveness": 0.5},
        )
        pop.add(a)
    return pop


def test_compute_fitness_uses_global_elo():
    pop = _make_dummy_population()
    agents = list(pop.agents.values())
    fits = [compute_fitness(a) for a in agents]
    assert fits[0] < fits[-1]  # higher global ELO → higher fitness


def test_select_elites_and_next_generation():
    pop = _make_dummy_population()
    cfg = GAConfig(population_size=4, elite_fraction=0.25, mutation_prob=1.0, mutation_std=0.1)
    rng = random.Random(123)

    elites = select_elites(pop, cfg, compute_fitness)
    assert len(elites) == 1
    elite = elites[0]

    new_pop = next_generation(pop, cfg, rng=rng, fitness_fn=compute_fitness)
    assert len(new_pop.agents) == cfg.population_size

    # Elite should still be present unchanged
    assert elite.id in new_pop.agents
    assert new_pop.get(elite.id) is elite

    # There should be children with mutated IDs
    child_ids = [aid for aid in new_pop.all_ids() if aid != elite.id]
    assert child_ids
    for cid in child_ids:
        assert cid.startswith("A")
        assert "-c" in cid

    # Traits of children should remain within [0,1]
    for cid in child_ids:
        child = new_pop.get(cid)
        for v in child.traits.values():
            assert 0.0 <= v <= 1.0


def test_can_use_as_ga_parent_excludes_reference_agents():
    """Reference agents (can_use_as_ga_parent=False) stay in population but are not used as parents."""
    pop = Population()
    pop.add(
        Agent(id="A0", name="A0", player_counts=[4], elo_global=1600, can_use_as_ga_parent=True)
    )
    pop.add(
        Agent(id="ref", name="ref", player_counts=[4], elo_global=1550, can_use_as_ga_parent=False)
    )
    pop.add(
        Agent(id="A1", name="A1", player_counts=[4], elo_global=1500, can_use_as_ga_parent=True)
    )

    cfg = GAConfig(population_size=4, elite_fraction=0.25, mutation_prob=0.0, mutation_std=0.0)
    rng = random.Random(42)
    new_pop = next_generation(pop, cfg, rng=rng)

    # Reference agent must still be present (unchanged)
    assert "ref" in new_pop.agents
    assert new_pop.get("ref").can_use_as_ga_parent is False
    # Reference agent must never have been used as parent: no child should have "ref" as parent
    for aid, agent in new_pop.agents.items():
        if aid != "ref":
            assert "ref" not in agent.parents

