"""Tests for genetic algorithm helpers."""

import random

from tarot.ga import (
    GAConfig,
    combine_agents,
    compute_fitness,
    mutate_agent,
    next_generation,
    select_elites,
)
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


def test_elite_clone_fraction():
    """With elite_clone_fraction > 0, some offspring slots are filled by cloning elites."""
    pop = _make_dummy_population()
    cfg = GAConfig(
        population_size=4,
        elite_fraction=0.25,
        elite_clone_fraction=0.5,  # half of offspring slots = clones
        mutation_prob=0.0,
        mutation_std=0.0,
    )
    rng = random.Random(999)

    new_pop = next_generation(pop, cfg, rng=rng, fitness_fn=compute_fitness)
    assert len(new_pop.agents) == 4

    # Elite A3 (highest ELO) should be kept
    assert "A3" in new_pop.agents
    # Some children should be clones (parent-id-cloneN)
    clone_ids = [aid for aid in new_pop.all_ids() if "-clone" in aid]
    mutate_ids = [aid for aid in new_pop.all_ids() if "-c" in aid and "-clone" not in aid]
    assert len(clone_ids) >= 1
    assert len(clone_ids) + len(mutate_ids) == 3  # 3 offspring slots


def test_combine_agents_average():
    """combine_agents with 'average' produces per-trait mean and records both parents."""
    rng = random.Random(1)
    p1 = Agent(id="P1", name="P1", player_counts=[4], traits={"x": 0.2, "y": 0.8})
    p2 = Agent(id="P2", name="P2", player_counts=[4], traits={"x": 0.8, "y": 0.2})
    child = combine_agents(p1, p2, "child", "average", rng)
    assert child.id == "child"
    assert child.parents == ["P1", "P2"]
    assert child.traits["x"] == 0.5
    assert child.traits["y"] == 0.5
    assert all(0.0 <= v <= 1.0 for v in child.traits.values())


def test_combine_agents_crossover():
    """combine_agents with 'crossover' picks per-trait from one parent; traits in [0,1]."""
    rng = random.Random(2)
    p1 = Agent(id="P1", name="P1", player_counts=[4], traits={"a": 0.0, "b": 1.0})
    p2 = Agent(id="P2", name="P2", player_counts=[4], traits={"a": 1.0, "b": 0.0})
    child = combine_agents(p1, p2, "c1", "crossover", rng)
    assert child.parents == ["P1", "P2"]
    assert child.traits["a"] in (0.0, 1.0)
    assert child.traits["b"] in (0.0, 1.0)
    assert all(0.0 <= v <= 1.0 for v in child.traits.values())


def test_next_generation_count_based_sexual_offspring():
    """Count-based GA: sexual_offspring_count + mutate_count + clone_count fill slots; sexual offspring have two parents."""
    pop = _make_dummy_population()
    # 4 slots: 1 sexual, 2 mutated, 1 cloned
    cfg = GAConfig(
        population_size=4,
        sexual_offspring_count=1,
        mutate_count=2,
        clone_count=1,
        mutation_prob=0.0,
        mutation_std=0.0,
        sexual_parent_with_replacement=True,
        sexual_parent_fitness_weighted=True,
        sexual_trait_combination="average",
    )
    rng = random.Random(42)

    new_pop = next_generation(pop, cfg, rng=rng, fitness_fn=compute_fitness)
    assert len(new_pop.agents) == 4

    sex_ids = [aid for aid in new_pop.all_ids() if aid.startswith("sex-")]
    clone_ids = [aid for aid in new_pop.all_ids() if "-clone" in aid]
    mut_ids = [aid for aid in new_pop.all_ids() if "-c" in aid and "-clone" not in aid and not aid.startswith("sex-")]

    assert len(sex_ids) == 1, "expected one sexual offspring"
    assert len(clone_ids) == 1
    assert len(mut_ids) == 2

    sexual = new_pop.get(sex_ids[0])
    assert len(sexual.parents) == 2, "sexual offspring must have two parents"
    assert all(0.0 <= v <= 1.0 for v in sexual.traits.values())

