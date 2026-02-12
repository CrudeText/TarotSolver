"""Tests for population augmentation helpers."""

import random

from tarot.population_helpers import (
    clone_agents,
    generate_random_agents,
    mutate_from_base,
)
from tarot.tournament import Agent, Population


def test_generate_random_agents():
    rng = random.Random(0)
    agents = generate_random_agents(5, [4], rng)
    assert len(agents) == 5
    for i, a in enumerate(agents):
        assert a.id == f"rand{i}"
        assert a.player_counts == [4]
        assert a.checkpoint_path is None
        assert "aggressiveness" in a.traits
        assert "defensiveness" in a.traits
        assert 0.0 <= a.traits["aggressiveness"] <= 1.0
        assert 0.0 <= a.traits["defensiveness"] <= 1.0


def test_mutate_from_base():
    base = [
        Agent(id="B0", name="B0", player_counts=[4], traits={"aggressiveness": 0.5}),
        Agent(id="B1", name="B1", player_counts=[4], traits={"aggressiveness": 0.8}),
    ]
    rng = random.Random(1)
    children = mutate_from_base(base, 4, mutation_prob=0.5, mutation_std=0.1, rng=rng)
    assert len(children) == 4
    ids = {a.id for a in children}
    assert len(ids) == 4
    for c in children:
        assert c.generation == 1
        assert c.parents
        assert c.matches_played == 0
        assert "aggressiveness" in c.traits
        assert 0.0 <= c.traits["aggressiveness"] <= 1.0


def test_clone_agents():
    base = [
        Agent(id="C0", name="C0", player_counts=[4], traits={"x": 0.3}, checkpoint_path="/p"),
    ]
    rng = random.Random(2)
    clones = clone_agents(base, 3, rng)
    assert len(clones) == 3
    for c in clones:
        assert c.traits == {"x": 0.3}
        assert c.checkpoint_path == "/p"
        assert c.matches_played == 0
        assert c.total_match_score == 0.0
    assert len({c.id for c in clones}) == 3
