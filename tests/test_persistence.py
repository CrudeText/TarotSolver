"""Tests for population serialization."""

from tarot.persistence import (
    population_from_dict,
    population_from_json,
    population_to_dict,
    population_to_json,
)
from tarot.tournament import Agent, Population


def _make_population() -> Population:
    pop = Population()
    pop.add(
        Agent(
            id="A0",
            name="Agent Zero",
            player_counts=[3, 4, 5],
            elo_3p=1500.0,
            elo_4p=1520.0,
            elo_5p=1480.0,
            elo_global=1500.0,
            generation=2,
            traits={"aggressiveness": 0.7, "defensiveness": 0.3},
            checkpoint_path="checkpoints/a0",
            arch_name="tarot_mlp_v1",
            parents=["A0-c1"],
            can_use_as_ga_parent=True,
            matches_played=10,
            total_match_score=150.0,
        )
    )
    pop.add(
        Agent(
            id="ref",
            name="Reference",
            player_counts=[4],
            elo_global=1550.0,
            can_use_as_ga_parent=False,
            matches_played=50,
        )
    )
    return pop


def test_round_trip_dict():
    pop = _make_population()
    d = population_to_dict(pop)
    restored = population_from_dict(d)

    assert len(restored.agents) == len(pop.agents)
    for aid, agent in pop.agents.items():
        r = restored.get(aid)
        assert r.id == agent.id
        assert r.name == agent.name
        assert r.player_counts == agent.player_counts
        assert r.elo_3p == agent.elo_3p
        assert r.elo_4p == agent.elo_4p
        assert r.elo_5p == agent.elo_5p
        assert r.elo_global == agent.elo_global
        assert r.generation == agent.generation
        assert r.traits == agent.traits
        assert r.checkpoint_path == agent.checkpoint_path
        assert r.arch_name == agent.arch_name
        assert r.parents == agent.parents
        assert r.can_use_as_ga_parent == agent.can_use_as_ga_parent
        assert r.matches_played == agent.matches_played
        assert r.total_match_score == agent.total_match_score


def test_round_trip_json():
    pop = _make_population()
    s = population_to_json(pop)
    restored = population_from_json(s)
    assert len(restored.agents) == len(pop.agents)
    for aid in pop.agents:
        assert restored.get(aid).id == pop.get(aid).id
        assert restored.get(aid).can_use_as_ga_parent == pop.get(aid).can_use_as_ga_parent


def test_metadata_preserved():
    pop = _make_population()
    meta = {"league_config": {"player_count": 4}}
    d = population_to_dict(pop, metadata=meta)
    assert d["metadata"] == meta
