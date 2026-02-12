"""Tests for league orchestration (tournaments + GA, PPO optional)."""

import random

from tarot.ga import GAConfig
from tarot.league import LeagueConfig, LeagueRunControl, run_league_generation, run_league_generations
from tarot.tournament import Agent, Population


def _make_population(size: int = 4) -> Population:
    pop = Population()
    for i in range(size):
        pop.add(
            Agent(
                id=f"A{i}",
                name=f"A{i}",
                player_counts=[4],
            )
        )
    return pop


def test_run_league_generation_without_ppo_or_ga():
    pop = _make_population()
    cfg = LeagueConfig(
        player_count=4,
        deals_per_match=1,
        rounds_per_generation=1,
        ppo_top_k=0,
        ppo_updates_per_agent=0,
        ga_config=None,
    )

    new_pop, summary = run_league_generation(pop, cfg, rng=random.Random(0))

    # Population should be unchanged when GA is disabled.
    assert new_pop is pop
    # All agents should have had at least zero matches (may be unused if size not multiple of 4).
    assert len(pop.agents) == 4
    # Summary should contain reasonable keys.
    assert "elo_min" in summary and "elo_max" in summary and "elo_mean" in summary


def test_run_league_generation_with_ga_only():
    pop = _make_population()
    cfg = LeagueConfig(
        player_count=4,
        deals_per_match=1,
        rounds_per_generation=1,
        ppo_top_k=0,
        ppo_updates_per_agent=0,
        ga_config=GAConfig(population_size=4, elite_fraction=0.25, mutation_prob=1.0, mutation_std=0.1),
    )

    new_pop, summary = run_league_generation(pop, cfg, rng=random.Random(1))

    # GA should have produced a new population object with the requested size.
    assert new_pop is not pop
    assert len(new_pop.agents) == 4
    assert "elo_min" in summary


def test_run_league_generations_yields_per_generation():
    pop = _make_population()
    cfg = LeagueConfig(
        player_count=4,
        deals_per_match=1,
        rounds_per_generation=1,
        ppo_top_k=0,
        ppo_updates_per_agent=0,
        ga_config=GAConfig(population_size=4, elite_fraction=0.25, mutation_prob=1.0, mutation_std=0.1),
    )
    gen = run_league_generations(pop, cfg, num_generations=3, rng=random.Random(2))
    results = list(gen)
    assert len(results) == 3
    for i, (p, summary, idx) in enumerate(results):
        assert idx == i
        assert "elo_min" in summary and "elo_mean" in summary
        assert len(p.agents) == 4


def test_run_league_generations_respects_cancel():
    pop = _make_population()
    cfg = LeagueConfig(
        player_count=4,
        deals_per_match=1,
        rounds_per_generation=1,
        ppo_top_k=0,
        ga_config=None,
    )
    control = LeagueRunControl()
    control.request_cancel()
    gen = run_league_generations(pop, cfg, num_generations=5, rng=random.Random(3), control=control)
    results = list(gen)
    assert len(results) == 0

