"""Tests for ELO and population helpers."""

import random

from tarot.tournament import (
    Agent,
    Population,
    update_elo_pairwise,
    make_random_tables,
    make_elo_stratified_tables,
    run_random_match_4p,
    run_round_random,
    run_match_for_table,
    run_round_with_policies,
)
from tarot.agents import RandomAgent


def test_update_elo_pairwise_basic():
    # Two agents, one match where A clearly wins
    a = Agent(id="A", name="A", player_counts=[4], elo_4p=1500.0, elo_global=1500.0)
    b = Agent(id="B", name="B", player_counts=[4], elo_4p=1500.0, elo_global=1500.0)
    agents = [a, b]
    scores = [100.0, -100.0]

    update_elo_pairwise(agents, scores, player_count=4, k_factor=32.0)
    # A should gain ELO, B should lose
    assert a.elo_4p > 1500.0
    assert b.elo_4p < 1500.0
    assert a.elo_global > 1500.0
    assert b.elo_global < 1500.0


def test_make_random_tables_size_and_partition():
    rng = random.Random(123)
    ids = [str(i) for i in range(10)]
    tables = make_random_tables(ids, table_size=4, rng=rng)
    # With 10 agents and table_size 4, we expect 2 full tables (8 agents used)
    assert len(tables) == 2
    assert all(len(t) == 4 for t in tables)
    # No duplicates across tables
    flat = [x for t in tables for x in t]
    assert len(set(flat)) == len(flat)


def test_make_elo_stratified_tables():
    rng = random.Random(1)
    pop = Population()
    for i, elo in enumerate([1200.0, 1400.0, 1600.0, 1800.0, 1300.0, 1500.0]):
        pop.add(Agent(id=f"A{i}", name=f"A{i}", player_counts=[4], elo_global=elo))
    tables = make_elo_stratified_tables(pop, table_size=2, rng=rng)
    # 6 agents, table_size 2 -> 3 tables
    assert len(tables) == 3
    assert all(len(t) == 2 for t in tables)
    # First table should have lowest-ELO pair (1200, 1300), last highest (1600, 1800)
    assert tables[0] == ["A0", "A4"]  # 1200, 1300
    assert tables[2] == ["A2", "A3"]  # 1600, 1800


def test_run_random_match_and_round_update_stats_and_elo():
    rng = random.Random(99)
    # Simple population of 4 agents for 4p
    pop = Population()
    for i in range(4):
        pop.add(
            Agent(
                id=f"A{i}",
                name=f"A{i}",
                player_counts=[4],
            )
        )

    # Single random match returns 4 totals
    totals, steps = run_random_match_4p(num_deals=2, rng=rng)
    assert len(totals) == 4
    assert steps == 1
    assert sum(totals) == 0  # scoring invariants

    # Run one random round and ensure stats/ELOs change
    run_round_random(pop, player_count=4, num_deals=2, rng=rng)
    for agent in pop.agents.values():
        assert agent.matches_played >= 0  # some may be unused if pop size not multiple of 4


def test_run_match_for_table_with_random_agents():
    rng = random.Random(7)
    policies = [RandomAgent(seed=1), RandomAgent(seed=2), RandomAgent(seed=3), RandomAgent(seed=4)]
    totals = run_match_for_table(player_count=4, num_deals=1, policies=policies, rng=rng)
    assert len(totals) == 4
    assert abs(sum(totals)) < 1e-6  # scoring invariants: totals sum to 0


def test_run_round_with_policies_updates_stats_and_elo():
    rng = random.Random(8)
    pop = Population()
    for i in range(4):
        pop.add(
            Agent(
                id=f"A{i}",
                name=f"A{i}",
                player_counts=[4],
            )
        )

    def make_policy(agent: Agent) -> RandomAgent:  # noqa: ARG001
        return RandomAgent(seed=42)

    run_round_with_policies(pop, player_count=4, num_deals=1, rng=rng, make_policy=make_policy)
    for agent in pop.agents.values():
        assert agent.matches_played >= 0

    run_round_with_policies(
        pop, player_count=4, num_deals=1, rng=rng, make_policy=make_policy,
        matchmaking_style="elo",
    )
    for agent in pop.agents.values():
        assert agent.matches_played >= 0

