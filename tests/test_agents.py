"""Tests for baseline agents."""

from tarot.agents import RandomAgent


def test_random_agent_respects_legal_mask():
    agent = RandomAgent(seed=123)
    obs = [0.0, 1.0, 2.0]  # dummy; RandomAgent ignores obs content
    legal = [False, True, False, True, False]

    # Sample multiple times to ensure we never pick illegal indices
    for _ in range(50):
        a = agent.act(obs, legal)
        assert a in (1, 3)

