"""Smoke tests for TarotEnv4P/3P/5P."""
import random

from tarot.env_game import TarotEnv4P, TarotEnv3P, TarotEnv5P
from tarot.env import NUM_ACTIONS


def test_env4p_single_match_random_policy():
    # Deterministic RNG for reproducibility
    rng = random.Random(42)
    env = TarotEnv4P(num_deals=3, learning_player=0, rng=rng)

    step = env.reset()
    assert not step.done
    assert len(step.legal_actions_mask) == NUM_ACTIONS

    total_reward = 0.0
    steps = 0

    while not step.done and steps < 10_000:
        # Sample uniformly among legal actions
        legal_indices = [i for i, ok in enumerate(step.legal_actions_mask) if ok]
        assert legal_indices, "There should always be at least one legal action"
        a = rng.choice(legal_indices)
        step = env.step(a)
        total_reward += step.reward
        steps += 1

    assert step.done
    # Reward should be the final match total for the learning player (can be any int)
    assert isinstance(total_reward, float)


def test_env3p_single_match_random_policy():
    rng = random.Random(43)
    env = TarotEnv3P(num_deals=3, learning_player=0, rng=rng)
    step = env.reset()
    assert not step.done
    assert len(step.legal_actions_mask) == NUM_ACTIONS

    steps = 0
    while not step.done and steps < 10_000:
        legal_indices = [i for i, ok in enumerate(step.legal_actions_mask) if ok]
        assert legal_indices
        a = rng.choice(legal_indices)
        step = env.step(a)
        steps += 1

    assert step.done


def test_env5p_single_match_random_policy():
    rng = random.Random(44)
    env = TarotEnv5P(num_deals=3, learning_player=0, rng=rng)
    step = env.reset()
    assert not step.done
    assert len(step.legal_actions_mask) == NUM_ACTIONS

    steps = 0
    while not step.done and steps < 10_000:
        legal_indices = [i for i, ok in enumerate(step.legal_actions_mask) if ok]
        assert legal_indices
        a = rng.choice(legal_indices)
        step = env.step(a)
        steps += 1

    assert step.done

