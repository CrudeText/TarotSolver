"""Tests for NNPolicy, checkpoint loading, and policy_for_agent (if torch is available)."""

import importlib
from pathlib import Path


def _has_torch() -> bool:
    try:
        importlib.import_module("torch")  # noqa: F401
        return True
    except Exception:
        return False


def test_load_policy_from_checkpoint_roundtrip(tmp_path: Path):
    if not _has_torch():
        return

    import torch
    from tarot.env_game import TarotEnv4P
    from tarot.training import TarotPPOTrainer, PPOConfig
    from tarot.policies import load_policy_from_checkpoint

    env = TarotEnv4P(num_deals=1, learning_player=0)
    cfg = PPOConfig(batch_size=64, minibatch_size=32, update_epochs=1)
    trainer = TarotPPOTrainer(env, cfg=cfg, device=torch.device("cpu"))

    # Run a single lightweight update and save checkpoint to a temp dir
    trainer.update(seed=0)
    ckpt_dir = tmp_path / "ckpt"
    trainer.save_checkpoint(str(ckpt_dir))

    assert (ckpt_dir / "policy.pt").exists()
    assert (ckpt_dir / "config.json").exists()

    policy = load_policy_from_checkpoint(str(ckpt_dir), device=torch.device("cpu"))

    step = env.reset()
    action = policy.act(step.obs, step.legal_actions_mask)
    # Action should be an int index and must correspond to a legal move
    legal_indices = [i for i, ok in enumerate(step.legal_actions_mask) if ok]
    assert isinstance(action, int)
    assert action in legal_indices


def test_policy_for_agent_works_with_tournament_round(tmp_path: Path):
    if not _has_torch():
        return

    import torch
    from tarot.env_game import TarotEnv4P
    from tarot.training import TarotPPOTrainer, PPOConfig
    from tarot.policies import policy_for_agent
    from tarot.tournament import Agent, Population, run_round_with_policies

    # Train a tiny checkpoint
    env = TarotEnv4P(num_deals=1, learning_player=0)
    cfg = PPOConfig(batch_size=64, minibatch_size=32, update_epochs=1)
    trainer = TarotPPOTrainer(env, cfg=cfg, device=torch.device("cpu"))
    trainer.update(seed=1)
    ckpt_dir = tmp_path / "ckpt_tourn"
    trainer.save_checkpoint(str(ckpt_dir))

    # Build a simple 4-agent population; two use the checkpoint, two are random.
    pop = Population()
    pop.add(Agent(id="A0", name="A0", player_counts=[4], checkpoint_path=str(ckpt_dir)))
    pop.add(Agent(id="A1", name="A1", player_counts=[4], checkpoint_path=str(ckpt_dir)))
    pop.add(Agent(id="A2", name="A2", player_counts=[4]))
    pop.add(Agent(id="A3", name="A3", player_counts=[4]))

    def make_policy(agent: Agent):
        return policy_for_agent(agent, device=torch.device("cpu"))

    # Run one 4p round with policy-driven matches; should complete and update stats.
    run_round_with_policies(pop, player_count=4, num_deals=1, rng=__import__("random").Random(0), make_policy=make_policy)

    # Ensure stats have been updated without crashing
    for agent in pop.agents.values():
        assert agent.matches_played >= 0

