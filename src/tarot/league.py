"""
League orchestration: tournaments + optional PPO fine-tuning + GA evolution.

The core entry point is ``run_league_generation``, which:

- runs a configurable number of tournament rounds using `run_round_with_policies`
- optionally fine-tunes the top-K agents with PPO (if torch is available)
- applies GA `next_generation` to produce the next population

For multi-generation runs with pause/cancel support, use ``run_league_generations``.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
import threading
from typing import Dict, Iterator, List, Tuple

from .ga import GAConfig, compute_fitness, next_generation
from .policies import policy_for_agent
from .tournament import Agent, Population, run_round_with_policies


@dataclass
class LeagueConfig:
    """Configuration for a single league generation."""

    player_count: int = 4
    deals_per_match: int = 5
    rounds_per_generation: int = 3
    # PPO fine-tuning (optional; 0 disables)
    ppo_top_k: int = 0
    ppo_updates_per_agent: int = 0
    # GA configuration (optional; if None, population is kept as-is)
    ga_config: GAConfig | None = None
    # Fitness weights for compute_fitness
    fitness_weight_global_elo: float = 1.0
    fitness_weight_avg_score: float = 0.0


def _fitness_fn_from_config(cfg: LeagueConfig):
    def _fitness(agent: Agent) -> float:
        return compute_fitness(
            agent,
            weight_global_elo=cfg.fitness_weight_global_elo,
            weight_avg_score=cfg.fitness_weight_avg_score,
        )

    return _fitness


def _run_tournament_rounds(
    pop: Population,
    cfg: LeagueConfig,
    rng: random.Random,
) -> None:
    """Run tournament rounds in-place, updating ELOs and match stats."""
    import torch

    def make_policy(agent: Agent):
        return policy_for_agent(agent, device=torch.device("cpu"))

    for _ in range(cfg.rounds_per_generation):
        run_round_with_policies(
            pop,
            player_count=cfg.player_count,
            num_deals=cfg.deals_per_match,
            rng=rng,
            make_policy=make_policy,
        )


def _ppo_finetune_top_agents(
    pop: Population,
    cfg: LeagueConfig,
    rng: random.Random,
) -> None:
    """
    Optional local PPO refinement for the top-K agents by fitness.

    This is a lightweight hook; if torch is not installed or ppo_top_k is 0,
    this function is a no-op.
    """
    if cfg.ppo_top_k <= 0 or cfg.ppo_updates_per_agent <= 0:
        return

    try:
        import torch
        from .env_game import TarotEnv4P
        from .training import PPOConfig, TarotPPOTrainer
        from .training import TarotPPOTrainer as _Trainer  # noqa: F401
    except Exception:
        # Torch or training components not available; skip PPO.
        return

    fitness_fn = _fitness_fn_from_config(cfg)
    agents_sorted: List[Tuple[Agent, float]] = sorted(
        [(a, fitness_fn(a)) for a in pop.agents.values()],
        key=lambda af: af[1],
        reverse=True,
    )
    top_agents = [a for a, _ in agents_sorted[: cfg.ppo_top_k] if a.checkpoint_path]
    if not top_agents:
        return

    device = torch.device("cpu")

    for idx, agent in enumerate(top_agents):
        # Build a small env for fine-tuning this agent.
        env_rng = random.Random(rng.random())
        env = TarotEnv4P(num_deals=cfg.deals_per_match, learning_player=0, rng=env_rng)

        # Start from the agent's current checkpoint if present; else, skip.
        from .training import TarotPPOTrainer

        ppo_cfg = PPOConfig()
        trainer = TarotPPOTrainer(env, cfg=ppo_cfg, device=device)

        # Run a few PPO updates; we don't care about stats here, only improvement.
        for _ in range(cfg.ppo_updates_per_agent):
            trainer.update()

        # Save updated checkpoint to a per-agent directory and point agent to it.
        # For now we use a simple naming scheme; GUI/CLI can manage directories later.
        ckpt_dir = f"checkpoints/league_4p_agent_{agent.id}"
        trainer.save_checkpoint(ckpt_dir)
        agent.checkpoint_path = ckpt_dir


def run_league_generation(
    pop: Population,
    cfg: LeagueConfig,
    rng: random.Random | None = None,
) -> Tuple[Population, Dict[str, float]]:
    """
    Run one league "generation" on the given population.

    Steps:
      1. Tournament rounds (update ELOs and match stats in-place on `pop`).
      2. Optional PPO fine-tuning of top-K agents (also in-place).
      3. GA evolution via `next_generation` (if cfg.ga_config is set).

    Returns:
      - new_pop: the resulting Population (possibly the same as input if GA disabled)
      - summary: dictionary with simple aggregate metrics (min/mean/max global ELO).
    """
    rng = rng or random.Random()

    # 1) Tournament rounds
    _run_tournament_rounds(pop, cfg, rng)

    # 2) Optional PPO fine-tuning
    _ppo_finetune_top_agents(pop, cfg, rng)

    # Compute simple summary metrics on the current population
    elos = [a.elo_global for a in pop.agents.values()]
    summary = {
        "elo_min": min(elos) if elos else 0.0,
        "elo_mean": sum(elos) / len(elos) if elos else 0.0,
        "elo_max": max(elos) if elos else 0.0,
        "num_agents": float(len(pop.agents)),
    }

    # 3) GA evolution (optional)
    if cfg.ga_config is None:
        return pop, summary

    fitness_fn = _fitness_fn_from_config(cfg)
    new_pop = next_generation(pop, cfg.ga_config, rng=rng, fitness_fn=fitness_fn)
    return new_pop, summary


class LeagueRunControl:
    """
    Thread-safe control for multi-generation league runs.

    Set cancel_requested from another thread to stop the run at the next
    generation boundary. Pause is handled by the caller: stop iterating
    on run_league_generations to pause.
    """

    def __init__(self) -> None:
        self.cancel_requested = threading.Event()

    def request_cancel(self) -> None:
        self.cancel_requested.set()


def run_league_generations(
    pop: Population,
    cfg: LeagueConfig,
    num_generations: int,
    rng: random.Random | None = None,
    control: LeagueRunControl | None = None,
) -> Iterator[Tuple[Population, Dict[str, float], int]]:
    """
    Run multiple league generations, yielding after each one.

    Yields (population, summary, generation_index) after each generation.
    The caller can stop iterating to pause. If control is provided and
    control.cancel_requested is set, the generator stops at the next
    generation boundary.

    Args:
        pop: Initial population.
        cfg: League configuration.
        num_generations: Number of generations to run.
        rng: Random generator.
        control: Optional control for cancel signalling.

    Yields:
        (population, summary_dict, generation_index) for each completed generation.
    """
    rng = rng or random.Random()
    current_pop = pop

    for gen_idx in range(num_generations):
        if control and control.cancel_requested.is_set():
            return

        new_pop, summary = run_league_generation(current_pop, cfg, rng=rng)
        yield new_pop, summary, gen_idx
        current_pop = new_pop


__all__ = ["LeagueConfig", "LeagueRunControl", "run_league_generation", "run_league_generations"]

