"""
Command-line interface for training and evaluating Tarot RL agents.

Usage examples (after installing with the ``rl`` extra):

    python -m tarot.cli train-ppo-4p --updates 10 --deals-per-match 5 \\
        --batch-size 1024 --minibatch-size 256 --checkpoint-dir checkpoints/run1
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import torch

from .env_game import TarotEnv4P
from .training import PPOConfig, TarotPPOTrainer
from .policies import load_policy_from_checkpoint
from .league import LeagueConfig, run_league_generation
from .ga import GAConfig
from .tournament import Agent, Population


def _add_train_ppo_4p_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "train-ppo-4p",
        help="Train a 4-player Tarot policy with custom PPO.",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=100,
        help="Number of PPO update cycles to run.",
    )
    parser.add_argument(
        "--deals-per-match",
        type=int,
        default=5,
        help="Number of deals per match (episode length in deals).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Number of environment steps per PPO batch.",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=256,
        help="Minibatch size for PPO updates.",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=4,
        help="Number of PPO epochs per update.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for environment and training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Torch device string, e.g. "cpu" or "cuda".',
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/ppo_4p_run",
        help="Directory where the final checkpoint will be saved.",
    )
    parser.set_defaults(func=_cmd_train_ppo_4p)


def _cmd_train_ppo_4p(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    env = TarotEnv4P(num_deals=args.deals_per_match, learning_player=0)
    cfg = PPOConfig(
        obs_dim=412,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        max_grad_norm=0.5,
    )
    trainer = TarotPPOTrainer(env, cfg=cfg, device=device)

    for i in range(1, args.updates + 1):
        stats = trainer.update(seed=args.seed + i)
        print(
            f"[update {i}/{args.updates}] "
            f"loss={stats.get('loss', 0.0):.4f} "
            f"policy={stats.get('policy_loss', 0.0):.4f} "
            f"value={stats.get('value_loss', 0.0):.4f} "
            f"entropy={stats.get('entropy', 0.0):.4f}",
            flush=True,
        )

    out_dir = Path(args.checkpoint_dir)
    trainer.save_checkpoint(str(out_dir))
    print(f"Saved checkpoint to {out_dir.resolve()}")


def _add_eval_4p_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "eval-4p",
        help="Evaluate a 4-player policy checkpoint vs random opponents.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Checkpoint directory produced by train-ppo-4p.",
    )
    parser.add_argument(
        "--matches",
        type=int,
        default=20,
        help="Number of matches to play for evaluation.",
    )
    parser.add_argument(
        "--deals-per-match",
        type=int,
        default=5,
        help="Deals per match during evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Torch device string, e.g. "cpu" or "cuda".',
    )
    parser.set_defaults(func=_cmd_eval_4p)


def _cmd_eval_4p(args: argparse.Namespace) -> None:
    import random

    device = torch.device(args.device)
    env = TarotEnv4P(num_deals=args.deals_per_match, learning_player=0, rng=random.Random(args.seed))
    policy = load_policy_from_checkpoint(args.checkpoint_dir, device=device, deterministic=False)

    total_return = 0.0
    for match in range(1, args.matches + 1):
        step = env.reset()
        ep_return = 0.0
        while not step.done:
            action = policy.act(step.obs, step.legal_actions_mask)
            step = env.step(action)
            ep_return += step.reward
        total_return += ep_return
        print(f"[match {match}/{args.matches}] return={ep_return:.1f}")

    avg_return = total_return / float(args.matches)
    print(f"Average return over {args.matches} matches: {avg_return:.2f}")


def _add_league_4p_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "league-4p",
        help="Run a 4-player league: tournaments + optional PPO + GA.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Number of league generations to run.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=8,
        help="Number of agents in the population.",
    )
    parser.add_argument(
        "--rounds-per-generation",
        type=int,
        default=3,
        help="Tournament rounds per generation.",
    )
    parser.add_argument(
        "--deals-per-match",
        type=int,
        default=3,
        help="Deals per match in league tournaments.",
    )
    parser.add_argument(
        "--elite-fraction",
        type=float,
        default=0.25,
        help="Fraction of elites kept unchanged by GA.",
    )
    parser.add_argument(
        "--mutation-prob",
        type=float,
        default=0.5,
        help="Trait mutation probability per gene.",
    )
    parser.add_argument(
        "--mutation-std",
        type=float,
        default=0.1,
        help="Trait mutation std-dev (values are clamped to [0,1]).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for league tournaments and GA.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/league_4p_run",
        help="Directory where per-generation summaries will be written.",
    )
    parser.set_defaults(func=_cmd_league_4p)


def _cmd_league_4p(args: argparse.Namespace) -> None:
    import json
    import random

    rng = random.Random(args.seed)

    # Initial population: simple random agents with no checkpoints yet.
    pop = Population()
    for i in range(args.population_size):
        pop.add(
            Agent(
                id=f"A{i}",
                name=f"A{i}",
                player_counts=[4],
            )
        )

    ga_cfg = GAConfig(
        population_size=args.population_size,
        elite_fraction=args.elite_fraction,
        mutation_prob=args.mutation_prob,
        mutation_std=args.mutation_std,
    )

    league_cfg = LeagueConfig(
        player_count=4,
        deals_per_match=args.deals_per_match,
        rounds_per_generation=args.rounds_per_generation,
        ppo_top_k=0,
        ppo_updates_per_agent=0,
        ga_config=ga_cfg,
    )

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for gen in range(args.generations):
        pop, summary = run_league_generation(pop, league_cfg, rng=rng)
        print(
            f"[generation {gen}] "
            f"Elo min={summary['elo_min']:.1f} "
            f"mean={summary['elo_mean']:.1f} "
            f"max={summary['elo_max']:.1f}",
            flush=True,
        )

        gen_file = out_root / f"generation_{gen:03d}.json"
        data = {
            "generation": gen,
            "league_config": {
                "player_count": league_cfg.player_count,
                "deals_per_match": league_cfg.deals_per_match,
                "rounds_per_generation": league_cfg.rounds_per_generation,
                "fitness_weight_global_elo": league_cfg.fitness_weight_global_elo,
                "fitness_weight_avg_score": league_cfg.fitness_weight_avg_score,
                "ga_config": {
                    "population_size": ga_cfg.population_size,
                    "elite_fraction": ga_cfg.elite_fraction,
                    "mutation_prob": ga_cfg.mutation_prob,
                    "mutation_std": ga_cfg.mutation_std,
                },
            },
            "summary": summary,
            "agents": [
                {
                    "id": a.id,
                    "name": a.name,
                    "player_counts": a.player_counts,
                    "elo_3p": a.elo_3p,
                    "elo_4p": a.elo_4p,
                    "elo_5p": a.elo_5p,
                    "elo_global": a.elo_global,
                    "generation": a.generation,
                    "traits": a.traits,
                    "checkpoint_path": a.checkpoint_path,
                    "arch_name": a.arch_name,
                    "parents": a.parents,
                    "matches_played": a.matches_played,
                    "total_match_score": a.total_match_score,
                }
                for a in pop.agents.values()
            ],
        }
        with gen_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tarot", description="Tarot RL and tournament CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_train_ppo_4p_parser(subparsers)
    _add_eval_4p_parser(subparsers)
    _add_league_4p_parser(subparsers)
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

