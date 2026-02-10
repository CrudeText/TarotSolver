"""
Tiny CLI to run random matches in TarotEnv* environments.

Usage (from project root, after installing in editable mode):
    python -m tarot.play_random
"""
from __future__ import annotations

import argparse
import random
from typing import Type

from .agents import RandomAgent
from .env_game import TarotEnv3P, TarotEnv4P, TarotEnv5P, StepResult


def run_random_match(
    env_cls: Type,
    num_deals: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    env = env_cls(num_deals=num_deals, learning_player=0, rng=rng)
    agent = RandomAgent(seed=seed)

    step: StepResult = env.reset()
    total_reward = 0.0
    steps = 0

    while not step.done and steps < 100_000:
        action = agent.act(step.obs, step.legal_actions_mask)
        step = env.step(action)
        total_reward += step.reward
        steps += 1

    print(
        f"{env_cls.__name__}: num_deals={num_deals}, steps={steps}, "
        f"final_reward_for_player0={total_reward}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random matches in Tarot environments.")
    parser.add_argument(
        "--variant",
        choices=["3p", "4p", "5p", "all"],
        default="4p",
        help="Which player-count variant to run.",
    )
    parser.add_argument(
        "--deals",
        type=int,
        default=3,
        help="Number of deals per match.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    variants = []
    if args.variant == "all":
        variants = [TarotEnv3P, TarotEnv4P, TarotEnv5P]
    elif args.variant == "3p":
        variants = [TarotEnv3P]
    elif args.variant == "4p":
        variants = [TarotEnv4P]
    elif args.variant == "5p":
        variants = [TarotEnv5P]

    for env_cls in variants:
        run_random_match(env_cls, num_deals=args.deals, seed=args.seed)


if __name__ == "__main__":
    main()

