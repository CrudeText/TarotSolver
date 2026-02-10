"""
Population, ELO, and simple tournament orchestration.

This layer is intentionally model-agnostic: it doesn't know about NN libraries,
only about agents identified by IDs with ELO ratings and a callable policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Callable, Dict, List, Sequence

from .agents import Policy
from .bidding import Contract
from .deck import Card
from .env import (
    NUM_ACTIONS,
    NUM_BID_ACTIONS,
    NUM_CARD_ACTIONS,
    card_index,
    encode_play_observation_3p,
    encode_play_observation_4p,
    encode_play_observation_5p,
    legal_action_mask_play_from_hand_and_legal_cards,
)
from .game import run_match_3p, run_match_4p, run_match_5p


AgentId = str


@dataclass
class Agent:
    """Metadata and ratings for one agent in the population."""

    id: AgentId
    name: str
    player_counts: List[int]  # e.g. [3, 4, 5] or a subset
    # Per–player-count ELOs and global ELO
    elo_3p: float = 1500.0
    elo_4p: float = 1500.0
    elo_5p: float = 1500.0
    elo_global: float = 1500.0
    generation: int = 0
    traits: Dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None
    arch_name: str | None = None
    parents: List[AgentId] = field(default_factory=list)

    matches_played: int = 0
    total_match_score: float = 0.0

    def record_match_score(self, score: float) -> None:
        self.matches_played += 1
        self.total_match_score += score


@dataclass
class Population:
    """Collection of agents taking part in training/tournaments."""

    agents: Dict[AgentId, Agent] = field(default_factory=dict)

    def add(self, agent: Agent) -> None:
        self.agents[agent.id] = agent

    def get(self, agent_id: AgentId) -> Agent:
        return self.agents[agent_id]

    def all_ids(self) -> List[AgentId]:
        return list(self.agents.keys())


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score of A vs B under standard Elo."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo_pairwise(
    agents: List[Agent],
    match_scores: Sequence[float],
    player_count: int,
    k_factor: float = 32.0,
    margin_scale: float = 50.0,
) -> None:
    """
    Update ELO ratings from a multi-player match by treating it as pairwise comparisons.

    - For each ordered pair (i, j), we derive a "result" score_ij in [0, 1]
      from the match-score difference (margin-aware), then accumulate Elo
      deltas over all pairs and apply them once per agent.
    """
    n = len(agents)
    assert len(match_scores) == n

    deltas = [0.0 for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s_i = match_scores[i]
            s_j = match_scores[j]
            diff = s_i - s_j
            # Map margin to a result in (0,1) using a logistic curve: big positive
            # margins → ~1, big negative margins → ~0, tie → 0.5.
            score_ij = 1.0 / (1.0 + math.exp(-diff / margin_scale))

            # Use the appropriate per-count ELO for expectation
            if player_count == 3:
                rating_i = agents[i].elo_3p
                rating_j = agents[j].elo_3p
            elif player_count == 4:
                rating_i = agents[i].elo_4p
                rating_j = agents[j].elo_4p
            else:
                rating_i = agents[i].elo_5p
                rating_j = agents[j].elo_5p

            exp_ij = _expected_score(rating_i, rating_j)
            deltas[i] += k_factor * (score_ij - exp_ij)

    for i, agent in enumerate(agents):
        delta = deltas[i]
        if player_count == 3:
            agent.elo_3p += delta
        elif player_count == 4:
            agent.elo_4p += delta
        else:
            agent.elo_5p += delta
        # Global ELO: apply same delta so it reflects aggregate performance
        agent.elo_global += delta


def make_random_tables(
    agent_ids: List[AgentId],
    table_size: int,
    rng: random.Random,
) -> List[List[AgentId]]:
    """
    Split agents into random tables of given size (dropping leftovers if not divisible).
    """
    shuffled = list(agent_ids)
    rng.shuffle(shuffled)
    tables: List[List[AgentId]] = []
    for i in range(0, len(shuffled) - len(shuffled) % table_size, table_size):
        tables.append(shuffled[i : i + table_size])
    return tables


def _random_bid_4p(rng: random.Random) -> int | None:
    options: List[int | None] = [None, int(Contract.PRISE), int(Contract.GARDE)]
    if rng.random() < 0.3:
        options.extend([int(Contract.GARDE_SANS), int(Contract.GARDE_CONTRE)])
    return rng.choice(options)


def _random_play(state, player: int, rng: random.Random) -> Card:
    legal = state.legal_cards(player)
    if not legal:
        raise RuntimeError("No legal plays in random tournament policy")
    return rng.choice(legal)


def run_random_match_4p(num_deals: int, rng: random.Random) -> Tuple[List[float], int]:
    """
    Run a random 4-player match directly via the engine; returns:
      - per-player total match scores (length 4)
      - dummy steps count (always 1, since we call run_match_4p once).
    """

    def get_bid(player: int, history: List[Tuple[int, int | None]]) -> int | None:
        return _random_bid_4p(rng)

    def get_play(state, player: int) -> Card:
        return _random_play(state, player, rng)

    totals, _per_deal = run_match_4p(num_deals, get_bid, get_play, rng=rng)
    return list(totals), 1


def run_random_match_3p(num_deals: int, rng: random.Random) -> Tuple[List[float], int]:
    def get_bid(player: int, history: List[Tuple[int, int | None]]) -> int | None:
        return _random_bid_4p(rng)

    def get_play(state, player: int) -> Card:
        return _random_play(state, player, rng)

    totals, _per_deal = run_match_3p(num_deals, get_bid, get_play, rng=rng)
    return list(totals), 1


def run_random_match_5p(num_deals: int, rng: random.Random) -> Tuple[List[float], int]:
    def get_bid(player: int, history: List[Tuple[int, int | None]]) -> int | None:
        return _random_bid_4p(rng)

    def get_play(state, player: int) -> Card:
        return _random_play(state, player, rng)

    totals, _per_deal = run_match_5p(num_deals, get_bid, get_play, rng=rng)
    return list(totals), 1


def run_round_random(
    pop: Population,
    player_count: int,
    num_deals: int,
    rng: random.Random,
) -> None:
    """
    Run one "round" of random matches for the given player_count (3/4/5).

    For now, all seats play randomly; we still update agents' per-count and global
    ELOs and match stats based on their table results. This gives us a full
    tournament / GA wiring even before plugging in real policies.
    """
    if player_count == 3:
        table_size = 3
        run_match = run_random_match_3p
    elif player_count == 4:
        table_size = 4
        run_match = run_random_match_4p
    else:
        table_size = 5
        run_match = run_random_match_5p

    tables = make_random_tables(pop.all_ids(), table_size=table_size, rng=rng)
    for table_ids in tables:
        agents = [pop.get(aid) for aid in table_ids]
        totals, _ = run_match(num_deals, rng)
        # Update stats and ELOs
        for agent, score in zip(agents, totals):
            agent.record_match_score(score)
        update_elo_pairwise(agents, totals, player_count=player_count)


def _action_to_card_from_hand(action: int, hand: Sequence[Card]) -> Card | None:
    """
    Map a global action index back to a Card instance from a given hand.

    Returns None if the action does not correspond to any card in the hand.
    """
    if not (NUM_BID_ACTIONS <= action < NUM_ACTIONS):
        return None
    card_idx = action - NUM_BID_ACTIONS
    if not (0 <= card_idx < NUM_CARD_ACTIONS):
        return None
    for c in hand:
        if card_index(c) == card_idx:
            return c
    return None


def run_match_for_table(
    player_count: int,
    num_deals: int,
    policies: Sequence[Policy],
    rng: random.Random,
) -> List[float]:
    """
    Run a match for one table using per-seat policies.

    For now, bidding remains a simple random policy; policies control **play
    decisions** only. This already connects Agent → Policy → engine so that
    tournaments can evolve real models instead of purely random ones.
    """
    assert len(policies) == player_count

    if player_count == 4:

        def get_bid(player: int, history):  # noqa: ARG001
            return _random_bid_4p(rng)

        def get_play(state, player: int) -> Card:
            policy = policies[player]
            hand = state.hands[player]
            legal_cards = state.legal_cards(player)
            if not legal_cards:
                raise RuntimeError("No legal plays available in run_match_for_table (4p)")

            obs = encode_play_observation_4p(state, player_index=player)
            mask = legal_action_mask_play_from_hand_and_legal_cards(hand, legal_cards)
            action = policy.act(obs, mask)
            card = _action_to_card_from_hand(action, hand)
            if card is None or card not in legal_cards:
                card = rng.choice(legal_cards)
            return card

        totals, _ = run_match_4p(num_deals, get_bid, get_play, rng=rng)
        return list(totals)

    if player_count == 3:

        def get_bid(player: int, history):  # noqa: ARG001
            return _random_bid_4p(rng)

        def get_play(state, player: int) -> Card:
            policy = policies[player]
            hand = state.hands[player]
            legal_cards = state.legal_cards(player)
            if not legal_cards:
                raise RuntimeError("No legal plays available in run_match_for_table (3p)")

            obs = encode_play_observation_3p(state, player_index=player)
            mask = legal_action_mask_play_from_hand_and_legal_cards(hand, legal_cards)
            action = policy.act(obs, mask)
            card = _action_to_card_from_hand(action, hand)
            if card is None or card not in legal_cards:
                card = rng.choice(legal_cards)
            return card

        totals, _ = run_match_3p(num_deals, get_bid, get_play, rng=rng)
        return list(totals)

    if player_count == 5:

        def get_bid(player: int, history):  # noqa: ARG001
            return _random_bid_4p(rng)

        def get_play(state, player: int) -> Card:
            policy = policies[player]
            hand = state.hands[player]
            legal_cards = state.legal_cards(player)
            if not legal_cards:
                raise RuntimeError("No legal plays available in run_match_for_table (5p)")

            obs = encode_play_observation_5p(state, player_index=player)
            mask = legal_action_mask_play_from_hand_and_legal_cards(hand, legal_cards)
            action = policy.act(obs, mask)
            card = _action_to_card_from_hand(action, hand)
            if card is None or card not in legal_cards:
                card = rng.choice(legal_cards)
            return card

        totals, _ = run_match_5p(num_deals, get_bid, get_play, rng=rng)
        return list(totals)

    raise ValueError(f"Unsupported player_count {player_count}; expected 3, 4, or 5.")


def run_round_with_policies(
    pop: Population,
    player_count: int,
    num_deals: int,
    rng: random.Random,
    make_policy: Callable[[Agent], Policy],
) -> None:
    """
    Run one tournament round using a per-Agent policy factory.

    Each agent at a table is converted into a Policy via ``make_policy``; these
    policies control all **play** decisions for that table. ELO and match stats
    are updated exactly as in ``run_round_random``.
    """
    if player_count not in (3, 4, 5):
        raise ValueError(f"Unsupported player_count {player_count}; expected 3, 4, or 5.")

    table_size = player_count
    tables = make_random_tables(pop.all_ids(), table_size=table_size, rng=rng)
    for table_ids in tables:
        table_agents = [pop.get(aid) for aid in table_ids]
        policies = [make_policy(a) for a in table_agents]
        totals = run_match_for_table(player_count, num_deals, policies, rng)
        for agent, score in zip(table_agents, totals):
            agent.record_match_score(score)
        update_elo_pairwise(table_agents, totals, player_count=player_count)


__all__ = [
    "Agent",
    "AgentId",
    "Population",
    "update_elo_pairwise",
    "make_random_tables",
    "run_random_match_3p",
    "run_random_match_4p",
    "run_random_match_5p",
    "run_round_random",
    "run_match_for_table",
    "run_round_with_policies",
]

