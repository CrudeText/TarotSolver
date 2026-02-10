"""
Score calculation: card points, minimum by Bouts, (écart + 25) × coefficient + primes.
FFT: 91 points per deal; just-made = +25; Prise×1, Garde×2, Garde sans×4, Garde contre×6.
"""
from __future__ import annotations

from .deck import Card, cards_point_total, minimum_points_for_bouts
from .bidding import Contract, contract_multiplier

# Primes (4p)
POIGNEE_SIMPLE = 20
POIGNEE_DOUBLE = 30
POIGNEE_TRIPLE = 40
PETIT_AU_BOUT = 10
CHELEM_ANNOUNCED = 400
CHELEM_NOT_ANNOUNCED = 200
CHELEM_ANNOUNCED_FAILED = -200
CHELEM_DEFENSE = 200  # per defender if defense does slam


def points_in_cards(cards: list[Card], use_half_points: bool = False) -> float:
    """Total points in a set of cards (max 91 per deal)."""
    return cards_point_total(cards, use_half_points=use_half_points)


def taker_made_contract(
    taker_points: float,
    num_bouts: int,
    use_half_points: bool = False,
) -> bool:
    """True if taker reached the minimum (contract made or just made)."""
    minimum = minimum_points_for_bouts(num_bouts)
    if use_half_points:
        return taker_points >= minimum
    return int(round(taker_points)) >= minimum


def deal_base_score(
    taker_points: float,
    num_bouts: int,
    contract: Contract,
    use_half_points: bool = False,
) -> int:
    """
    Base score for the deal: positive = taker won, negative = taker lost.
    Formula: (points_diff + 25) * multiplier. Just made (diff=0) gives +25. If diff < 0, return negative.
    """
    minimum = minimum_points_for_bouts(num_bouts)
    pts = int(round(taker_points))
    diff = pts - minimum
    mult = contract_multiplier(contract)
    raw = (diff + 25) * mult
    return raw if diff >= 0 else -raw


def deal_base_score_3p(
    taker_points_half: float,
    num_bouts: int,
    contract: Contract,
) -> int:
    """
    Base score for 3-player deal: positive = taker won, negative = taker lost.
    Uses half-point rule: 0.5 pt goes to the winning camp.
    Steps:
      - If taker_points_half >= minimum: taker wins; round up (x.5 -> x+1).
      - Else: taker loses; round down (x.5 -> x-1).
      - Then apply same (diff + 25) * multiplier formula.
    """
    minimum = minimum_points_for_bouts(num_bouts)
    if taker_points_half >= minimum:
        pts_int = int(taker_points_half + 0.5)
        diff = pts_int - minimum  # >= 0
        mult = contract_multiplier(contract)
        return (diff + 25) * mult
    pts_int = int(taker_points_half - 0.5)
    diff = pts_int - minimum  # < 0
    mult = contract_multiplier(contract)
    raw = (abs(diff) + 25) * mult
    return -raw


def apply_primes(
    base_score: int,
    petit_au_bout_taker: bool | None,
    poignee_taker_side: bool | None,
    poignee_points: int,
    chelem_points: int,
    contract: Contract,
) -> int:
    """
    petit_au_but_taker: True if taker's side has Petit au Bout, False if defense, None if not applicable.
    poignee_taker_side: True if taker's side showed poignee and won the deal (or defense showed and taker lost).
    poignee_points: 20, 30, or 40.
    chelem_points: 400, 200, -200, or 0.
    Petit au Bout: 10 * multiplier, added or subtracted from base.
    """
    mult = contract_multiplier(contract)
    score = base_score
    if petit_au_bout_taker is not None:
        if petit_au_bout_taker:
            score += PETIT_AU_BOUT * mult
        else:
            score -= PETIT_AU_BOUT * mult
    if poignee_points and poignee_taker_side is not None:
        if poignee_taker_side:
            score += poignee_points
        else:
            score -= poignee_points
    score += chelem_points
    return score


def mark_4p_with_taker(deal_score: int, taker: int) -> tuple[int, int, int, int]:
    """Per-player scores: taker gets 3*deal_score, each defender gets -deal_score. Sum = 0."""
    scores = [-deal_score, -deal_score, -deal_score, -deal_score]
    scores[taker] = 3 * deal_score
    return (scores[0], scores[1], scores[2], scores[3])


def mark_3p_with_taker(deal_score: int, taker: int) -> tuple[int, int, int]:
    """
    Per-player scores for 3 players:
    - taker gets 2 * deal_score
    - each defender gets -deal_score
    Sum = 0.
    """
    scores = [-deal_score, -deal_score, -deal_score]
    scores[taker] = 2 * deal_score
    return (scores[0], scores[1], scores[2])


def mark_5p_with_taker(
    deal_score: int,
    taker: int,
    partner: int | None,
) -> tuple[int, int, int, int, int]:
    """
    Per-player scores for 5 players.

    - If partner is None (taker alone vs 4): taker gets 4 * deal_score, each defender gets -deal_score.
    - If partner is not None (2 vs 3):
        - defense as in 4p: each defender gets -deal_score
        - attackers: 2/3 of total to taker, 1/3 to partner: taker = 2*deal_score, partner = deal_score.

    Sum = 0 in both cases.
    """
    scores = [-deal_score, -deal_score, -deal_score, -deal_score, -deal_score]
    if partner is None:
        scores[taker] = 4 * deal_score
    else:
        scores[taker] = 2 * deal_score
        scores[partner] = deal_score
    return (scores[0], scores[1], scores[2], scores[3], scores[4])
