"""
Bidding (enchères) for 4 players.
Order: Prise < Garde < Garde sans le Chien < Garde contre le Chien.
First to speak = right of dealer; each player speaks once; can pass or bid; higher bid wins.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Callable

from .deal import first_to_bid_4p, first_to_bid_3p, first_to_bid_5p


class Contract(IntEnum):
    """Contract levels in ascending order."""
    PRISE = 1       # Petite
    GARDE = 2
    GARDE_SANS = 3  # Garde sans le Chien
    GARDE_CONTRE = 4  # Garde contre le Chien


CONTRACT_NAMES = {
    Contract.PRISE: "Prise",
    Contract.GARDE: "Garde",
    Contract.GARDE_SANS: "Garde sans le Chien",
    Contract.GARDE_CONTRE: "Garde contre le Chien",
}


def contract_multiplier(contract: Contract) -> int:
    """Score multiplier for the contract."""
    return {Contract.PRISE: 1, Contract.GARDE: 2, Contract.GARDE_SANS: 4, Contract.GARDE_CONTRE: 6}[contract]


def can_take_chien(contract: Contract) -> bool:
    """Prise and Garde: taker receives and uses the Chien. Garde sans/contre: no."""
    return contract in (Contract.PRISE, Contract.GARDE)


def chien_to_defense(contract: Contract) -> bool:
    """Garde contre: Chien is given to the defender facing the taker (counted with Defense)."""
    return contract == Contract.GARDE_CONTRE


class BiddingResult:
    """Result of the bidding phase."""
    __slots__ = ("taker", "contract", "bids")

    def __init__(self, taker: int, contract: Contract, bids: list[tuple[int, int | None]]):
        self.taker = taker  # player index (0..3, 0..2, or 0..4 depending on variant)
        self.contract = contract
        # bids: list of (player_index, bid) where bid is Contract value or None for pass


def run_bidding_4p(
    dealer: int,
    get_bid: Callable[[int, list[tuple[int, int | None]]], int | None],
) -> BiddingResult | None:
    """
    Run the bidding round. get_bid(player_index, history) returns Contract value or None (pass).
    history is list of (player, bid) so far. Returns BiddingResult or None if everyone passed.
    """
    first = first_to_bid_4p(dealer)
    order = [first, (first + 1) % 4, (first + 2) % 4, (first + 3) % 4]
    history: list[tuple[int, int | None]] = []
    current_high: int | None = None
    current_taker: int | None = None

    for player in order:
        bid = get_bid(player, list(history))
        history.append((player, bid))
        if bid is not None:
            if current_high is None or bid > current_high:
                current_high = bid
                current_taker = player

    if current_taker is None or current_high is None:
        return None
    return BiddingResult(taker=current_taker, contract=Contract(current_high), bids=history)


def run_bidding_3p(
    dealer: int,
    get_bid: Callable[[int, list[tuple[int, int | None]]], int | None],
) -> BiddingResult | None:
    """
    Bidding for 3 players. Same contracts, but only 3 seats.
    """
    first = first_to_bid_3p(dealer)
    order = [first, (first + 1) % 3, (first + 2) % 3]
    history: list[tuple[int, int | None]] = []
    current_high: int | None = None
    current_taker: int | None = None

    for player in order:
        bid = get_bid(player, list(history))
        history.append((player, bid))
        if bid is not None:
            if current_high is None or bid > current_high:
                current_high = bid
                current_taker = player

    if current_taker is None or current_high is None:
        return None
    return BiddingResult(taker=current_taker, contract=Contract(current_high), bids=history)


def run_bidding_5p(
    dealer: int,
    get_bid: Callable[[int, list[tuple[int, int | None]]], int | None],
) -> BiddingResult | None:
    """
    Bidding for 5 players. Same contracts, but 5 seats.
    """
    first = first_to_bid_5p(dealer)
    order = [first, (first + 1) % 5, (first + 2) % 5, (first + 3) % 5, (first + 4) % 5]
    history: list[tuple[int, int | None]] = []
    current_high: int | None = None
    current_taker: int | None = None

    for player in order:
        bid = get_bid(player, list(history))
        history.append((player, bid))
        if bid is not None:
            if current_high is None or bid > current_high:
                current_high = bid
                current_taker = player

    if current_taker is None or current_high is None:
        return None
    return BiddingResult(taker=current_taker, contract=Contract(current_high), bids=history)
