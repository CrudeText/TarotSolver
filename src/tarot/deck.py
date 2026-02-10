"""
Tarot deck: 78 cards (4 suits × 14, 21 trumps, Excuse).
FFT rules: Bouts = Excuse, 1 (Petit), 21. Card values for counting.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class Suit(IntEnum):
    """Pique, Cœur, Carreau, Trèfle. Order used for tie-break (smallest = Trèfle)."""
    SPADES = 0
    HEARTS = 1
    DIAMONDS = 2
    CLUBS = 3


# Rank in a suit: 1=As (lowest), 2..10, 11=Valet, 12=Cavalier, 13=Dame, 14=Roi
RANK_ACE = 1
RANK_VALET = 11
RANK_CAVALIER = 12
RANK_DAME = 13
RANK_ROI = 14

# Bouts (Oudlers): Excuse, Petit (1), 21
BOUT_PETIT = 1
BOUT_21 = 21


@dataclass(frozen=True)
class Card:
    """
    A single tarot card. Either:
    - suited: suit + rank (1=As .. 14=Roi)
    - trump: number 1..21 (1=Petit, 21=strongest)
    - excuse: no suit/rank/trump
    """

    kind: str  # "suit" | "trump" | "excuse"
    suit: Optional[Suit] = None
    rank: Optional[int] = None  # 1..14 for suited
    trump: Optional[int] = None  # 1..21 for trumps

    def __post_init__(self) -> None:
        if self.kind == "suit":
            assert self.suit is not None and self.rank is not None
            assert 1 <= self.rank <= 14
        elif self.kind == "trump":
            assert self.trump is not None
            assert 1 <= self.trump <= 21
        elif self.kind == "excuse":
            assert self.suit is None and self.rank is None and self.trump is None
        else:
            raise ValueError(f"Unknown card kind: {self.kind}")

    def is_excuse(self) -> bool:
        return self.kind == "excuse"

    def is_trump(self) -> bool:
        return self.kind == "trump"

    def is_suit(self) -> bool:
        return self.kind == "suit"

    def is_bout(self) -> bool:
        """True if this card is one of the 3 Bouts (Excuse, 1, 21)."""
        if self.kind == "excuse":
            return True
        if self.kind == "trump" and self.trump in (BOUT_PETIT, BOUT_21):
            return True
        return False

    def is_petit(self) -> bool:
        """True if this is the Petit (trump 1)."""
        return self.kind == "trump" and self.trump == BOUT_PETIT

    def point_value_half(self) -> float:
        """
        Point value in "half-point" units (for 3p/5p ½ point rule).
        Oudler 4.5, Roi 4.5, Dame 3.5, Cavalier 2.5, Valet 1.5, other 0.5.
        """
        if self.kind == "excuse":
            return 4.5
        if self.kind == "trump":
            return 4.5 if self.trump in (BOUT_PETIT, BOUT_21) else 0.5
        # suited
        if self.rank == RANK_ROI:
            return 4.5
        if self.rank == RANK_DAME:
            return 3.5
        if self.rank == RANK_CAVALIER:
            return 2.5
        if self.rank == RANK_VALET:
            return 1.5
        return 0.5

    def __str__(self) -> str:
        if self.kind == "excuse":
            return "Excuse"
        if self.kind == "trump":
            return f"Atout-{self.trump}"
        r = self.rank
        rank_str = {14: "R", 13: "D", 12: "C", 11: "V"}.get(r) or str(r)
        if r == 1:
            rank_str = "A"
        suit_char = "♠♥♦♣"[self.suit]
        return f"{rank_str}{suit_char}"

    def __repr__(self) -> str:
        return str(self)


EXCUSE = Card(kind="excuse")


def make_suit_card(suit: Suit, rank: int) -> Card:
    return Card(kind="suit", suit=suit, rank=rank)


def make_trump_card(number: int) -> Card:
    return Card(kind="trump", trump=number)


def make_deck_78() -> list[Card]:
    """Build a full 78-card tarot deck (order suitable for distribution)."""
    deck: list[Card] = []
    for s in Suit:
        for rank in range(1, 15):
            deck.append(make_suit_card(s, rank))
    for n in range(1, 22):
        deck.append(make_trump_card(n))
    deck.append(EXCUSE)
    return deck


def cards_point_total(cards: list[Card], use_half_points: bool = False) -> float:
    """
    Total points in a set of cards. 91 total per deal.
    If use_half_points=True, sum half-point values (for 3p/5p).
    If False, count in pairs (FFT rule): each pair gives integer points.
    """
    if use_half_points:
        return sum(c.point_value_half() for c in cards)
    half = sum(c.point_value_half() for c in cards)
    return round(half)  # or floor; FFT counts in pairs so total is integer


def minimum_points_for_bouts(num_bouts: int) -> int:
    """Points the taker must reach given number of Bouts in their tricks."""
    return {0: 56, 1: 51, 2: 41, 3: 36}[num_bouts]
