"""
Distribution (deal) for 4, 3, and 5 players.
4p: 3 par 3, Chien 6. 3p: 4 par 4, Chien 6. 5p: 3 par 3, Chien 3.
In all cases, first and last card of the pack do not go to the Chien.
"""
from __future__ import annotations

import random
from typing import NamedTuple

from .deck import Card, make_deck_78

# 4 players: dealer (0), right (1), across (2), left (3). First to speak = right of dealer = 1.
# 3 players: dealer (0), right (1), left (2). First to speak = right of dealer = 1.
# 5 players: dealer (0), then players 1..4 clockwise; first to speak = right of dealer = 1.
# Deal order (who gets cards in order, starting from index 0 of the shuffled pack).
DEAL_ORDER_4P = (1, 2, 3, 0)      # right, across, left, dealer
DEAL_ORDER_3P = (1, 2, 0)         # right, left, dealer
DEAL_ORDER_5P = (1, 2, 3, 4, 0)   # right, then around table back to dealer

# Chien indices per variant.
# FFT: first and last card of the pack (indices 0 and 77) do not go to the Chien.
# We pick fixed indices in [1, 76] that respect these constraints.
CHIEN_INDICES_4P = (11, 23, 35, 47, 59, 71)           # 6 cards
CHIEN_INDICES_3P = (10, 22, 34, 46, 58, 70)           # 6 cards
CHIEN_INDICES_5P = (13, 39, 65)                       # 3 cards


class Deal4P(NamedTuple):
    """Result of a 4-player deal. Hands and chien are lists (can be mutated for play)."""
    hands: tuple[list[Card], list[Card], list[Card], list[Card]]  # 0=dealer, 1=right, 2=across, 3=left
    chien: list[Card]
    dealer: int  # 0..3


def deal_4p(deck: list[Card] | None = None, rng: random.Random | None = None) -> Deal4P:
    """
    Deal for 4 players. Each gets 18 cards, Chien gets 6.
    Deck order: first and last card (indices 0 and 77) go to players, not to Chien.
    """
    if deck is None:
        deck = make_deck_78()
    if rng is None:
        rng = random.Random()
    deck = list(deck)
    rng.shuffle(deck)

    chien_set = set(CHIEN_INDICES_4P)
    hands: list[list[Card]] = [[], [], [], []]
    chien: list[Card] = []

    idx = 0
    for i in range(78):
        card = deck[i]
        if i in chien_set:
            chien.append(card)
        else:
            player = DEAL_ORDER_4P[idx % 4]
            hands[player].append(card)
            idx += 1

    return Deal4P(
        hands=(hands[0], hands[1], hands[2], hands[3]),
        chien=chien,
        dealer=0,  # caller can set dealer; for a full game, dealer rotates
    )


class Deal3P(NamedTuple):
    """Result of a 3-player deal. Hands and chien are lists."""
    hands: tuple[list[Card], list[Card], list[Card]]  # 0=dealer, 1=right, 2=left
    chien: list[Card]
    dealer: int  # 0..2


def deal_3p(deck: list[Card] | None = None, rng: random.Random | None = None) -> Deal3P:
    """
    Deal for 3 players. Each gets 24 cards, Chien gets 6.
    Distribution: 4 by 4, counter-clockwise. First and last card (indices 0 and 77) go to players, not to Chien.
    """
    if deck is None:
        deck = make_deck_78()
    if rng is None:
        rng = random.Random()
    deck = list(deck)
    rng.shuffle(deck)

    chien_set = set(CHIEN_INDICES_3P)
    hands: list[list[Card]] = [[], [], []]
    chien: list[Card] = []

    idx = 0
    for i in range(78):
        card = deck[i]
        if i in chien_set:
            chien.append(card)
        else:
            player = DEAL_ORDER_3P[idx % 3]
            hands[player].append(card)
            idx += 1

    return Deal3P(
        hands=(hands[0], hands[1], hands[2]),
        chien=chien,
        dealer=0,
    )


class Deal5P(NamedTuple):
    """Result of a 5-player deal. Hands and chien are lists."""

    hands: tuple[list[Card], list[Card], list[Card], list[Card], list[Card]]  # 0=dealer, 1..4 around table
    chien: list[Card]
    dealer: int  # 0..4


def deal_5p(deck: list[Card] | None = None, rng: random.Random | None = None) -> Deal5P:
    """
    Deal for 5 players. Each gets 15 cards, Chien gets 3.
    Distribution: 3 by 3, counter-clockwise. First and last card (indices 0 and 77) go to players, not to Chien.
    """
    if deck is None:
        deck = make_deck_78()
    if rng is None:
        rng = random.Random()
    deck = list(deck)
    rng.shuffle(deck)

    chien_set = set(CHIEN_INDICES_5P)
    hands: list[list[Card]] = [[], [], [], [], []]
    chien: list[Card] = []

    idx = 0
    for i in range(78):
        card = deck[i]
        if i in chien_set:
            chien.append(card)
        else:
            player = DEAL_ORDER_5P[idx % 5]
            hands[player].append(card)
            idx += 1

    return Deal5P(
        hands=(hands[0], hands[1], hands[2], hands[3], hands[4]),
        chien=chien,
        dealer=0,
    )


def next_dealer_4p(dealer: int) -> int:
    """Dealer rotates in play direction (0 -> 1 -> 2 -> 3 -> 0)."""
    return (dealer + 1) % 4


def next_dealer_3p(dealer: int) -> int:
    """Dealer rotates in play direction (0 -> 1 -> 2 -> 0)."""
    return (dealer + 1) % 3


def next_dealer_5p(dealer: int) -> int:
    """Dealer rotates in play direction (0 -> 1 -> 2 -> 3 -> 4 -> 0)."""
    return (dealer + 1) % 5


def first_to_bid_4p(dealer: int) -> int:
    """Player to the right of the dealer speaks first."""
    return (dealer + 1) % 4


def first_to_play_4p(dealer: int) -> int:
    """Entame: player to the right of the dealer plays first."""
    return (dealer + 1) % 4


def first_to_bid_3p(dealer: int) -> int:
    """3p: player to the right of the dealer speaks first (1, then 2, then 0)."""
    return (dealer + 1) % 3


def first_to_play_3p(dealer: int) -> int:
    """3p entame: player to the right of the dealer plays first."""
    return (dealer + 1) % 3


def first_to_bid_5p(dealer: int) -> int:
    """5p: player to the right of the dealer speaks first (1, then 2,3,4, then 0)."""
    return (dealer + 1) % 5


def first_to_play_5p(dealer: int) -> int:
    """5p entame: player to the right of the dealer plays first."""
    return (dealer + 1) % 5


def petit_sec_4p(hand: list[Card]) -> bool:
    """
    True if the hand has "Petit sec": only one trump and it is the Petit (1), and no Excuse.
    FFT: such a player must announce and the deal is cancelled.
    """
    trumps = [c for c in hand if c.is_trump()]
    excuse = any(c.is_excuse() for c in hand)
    if excuse:
        return False
    if len(trumps) != 1:
        return False
    return trumps[0].is_petit()
