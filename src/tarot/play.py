"""
Trick-taking: legal moves, winner, Excuse handling.
FFT: follow suit or trump; at trump must overtrump or undertrump; Excuse doesn't win.
"""
from __future__ import annotations

from .deck import Card, EXCUSE, Suit


def led_suit_and_highest_trump(trick: list[tuple[int, Card]]) -> tuple[Suit | None, int | None]:
    """
    From a trick (list of (player, card)), return (led_suit, highest_trump_number).
    Led suit is from the first non-Excuse card (suit or trump). If first card is Excuse, led_suit stays None until next card.
    """
    led_suit: Suit | None = None
    highest_trump: int | None = None
    for _, c in trick:
        if c.is_excuse():
            continue
        if c.is_trump():
            if led_suit is None:
                led_suit = None  # we're in trump mode
            if highest_trump is None or c.trump > highest_trump:
                highest_trump = c.trump
        else:
            if led_suit is None:
                led_suit = c.suit
    return led_suit, highest_trump


def effective_led_suit(trick: list[tuple[int, Card]]) -> str | None:
    """
    "suit", "trump", or None (if only Excuse played so far).
    """
    for _, c in trick:
        if c.is_excuse():
            continue
        if c.is_trump():
            return "trump"
        return "suit"
    return None


def has_suit(hand: list[Card], suit: Suit) -> bool:
    return any(c.is_suit() and c.suit == suit for c in hand)


def has_trump(hand: list[Card]) -> bool:
    return any(c.is_trump() for c in hand)


def trumps_in_hand(hand: list[Card]) -> list[Card]:
    return [c for c in hand if c.is_trump()]


def can_overtrump(hand: list[Card], highest: int) -> bool:
    return any(c.is_trump() and c.trump > highest for c in hand)


def legal_plays(
    hand: list[Card],
    trick: list[tuple[int, Card]],
) -> list[Card]:
    """
    Return list of cards that can be legally played from hand given current trick.
    trick: list of (player_index, card) in order played.
    """
    if not trick:
        return list(hand)

    eff = effective_led_suit(trick)
    _, highest_trump = led_suit_and_highest_trump(trick)

    if eff is None:
        # Only Excuse so far: any card
        return list(hand)

    if eff == "trump":
        # Must follow trump: must overtrump if possible, else can play any trump
        trumps = trumps_in_hand(hand)
        if not trumps:
            return list(hand)  # no trump: discard any
        if highest_trump is None:
            return list(hand)
        over = [c for c in trumps if c.trump > highest_trump]
        if over:
            return over
        return trumps

    # Led suit is a suit (Suit enum)
    led_s = led_suit_and_highest_trump(trick)[0]
    if led_s is None:
        return list(hand)
    if has_suit(hand, led_s):
        return [c for c in hand if c.is_suit() and c.suit == led_s]
    if has_trump(hand):
        # Must cut (play trump). If previous cut, must overtrump or undertrump
        trumps = trumps_in_hand(hand)
        if highest_trump is not None and can_overtrump(hand, highest_trump):
            return [c for c in trumps if c.trump > highest_trump]
        return trumps
    return list(hand)


def _beats(card: Card, other: Card, led_suit: Suit | None, eff: str) -> bool:
    """True if card beats other in the trick."""
    if other.is_excuse():
        return True  # Excuse never wins
    if card.is_excuse():
        return False
    if eff == "trump":
        return card.is_trump() and (not other.is_trump() or card.trump > other.trump)
    # Led is suit
    if card.is_trump():
        return not other.is_trump() or card.trump > other.trump
    if other.is_trump():
        return False
    if card.suit == led_suit and other.suit == led_suit:
        return card.rank > other.rank
    if card.suit == led_suit:
        return True
    return False


def trick_winner(trick: list[tuple[int, Card]]) -> int:
    """
    Index of the player who wins the trick (0..3).
    Excuse never wins (unless Chelem). Trumps beat suit; highest trump wins; else highest of led suit.
    """
    eff = effective_led_suit(trick)
    led_suit, _ = led_suit_and_highest_trump(trick)
    if eff is None:
        return trick[0][0]
    best_player = trick[0][0]
    best_card = trick[0][1]
    for p, c in trick[1:]:
        if c.is_excuse():
            continue
        if _beats(c, best_card, led_suit, eff):
            best_card = c
            best_player = p
    return best_player


def count_bouts_in_cards(cards: list[Card]) -> int:
    return sum(1 for c in cards if c.is_bout())
