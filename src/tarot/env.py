"""
Observation / action encoding and (later) environment wrapper for Tarot.

Phase 2, step 1: define **flat observations** that:
- Represent the **full hand** of the current player, card-by-card.
- Keep enough structure that a model can learn patterns like
  "I have the top N cards of this suit" or "I control many high trumps".
- Encode what is on the table and who is taker/partner during play.
- Encode the bidding history and who has spoken during bidding.

We intentionally keep this module free of external RL library dependencies.
It focuses on turning engine state into vectors; a Gymnasium-style Env can
be built on top without changing the encoding.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

from .deck import Card
from .bidding import Contract
from .game import SingleDealState, SingleDealState3P, SingleDealState5P


NUM_CARDS: int = 78
_BIDDING_OBS_SIZE: int = 116  # 78 (hand) + 25 (bids) + 5 (spoken) + 5 (player idx) + 3 (player count)
NUM_BID_ACTIONS: int = 5      # PASS, PRISE, GARDE, GARDE_SANS, GARDE_CONTRE
NUM_CARD_ACTIONS: int = NUM_CARDS
NUM_ACTIONS: int = NUM_BID_ACTIONS + NUM_CARD_ACTIONS  # 5 + 78 = 83


def _one_hot(index: int | None, size: int) -> List[int]:
    vec = [0] * size
    if index is None:
        return vec
    if 0 <= index < size:
        vec[index] = 1
    return vec


def card_index(card: Card) -> int:
    """
    Stable index 0..77 for every card in the deck, matching make_deck_78():
      - 0..55  : suited cards (4 suits × 14 ranks, suit-major then rank 1..14)
      - 56..76 : trumps 1..21
      - 77     : Excuse
    This preserves exact identity and ordering of cards, so a model can infer
    patterns like "top N cards of a suit" from which indices are present.
    """
    if card.is_suit():
        assert card.suit is not None and card.rank is not None
        return int(card.suit) * 14 + (card.rank - 1)
    if card.is_trump():
        assert card.trump is not None
        return 56 + (card.trump - 1)
    # Excuse
    return 77


def encode_card_set(cards: Iterable[Card]) -> List[int]:
    """
    Binary 78-dim vector for a set of cards: 1 if card is present, else 0.
    Used for encoding:
      - current player's hand,
      - cards in tricks / won tricks,
      - chien (dog),
      - etc.
    """
    vec = [0] * NUM_CARDS
    for c in cards:
        idx = card_index(c)
        vec[idx] = 1
    return vec


def encode_hand(hand: Iterable[Card]) -> List[int]:
    """Alias for encode_card_set when used specifically for a player's hand."""
    return encode_card_set(hand)


# ---- Bidding-phase observation (3 / 4 / 5 players) ----


def _encode_bids_and_spoken(
    history: Sequence[tuple[int, int | None]],
) -> tuple[List[int], List[int]]:
    """
    Encode bidding history into:
    - best bid per player (5 players max) as 5-way one-hots each:
        0 = no bid / only passes
        1..4 = Contract enum value
      → 5 * 5 = 25 dims.
    - has_spoken flags for each of 5 players (bool), 5 dims.
    """
    best_bid: List[int | None] = [None] * 5
    spoken: List[bool] = [False] * 5

    for player, bid in history:
        if 0 <= player < 5:
            spoken[player] = True
            if bid is not None:
                best_bid[player] = bid

    bids_vec: List[int] = []
    for p in range(5):
        # 5-way one-hot: index 0 = none/pass, 1..4 = contract value
        val = best_bid[p] if best_bid[p] is not None else 0
        bids_vec.extend(_one_hot(val, 5))

    spoken_vec: List[int] = [1 if s else 0 for s in spoken]
    return bids_vec, spoken_vec


def _encode_bidding_common(
    hand: Sequence[Card],
    history: Sequence[tuple[int, int | None]],
    player_index: int,
    num_players: int,
) -> List[float]:
    """
    Bidding-phase encoding:

    - 78 card bits: current player's hand
    - 25 bits: best bid per player (up to 5 players)
    - 5 bits: has_spoken flags (up to 5 players)
    - 5 bits: current player index (0..4)
    - 3 bits: player count (3,4,5)
    """
    hand_vec = encode_hand(hand)
    bids_vec, spoken_vec = _encode_bids_and_spoken(history)

    meta: List[int] = []
    meta.extend(_one_hot(player_index, 5))

    # Player count (3,4,5) → one-hot of size 3: index 0=3p, 1=4p, 2=5p
    pc_map = {3: 0, 4: 1, 5: 2}
    meta.extend(_one_hot(pc_map.get(num_players), 3))

    vec_int: List[int] = hand_vec + bids_vec + spoken_vec + meta
    # Sanity check: keep constant size for debugging
    assert len(vec_int) == _BIDDING_OBS_SIZE
    return [float(x) for x in vec_int]


def encode_bidding_observation_4p(
    hand: Sequence[Card],
    history: Sequence[tuple[int, int | None]],
    player_index: int,
) -> List[float]:
    """Bidding-phase observation for 4-player deals."""
    return _encode_bidding_common(
        hand=hand,
        history=history,
        player_index=player_index,
        num_players=4,
    )


def encode_bidding_observation_3p(
    hand: Sequence[Card],
    history: Sequence[tuple[int, int | None]],
    player_index: int,
) -> List[float]:
    """Bidding-phase observation for 3-player deals."""
    return _encode_bidding_common(
        hand=hand,
        history=history,
        player_index=player_index,
        num_players=3,
    )


def encode_bidding_observation_5p(
    hand: Sequence[Card],
    history: Sequence[tuple[int, int | None]],
    player_index: int,
) -> List[float]:
    """Bidding-phase observation for 5-player deals."""
    return _encode_bidding_common(
        hand=hand,
        history=history,
        player_index=player_index,
        num_players=5,
    )


# ---- Play-phase observation (3 / 4 / 5 players) ----


def legal_action_mask_bidding(
    history: Sequence[tuple[int, int | None]],
) -> List[bool]:
    """
    Legal-action mask for bidding phase over the global action space.

    - Only indices 0..4 (bidding actions) can ever be legal.
    - We allow any of the 5 bidding options; game-specific rules (e.g. no lower
      bid than current highest) should be enforced in the bidding callback.
    - All card actions (indices >= 5) are masked out.
    """
    mask = [False] * NUM_ACTIONS
    for i in range(NUM_BID_ACTIONS):
        mask[i] = True
    return mask


def _encode_play_common(
    hand: Sequence[Card],
    current_trick: Sequence[tuple[int, Card]],
    taker_tricks: Sequence[Card],
    defense_tricks: Sequence[Card],
    chien: Sequence[Card],
    player_index: int,
    taker_index: int,
    partner_index: int | None,
    num_players: int,
    contract: Contract,
) -> List[float]:
    """
    Common play-phase encoding:

    - 5 × 78 card bits:
        [0:78)   : current player's hand
        [78:156) : current trick (set of cards, order ignored)
        [156:234): taker-side tricks
        [234:312): defense-side tricks
        [312:390): chien (dog) cards that still belong to one side
    - Metadata (discrete, one-hot):
        - current player index: 5 dims (indices 0..num_players-1 may be non-zero)
        - taker index: 5 dims
        - partner index: 5 dims (all zeros if no partner / not applicable)
        - player count one-hot: 3 dims for {3,4,5}
        - contract one-hot: 4 dims (PRISE, GARDE, GARDE_SANS, GARDE_CONTRE)
    """
    hand_vec = encode_hand(hand)
    trick_vec = encode_card_set(c for _, c in current_trick)
    taker_vec = encode_card_set(taker_tricks)
    defense_vec = encode_card_set(defense_tricks)
    chien_vec = encode_card_set(chien)

    meta: List[int] = []
    # Player index and taker/partner (size 5 to cover up to 5 players).
    meta.extend(_one_hot(player_index, 5))
    meta.extend(_one_hot(taker_index, 5))
    meta.extend(_one_hot(partner_index, 5))

    # Player count (3,4,5) → one-hot of size 3: index 0=3p, 1=4p, 2=5p
    pc_map = {3: 0, 4: 1, 5: 2}
    meta.extend(_one_hot(pc_map.get(num_players), 3))

    # Contract one-hot of size 4 (PRISE, GARDE, GARDE_SANS, GARDE_CONTRE)
    c_idx = int(contract) - 1  # Contract is 1..4
    meta.extend(_one_hot(c_idx, 4))

    # Convert everything to floats for RL libraries (0.0 / 1.0)
    vec_int: List[int] = hand_vec + trick_vec + taker_vec + defense_vec + chien_vec + meta
    return [float(x) for x in vec_int]


def encode_play_observation_4p(state: SingleDealState, player_index: int) -> List[float]:
    """Play-phase observation for 4-player deals."""
    return _encode_play_common(
        hand=state.hands[player_index],
        current_trick=state.current_trick,
        taker_tricks=state.taker_tricks,
        defense_tricks=state.defense_tricks,
        chien=state.chien,
        player_index=player_index,
        taker_index=state.taker,
        partner_index=None,
        num_players=4,
        contract=state.contract,
    )


def encode_play_observation_3p(state: SingleDealState3P, player_index: int) -> List[float]:
    """Play-phase observation for 3-player deals."""
    return _encode_play_common(
        hand=state.hands[player_index],
        current_trick=state.current_trick,
        taker_tricks=state.taker_tricks,
        defense_tricks=state.defense_tricks,
        chien=state.chien,
        player_index=player_index,
        taker_index=state.taker,
        partner_index=None,
        num_players=3,
        contract=state.contract,
    )


def encode_play_observation_5p(state: SingleDealState5P, player_index: int) -> List[float]:
    """Play-phase observation for 5-player deals."""
    return _encode_play_common(
        hand=state.hands[player_index],
        current_trick=state.current_trick,
        taker_tricks=state.taker_tricks,
        defense_tricks=state.defense_tricks,
        chien=state.chien,
        player_index=player_index,
        taker_index=state.taker,
        partner_index=state.partner,
        num_players=5,
        contract=state.contract,
    )


def legal_action_mask_play_from_hand_and_legal_cards(
    hand: Sequence[Card],
    legal_cards: Sequence[Card],
) -> List[bool]:
    """
    Compute a legal-action mask for the **play phase** from:
      - the current hand
      - the list of legal cards (subset of hand) from the engine.

    Mapping:
      - Bidding actions (0..4) are always False during play.
      - Card actions (5..82) are True iff that card is in both `hand` and `legal_cards`.
    """
    mask = [False] * NUM_ACTIONS
    legal_set = {id(c) for c in legal_cards}
    # For each card in hand that is legal, mark its card_index as playable.
    for c in hand:
        if id(c) not in legal_set:
            continue
        idx = card_index(c)  # 0..77
        action_idx = NUM_BID_ACTIONS + idx
        if 0 <= action_idx < NUM_ACTIONS:
            mask[action_idx] = True
    return mask


__all__ = [
    "NUM_CARDS",
    "NUM_ACTIONS",
    "NUM_BID_ACTIONS",
    "NUM_CARD_ACTIONS",
    "card_index",
    "encode_card_set",
    "encode_hand",
    "encode_bidding_observation_3p",
    "encode_bidding_observation_4p",
    "encode_bidding_observation_5p",
    "encode_play_observation_3p",
    "encode_play_observation_4p",
    "encode_play_observation_5p",
    "legal_action_mask_bidding",
    "legal_action_mask_play_from_hand_and_legal_cards",
]

