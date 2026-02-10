"""Tests for observation / encoding helpers in tarot.env."""
import random

from tarot.deck import make_deck_78
from tarot.env import (
    NUM_CARDS,
    NUM_ACTIONS,
    NUM_BID_ACTIONS,
    NUM_CARD_ACTIONS,
    card_index,
    encode_card_set,
    encode_hand,
    encode_bidding_observation_3p,
    encode_bidding_observation_4p,
    encode_bidding_observation_5p,
    encode_play_observation_3p,
    encode_play_observation_4p,
    encode_play_observation_5p,
    legal_action_mask_bidding,
    legal_action_mask_play_from_hand_and_legal_cards,
)
from tarot.bidding import Contract, run_bidding_4p, run_bidding_3p, run_bidding_5p
from tarot.deal import deal_4p, deal_3p, deal_5p
from tarot.game import SingleDealState, SingleDealState3P, SingleDealState5P


def test_card_index_covers_full_deck_without_collision():
    deck = make_deck_78()
    assert len(deck) == NUM_CARDS
    indices = [card_index(c) for c in deck]
    assert sorted(indices) == list(range(NUM_CARDS))
    # Action space should align: one bid region + one card region
    assert NUM_BID_ACTIONS == 5
    assert NUM_CARD_ACTIONS == NUM_CARDS
    assert NUM_ACTIONS == NUM_BID_ACTIONS + NUM_CARD_ACTIONS


def test_encode_hand_and_set_dimension_and_bits():
    deck = make_deck_78()
    hand = deck[:18]
    vec = encode_hand(hand)
    assert len(vec) == NUM_CARDS
    assert sum(vec) == len(hand)
    # encode_card_set should be identical for the same set
    vec2 = encode_card_set(hand)
    assert vec2 == vec


def test_encode_bidding_observation_shapes():
    # 4 players
    rng = random.Random(11)
    deal4 = deal_4p(rng=rng)
    history4 = [(0, None), (1, int(Contract.PRISE)), (2, None), (3, None)]
    obs4 = encode_bidding_observation_4p(deal4.hands[0], history4, player_index=0)
    assert len(obs4) == 116

    # 3 players
    rng = random.Random(12)
    deal3 = deal_3p(rng=rng)
    history3 = [(0, None), (1, int(Contract.GARDE)), (2, None)]
    obs3 = encode_bidding_observation_3p(deal3.hands[1], history3, player_index=1)
    assert len(obs3) == 116

    # 5 players
    rng = random.Random(13)
    deal5 = deal_5p(rng=rng)
    history5 = [(0, None), (1, int(Contract.PRISE)), (2, None), (3, None), (4, None)]
    obs5 = encode_bidding_observation_5p(deal5.hands[2], history5, player_index=2)
    assert len(obs5) == 116


def test_legal_action_mask_bidding_and_play_do_not_overlap():
    # Bidding mask: only first NUM_BID_ACTIONS actions should be True
    mask_bid = legal_action_mask_bidding(history=[])
    assert len(mask_bid) == NUM_ACTIONS
    assert all(mask_bid[i] for i in range(NUM_BID_ACTIONS))
    assert not any(mask_bid[i] for i in range(NUM_BID_ACTIONS, NUM_ACTIONS))

    # Play mask: only card actions can be True
    deck = make_deck_78()
    hand = deck[:5]
    legal = hand[:3]
    mask_play = legal_action_mask_play_from_hand_and_legal_cards(hand, legal)
    assert len(mask_play) == NUM_ACTIONS
    # No bid actions allowed
    assert not any(mask_play[i] for i in range(NUM_BID_ACTIONS))
    # At least as many True entries as len(legal), in the card region
    assert sum(1 for i in range(NUM_BID_ACTIONS, NUM_ACTIONS) if mask_play[i]) == len(legal)


def _make_state_4p() -> SingleDealState:
    rng = random.Random(123)
    deal = deal_4p(rng=rng)

    def get_bid(player, history):
        return Contract.PRISE if player == 1 else None

    bidding = run_bidding_4p(deal.dealer, get_bid)
    assert bidding is not None
    return SingleDealState(deal, bidding)


def _make_state_3p() -> SingleDealState3P:
    rng = random.Random(321)
    deal = deal_3p(rng=rng)

    def get_bid(player, history):
        return Contract.GARDE if player == 2 else None

    bidding = run_bidding_3p(deal.dealer, get_bid)
    assert bidding is not None
    return SingleDealState3P(deal, bidding)


def _make_state_5p(partner: int | None = None) -> SingleDealState5P:
    rng = random.Random(456)
    deal = deal_5p(rng=rng)

    def get_bid(player, history):
        return Contract.PRISE if player == 0 else None

    bidding = run_bidding_5p(deal.dealer, get_bid)
    assert bidding is not None
    # If partner is None, taker plays alone; else 2 vs 3
    return SingleDealState5P(deal, bidding, partner)


def test_encode_play_observation_4p_shape_and_hand_bits():
    state = _make_state_4p()
    obs = encode_play_observation_4p(state, player_index=0)
    # 5 × 78 card bits + 5 + 5 + 5 + 3 + 4 metadata = 390 + 22 = 412
    assert len(obs) == 412
    hand_vec = obs[0:NUM_CARDS]
    assert sum(hand_vec) == len(state.hands[0])


def test_encode_play_observation_3p_shape_and_hand_bits():
    state = _make_state_3p()
    obs = encode_play_observation_3p(state, player_index=1)
    assert len(obs) == 412
    hand_vec = obs[0:NUM_CARDS]
    assert sum(hand_vec) == len(state.hands[1])


def test_encode_play_observation_5p_shape_and_hand_bits_alone_and_with_partner():
    # Alone (no partner)
    state_alone = _make_state_5p(partner=None)
    obs_alone = encode_play_observation_5p(state_alone, player_index=0)
    assert len(obs_alone) == 412
    hand_vec_alone = obs_alone[0:NUM_CARDS]
    assert sum(hand_vec_alone) == len(state_alone.hands[0])

    # With partner
    state_with_partner = _make_state_5p(partner=3)
    obs_with_partner = encode_play_observation_5p(state_with_partner, player_index=0)
    assert len(obs_with_partner) == 412
    hand_vec_with_partner = obs_with_partner[0:NUM_CARDS]
    assert sum(hand_vec_with_partner) == len(state_with_partner.hands[0])

