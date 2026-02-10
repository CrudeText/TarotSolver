"""Smoke tests for the tarot engine."""
import random

from tarot.bidding import run_bidding_4p, run_bidding_3p, run_bidding_5p, Contract
from tarot.deal import deal_4p, deal_3p, deal_5p, petit_sec_4p
from tarot.deck import make_deck_78
from tarot.game import play_one_deal_4p, SingleDealState
from tarot.play import legal_plays


def test_deck_78():
    deck = make_deck_78()
    assert len(deck) == 78


def test_deal_4p():
    rng = random.Random(42)
    deal = deal_4p(rng=rng)
    assert len(deal.chien) == 6
    for i in range(4):
        assert len(deal.hands[i]) == 18
    all_cards = list(deal.chien)
    for h in deal.hands:
        all_cards.extend(h)
    assert len(all_cards) == 78
    assert len(set(id(c) for c in all_cards)) == 78  # 78 distinct cards


def test_deal_3p():
    rng = random.Random(43)
    deal = deal_3p(rng=rng)
    assert len(deal.chien) == 6
    for i in range(3):
        assert len(deal.hands[i]) == 24
    all_cards = list(deal.chien)
    for h in deal.hands:
        all_cards.extend(h)
    assert len(all_cards) == 78
    assert len(set(id(c) for c in all_cards)) == 78


def test_bidding_all_pass():
    def get_bid(player, history):
        return None
    result = run_bidding_4p(0, get_bid)
    assert result is None


def test_bidding_one_taker():
    def get_bid(player, history):
        return Contract.PRISE if player == 1 else None
    result = run_bidding_4p(0, get_bid)
    assert result is not None
    assert result.taker == 1
    assert result.contract == Contract.PRISE


def test_bidding_3p_one_taker():
    def get_bid(player, history):
        return Contract.PRISE if player == 1 else None
    result = run_bidding_3p(0, get_bid)
    assert result is not None
    assert result.taker == 1
    assert result.contract == Contract.PRISE


def test_legal_plays_empty_trick():
    from tarot.deck import make_suit_card, make_trump_card
    from tarot.deck import Suit
    hand = [make_suit_card(Suit.HEARTS, 10), make_trump_card(5)]
    legal = legal_plays(hand, [])
    assert len(legal) == 2


def test_play_one_deal_random():
    rng = random.Random(123)
    def get_bid(player, history):
        return Contract.PRISE if player == 1 else None
    def get_play(state: SingleDealState, player: int):
        legal = state.legal_cards(player)
        return rng.choice(legal)
    scores, deal, bidding = play_one_deal_4p(get_bid, get_play, rng=rng)
    if bidding is None:
        return
    assert len(scores) == 4
    assert sum(scores) == 0


def test_match():
    from tarot.game import run_match_4p
    rng = random.Random(456)
    def get_bid(player, history):
        return Contract.PRISE if player == 1 else None
    def get_play(state, player):
        return rng.choice(state.legal_cards(player))
    totals, per_deal = run_match_4p(2, get_bid, get_play, rng=rng)
    assert len(totals) == 4
    assert sum(totals) == 0
    assert len(per_deal) == 2


def test_poignee_chelem_callbacks():
    rng = random.Random(789)
    def get_bid(player, history):
        return Contract.PRISE if player == 1 else None
    def get_play(state, player):
        return rng.choice(state.legal_cards(player))
    def get_poignee(state, player):
        n = len([c for c in state.hands[player] if c.is_trump()])
        if any(c.is_excuse() for c in state.hands[player]):
            n += 1
        if n >= 10:
            return (10, 20)
        return None
    scores, _, bidding = play_one_deal_4p(
        get_bid, get_play, rng=rng,
        get_poignee=get_poignee,
        get_chelem=lambda d, b: None,
    )
    if bidding is None:
        return
    assert len(scores) == 4
    assert sum(scores) == 0


def test_poignee_3p_callbacks():
    from tarot.game import play_one_deal_3p, SingleDealState3P

    rng = random.Random(987)

    def get_bid(player, history):
        return Contract.PRISE if player == 1 else None

    def get_play(state: SingleDealState3P, player: int):
        return rng.choice(state.legal_cards(player))

    # Simple 3p poignée rule using 13+ atouts (including Excuse) for 20 pts
    def get_poignee(state: SingleDealState3P, player: int):
        n = len([c for c in state.hands[player] if c.is_trump()])
        if any(c.is_excuse() for c in state.hands[player]):
            n += 1
        if n >= 13:
            return (13, 20)
        return None

    scores, _, bidding = play_one_deal_3p(
        get_bid,
        get_play,
        rng=rng,
        get_poignee=get_poignee,
        get_chelem=lambda d, b: None,
    )
    if bidding is None:
        return
    assert len(scores) == 3
    assert sum(scores) == 0


def test_play_one_deal_3p_random():
    from tarot.game import play_one_deal_3p, SingleDealState3P
    rng = random.Random(321)
    def get_bid(player, history):
        return Contract.PRISE if player == 1 else None
    def get_play(state: SingleDealState3P, player: int):
        legal = state.legal_cards(player)
        return rng.choice(legal)
    scores, deal, bidding = play_one_deal_3p(get_bid, get_play, rng=rng)
    if bidding is None:
        return
    assert len(scores) == 3
    assert sum(scores) == 0


def test_match_3p():
    from tarot.game import run_match_3p
    rng = random.Random(654)
    def get_bid(player, history):
        return Contract.PRISE if player == 1 else None
    def get_play(state, player):
        return rng.choice(state.legal_cards(player))
    totals, per_deal = run_match_3p(2, get_bid, get_play, rng=rng)
    assert len(totals) == 3
    assert sum(totals) == 0
    assert len(per_deal) == 2


def test_deal_5p():
    rng = random.Random(44)
    deal = deal_5p(rng=rng)
    assert len(deal.chien) == 3
    for i in range(5):
        assert len(deal.hands[i]) == 15
    all_cards = list(deal.chien)
    for h in deal.hands:
        all_cards.extend(h)
    assert len(all_cards) == 78
    assert len(set(id(c) for c in all_cards)) == 78


def test_bidding_5p_one_taker():
    def get_bid(player, history):
        return Contract.PRISE if player == 1 else None

    result = run_bidding_5p(0, get_bid)
    assert result is not None
    assert result.taker == 1
    assert result.contract == Contract.PRISE


def test_play_one_deal_5p_random_alone():
    from tarot.game import play_one_deal_5p, SingleDealState5P

    rng = random.Random(777)

    def get_bid(player, history):
        return Contract.PRISE if player == 1 else None

    def get_play(state: SingleDealState5P, player: int):
        legal = state.legal_cards(player)
        return rng.choice(legal)

    # No partner callback => taker plays alone (1 vs 4)
    scores, deal, bidding, partner = play_one_deal_5p(
        get_bid,
        get_play,
        rng=rng,
        get_partner=None,
        get_poignee=None,
        get_chelem=None,
    )
    if bidding is None:
        return
    assert len(scores) == 5
    assert sum(scores) == 0
    # In 1v4, partner is None
    assert partner is None


def test_play_one_deal_5p_random_with_partner():
    from tarot.game import play_one_deal_5p, SingleDealState5P

    rng = random.Random(888)

    def get_bid(player, history):
        return Contract.PRISE if player == 1 else None

    def get_play(state: SingleDealState5P, player: int):
        legal = state.legal_cards(player)
        return rng.choice(legal)

    # Simple partner rule: player 3 is always partner when someone takes
    def get_partner(deal, bidding):
        return 3

    scores, deal, bidding, partner = play_one_deal_5p(
        get_bid,
        get_play,
        rng=rng,
        get_partner=get_partner,
        get_poignee=None,
        get_chelem=None,
    )
    if bidding is None:
        return
    assert len(scores) == 5
    assert sum(scores) == 0
    assert partner in range(5)
