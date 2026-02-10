"""Tarot game engine (FFT official rules)."""

__version__ = "0.1.0"

from .deck import Card, EXCUSE, make_deck_78, Suit
from .deal import (
    deal_4p,
    Deal4P,
    deal_3p,
    Deal3P,
    deal_5p,
    Deal5P,
    first_to_bid_4p,
    first_to_play_4p,
    first_to_bid_3p,
    first_to_play_3p,
    first_to_bid_5p,
    first_to_play_5p,
)
from .bidding import Contract, run_bidding_4p, run_bidding_3p, run_bidding_5p, BiddingResult
from .play import legal_plays, trick_winner
from .scoring import (
    points_in_cards,
    deal_base_score,
    deal_base_score_3p,
    mark_4p_with_taker,
    mark_3p_with_taker,
    mark_5p_with_taker,
)
from .game import (
    play_one_deal_4p,
    run_deal_4p,
    run_match_4p,
    SingleDealState,
    play_one_deal_3p,
    run_deal_3p,
    run_match_3p,
    SingleDealState3P,
    play_one_deal_5p,
    run_deal_5p,
    run_match_5p,
    SingleDealState5P,
)
