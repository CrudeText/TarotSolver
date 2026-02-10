"""
Single deal and match orchestration: deal → bid → chien/écart → play → score.
4 players fully implemented; 3 players implemented with half-point scoring,
full Excuse handling, Petit au Bout, Poignée, and Chelem (no 5-player yet).
"""
from __future__ import annotations

import random
from typing import Callable

from .bidding import BiddingResult, Contract, run_bidding_4p, run_bidding_3p, run_bidding_5p
from .deal import (
    Deal4P,
    Deal3P,
    Deal5P,
    deal_4p,
    deal_3p,
    deal_5p,
    first_to_play_4p,
    first_to_play_3p,
    first_to_play_5p,
    next_dealer_4p,
    next_dealer_3p,
    next_dealer_5p,
    petit_sec_4p,
)
from .deck import Card, EXCUSE, make_deck_78
from .play import legal_plays, trick_winner
from .scoring import (
    CHELEM_ANNOUNCED,
    CHELEM_ANNOUNCED_FAILED,
    CHELEM_NOT_ANNOUNCED,
    CHELEM_DEFENSE,
    apply_primes,
    deal_base_score,
    deal_base_score_3p,
    mark_4p_with_taker,
    mark_3p_with_taker,
    mark_5p_with_taker,
    points_in_cards,
)
from .play import count_bouts_in_cards


def _lowest_value_card(cards: list[Card]) -> Card | None:
    """Pick a card 'sans valeur' (low point value) for Excuse exchange. Prefer 0.5 pt cards."""
    if not cards:
        return None
    return min(cards, key=lambda c: (c.point_value_half(), str(c)))


class SingleDealState:
    """Mutable state for one deal: hands, chien, tricks, current trick, Excuse, poignée, chelem."""

    def __init__(self, deal: Deal4P, bidding: BiddingResult):
        self.hands = [list(h) for h in deal.hands]
        self.chien = list(deal.chien)
        self.dealer = deal.dealer
        self.taker = bidding.taker
        self.contract = bidding.contract
        self.taker_tricks: list[Card] = []
        self.defense_tricks: list[Card] = []
        self.current_trick: list[tuple[int, Card]] = []
        self.leader: int = first_to_play_4p(deal.dealer)
        # Trick counters (for Petit au Bout / Chelem)
        self.trick_count: int = 0  # number of completed tricks
        self.taker_trick_count: int = 0
        self.defense_trick_count: int = 0
        self.petit_au_bout_taker: bool | None = None
        # Excuse: when Excuse is played and that camp has no tricks yet, we delay the exchange
        self.pending_excuse: tuple[int, bool] | None = None  # (player who played Excuse, is_taker_side)
        # Poignée: (points value 20/30/40, announced by taker_side)
        self.poignee_points: int = 0
        self.poignee_taker_side: bool | None = None
        # Chelem: who announced (leads first), and outcome for scoring
        self.chelem_announcer: int | None = None
        self.chelem_points: int = 0

    def current_player(self) -> int:
        return (self.leader + len(self.current_trick)) % 4

    def is_taker(self, player: int) -> bool:
        return player == self.taker

    def play_card(self, player: int, card: Card) -> None:
        hand = self.hands[player]
        if card not in hand:
            raise ValueError(f"Card {card} not in hand")
        hand.remove(card)
        self.current_trick.append((player, card))

        if len(self.current_trick) == 4:
            winner = trick_winner(self.current_trick)
            trick_cards = list(self.current_trick)

            # Increment trick counters (for Chelem / Petit au Bout)
            self.trick_count += 1
            if self.is_taker(winner):
                self.taker_trick_count += 1
            else:
                self.defense_trick_count += 1

            # Distribute cards: Excuse does not go to winner; it goes to Excuse-player's camp (with possible exchange)
            excuse_player: int | None = None
            for p, c in trick_cards:
                if c.is_excuse():
                    excuse_player = p
                    break

            # Add non-Excuse cards to winner's camp
            for _, c in trick_cards:
                if c.is_excuse():
                    continue
                if self.is_taker(winner):
                    self.taker_tricks.append(c)
                else:
                    self.defense_tricks.append(c)

            # Handle Excuse (if played in this trick, or pending from a previous trick)
            if excuse_player is not None:
                excuse_taker_side = self.is_taker(excuse_player)
                from_pile = self.taker_tricks if excuse_taker_side else self.defense_tricks
                to_pile = self.defense_tricks if excuse_taker_side else self.taker_tricks
                if from_pile:
                    from_pile.append(EXCUSE)
                    low = _lowest_value_card([c for c in from_pile if not c.is_excuse()])
                    if low is not None:
                        from_pile.remove(low)
                        to_pile.append(low)
                else:
                    self.pending_excuse = (excuse_player, excuse_taker_side)
            elif self.pending_excuse is not None:
                pend_player, pend_side = self.pending_excuse
                if self.is_taker(winner) == pend_side:
                    from_pile = self.taker_tricks if pend_side else self.defense_tricks
                    to_pile = self.defense_tricks if pend_side else self.taker_tricks
                    from_pile.append(EXCUSE)
                    low = _lowest_value_card([c for c in from_pile if not c.is_excuse()])
                    if low is not None:
                        from_pile.remove(low)
                        to_pile.append(low)
                    self.pending_excuse = None

            # Petit au Bout: if Petit is in the last trick (18th)
            if self.trick_count == 18:
                for _, c in trick_cards:
                    if c.is_petit():
                        self.petit_au_bout_taker = self.is_taker(winner)
                        break

            self.current_trick = []
            self.leader = winner

    def legal_cards(self, player: int) -> list[Card]:
        return legal_plays(self.hands[player], self.current_trick)


def _count_trumps(hand: list[Card]) -> int:
    """Number of trumps (Excuse can replace one for poignée)."""
    n = sum(1 for c in hand if c.is_trump())
    if any(c.is_excuse() for c in hand):
        n += 1
    return n


def run_deal_4p(
    deal: Deal4P,
    bidding: BiddingResult,
    get_play: Callable[[SingleDealState, int], Card],
    rng: random.Random | None = None,
    get_poignee: Callable[[SingleDealState, int], tuple[int, int] | None] | None = None,
    get_chelem: Callable[[Deal4P, BiddingResult], int | None] | None = None,
) -> tuple[int, int, int, int]:
    """
    Run one deal after deal and bidding are done. Handles chien/écart, optional poignée and chelem.
    get_poignee(state, player) -> (num_atouts, points) e.g. (10, 20) or None.
    get_chelem(deal, bidding) -> player_index who announces chelem (and will lead), or None.
    Returns (score_p0, score_p1, score_p2, score_p3).
    """
    state = SingleDealState(deal, bidding)
    if bidding.contract in (Contract.PRISE, Contract.GARDE):
        state.hands[state.taker].extend(state.chien)
        state.chien.clear()
        hand = state.hands[state.taker]
        discardable = [c for c in hand if not c.is_bout() and not (c.is_suit() and c.rank == 14)]
        if len(discardable) >= 6:
            for c in discardable[:6]:
                hand.remove(c)
        else:
            for _ in range(6):
                if hand:
                    hand.pop()

    # Poignée: before first card, each player (taker first then 1,2,3) may announce 10/13/15 atouts
    if get_poignee is not None:
        for player in [state.taker, (state.taker + 1) % 4, (state.taker + 2) % 4, (state.taker + 3) % 4]:
            if state.poignee_points > 0:
                break
            out = get_poignee(state, player)
            if out is not None:
                num_atouts, points = out
                state.poignee_points = points
                state.poignee_taker_side = state.is_taker(player)

    # Chelem: if announced, announcer leads
    if get_chelem is not None:
        announcer = get_chelem(deal, bidding)
        if announcer is not None:
            state.chelem_announcer = announcer
            state.leader = announcer

    for _ in range(18):
        for _ in range(4):
            player = state.current_player()
            legal = state.legal_cards(player)
            card = get_play(state, player)
            if card not in legal:
                raise ValueError(f"Illegal play {card}; legal {legal}")
            state.play_card(player, card)

    if state.pending_excuse is not None:
        _, pend_side = state.pending_excuse
        if pend_side:
            state.taker_tricks.append(EXCUSE)
        else:
            state.defense_tricks.append(EXCUSE)
        state.pending_excuse = None

    taker_final = list(state.taker_tricks)
    defense_final = list(state.defense_tricks)
    if state.contract == Contract.GARDE_SANS:
        taker_final = taker_final + state.chien
    elif state.contract == Contract.GARDE_CONTRE:
        defense_final = defense_final + state.chien

    # Chelem primes (18 tricks = 72 cards per side)
    n_taker_tricks = len(taker_final) // 4
    n_defense_tricks = len(defense_final) // 4
    if n_taker_tricks == 18:
        state.chelem_points = CHELEM_ANNOUNCED if state.chelem_announcer is not None else CHELEM_NOT_ANNOUNCED
    elif n_defense_tricks == 18:
        state.chelem_points = -CHELEM_DEFENSE
    elif state.chelem_announcer is not None:
        state.chelem_points = CHELEM_ANNOUNCED_FAILED

    taker_pts = points_in_cards(taker_final)
    num_bouts = count_bouts_in_cards(taker_final)
    base = deal_base_score(taker_pts, num_bouts, state.contract)
    poignee_benefit_taker = None
    if state.poignee_points > 0 and state.poignee_taker_side is not None:
        poignee_benefit_taker = (state.poignee_taker_side and base > 0) or (
            not state.poignee_taker_side and base < 0
        )
    final_score = apply_primes(
        base,
        petit_au_bout_taker=state.petit_au_bout_taker,
        poignee_taker_side=poignee_benefit_taker,
        poignee_points=state.poignee_points or 0,
        chelem_points=state.chelem_points,
        contract=state.contract,
    )
    return mark_4p_with_taker(final_score, state.taker)


def play_one_deal_4p(
    get_bid: Callable[[int, list[tuple[int, int | None]]], int | None],
    get_play: Callable[[SingleDealState, int], Card],
    dealer: int = 0,
    rng: random.Random | None = None,
    get_poignee: Callable[[SingleDealState, int], tuple[int, int] | None] | None = None,
    get_chelem: Callable[[Deal4P, BiddingResult], int | None] | None = None,
) -> tuple[tuple[int, int, int, int], Deal4P | None, BiddingResult | None]:
    """
    Deal, bid, play one deal. If everyone passes or Petit sec, returns (0,0,0,0), deal, None.
    Optional get_poignee and get_chelem for poignée and chelem announcements.
    """
    if rng is None:
        rng = random.Random()
    deck = make_deck_78()
    deal = deal_4p(deck=deck, rng=rng)
    deal = Deal4P(hands=deal.hands, chien=deal.chien, dealer=dealer)
    for hand in deal.hands:
        if petit_sec_4p(hand):
            return (0, 0, 0, 0), deal, None
    bidding = run_bidding_4p(deal.dealer, get_bid)
    if bidding is None:
        return (0, 0, 0, 0), deal, None
    scores = run_deal_4p(
        deal, bidding, get_play, rng,
        get_poignee=get_poignee,
        get_chelem=get_chelem,
    )
    return scores, deal, bidding


def run_match_4p(
    num_deals: int,
    get_bid: Callable[[int, list[tuple[int, int | None]]], int | None],
    get_play: Callable[[SingleDealState, int], Card],
    rng: random.Random | None = None,
    get_poignee: Callable[[SingleDealState, int], tuple[int, int] | None] | None = None,
    get_chelem: Callable[[Deal4P, BiddingResult], int | None] | None = None,
) -> tuple[tuple[int, int, int, int], list[tuple[int, int, int, int]]]:
    """
    Run a match of num_deals. Dealer rotates 0->1->2->3->0. Redistributes if everyone passes or Petit sec.
    Returns (total_p0, total_p1, total_p2, total_p3), list of per-deal scores.
    """
    if rng is None:
        rng = random.Random()
    totals = [0, 0, 0, 0]
    per_deal: list[tuple[int, int, int, int]] = []
    dealer = 0
    played = 0
    while played < num_deals:
        scores, _deal, bidding = play_one_deal_4p(
            get_bid, get_play, dealer=dealer, rng=rng,
            get_poignee=get_poignee,
            get_chelem=get_chelem,
        )
        if bidding is None:
            continue
        per_deal.append(scores)
        for i in range(4):
            totals[i] += scores[i]
        played += 1
        dealer = next_dealer_4p(dealer)
    return (totals[0], totals[1], totals[2], totals[3]), per_deal


# ---- 3 players: full support (same bidding/contracts, 4x4 deal, ½-point scoring) ----


class SingleDealState3P:
    """State for one 3-player deal (with Excuse, Petit au Bout, Poignée, Chelem)."""

    def __init__(self, deal: Deal3P, bidding: BiddingResult):
        self.hands = [list(h) for h in deal.hands]
        self.chien = list(deal.chien)
        self.dealer = deal.dealer
        self.taker = bidding.taker
        self.contract = bidding.contract
        self.taker_tricks: list[Card] = []
        self.defense_tricks: list[Card] = []
        self.current_trick: list[tuple[int, Card]] = []
        self.leader: int = first_to_play_3p(deal.dealer)
        # Trick counters (for Petit au Bout / Chelem)
        self.trick_count: int = 0
        self.taker_trick_count: int = 0
        self.defense_trick_count: int = 0
        self.petit_au_bout_taker: bool | None = None
        # Excuse: when Excuse is played and that camp has no tricks yet, we delay the exchange
        self.pending_excuse: tuple[int, bool] | None = None  # (player who played Excuse, is_taker_side)
        # Poignée: (points value 20/30/40, announced by taker_side)
        self.poignee_points: int = 0
        self.poignee_taker_side: bool | None = None
        # Chelem: who announced (leads first), and outcome for scoring
        self.chelem_announcer: int | None = None
        self.chelem_points: int = 0

    def current_player(self) -> int:
        return (self.leader + len(self.current_trick)) % 3

    def is_taker(self, player: int) -> bool:
        return player == self.taker

    def play_card(self, player: int, card: Card) -> None:
        hand = self.hands[player]
        if card not in hand:
            raise ValueError(f"Card {card} not in hand")
        hand.remove(card)
        self.current_trick.append((player, card))

        if len(self.current_trick) == 3:
            winner = trick_winner(self.current_trick)
            trick_cards = list(self.current_trick)

            # Increment trick counters (for Chelem / Petit au Bout)
            self.trick_count += 1
            if self.is_taker(winner):
                self.taker_trick_count += 1
            else:
                self.defense_trick_count += 1

            # Distribute cards: Excuse does not go to winner; it goes to Excuse-player's camp (with possible exchange)
            excuse_player: int | None = None
            for p, c in trick_cards:
                if c.is_excuse():
                    excuse_player = p
                    break

            # Add non-Excuse cards to winner's camp
            for _, c in trick_cards:
                if c.is_excuse():
                    continue
                if self.is_taker(winner):
                    self.taker_tricks.append(c)
                else:
                    self.defense_tricks.append(c)

            # Handle Excuse (if played in this trick, or pending from a previous trick)
            if excuse_player is not None:
                excuse_taker_side = self.is_taker(excuse_player)
                from_pile = self.taker_tricks if excuse_taker_side else self.defense_tricks
                to_pile = self.defense_tricks if excuse_taker_side else self.taker_tricks
                if from_pile:
                    from_pile.append(EXCUSE)
                    low = _lowest_value_card([c for c in from_pile if not c.is_excuse()])
                    if low is not None:
                        from_pile.remove(low)
                        to_pile.append(low)
                else:
                    self.pending_excuse = (excuse_player, excuse_taker_side)
            elif self.pending_excuse is not None:
                pend_player, pend_side = self.pending_excuse
                if self.is_taker(winner) == pend_side:
                    from_pile = self.taker_tricks if pend_side else self.defense_tricks
                    to_pile = self.defense_tricks if pend_side else self.taker_tricks
                    from_pile.append(EXCUSE)
                    low = _lowest_value_card([c for c in from_pile if not c.is_excuse()])
                    if low is not None:
                        from_pile.remove(low)
                        to_pile.append(low)
                    self.pending_excuse = None

            # Petit au Bout: if Petit is in the last trick (24th)
            if self.trick_count == 24:
                for _, c in trick_cards:
                    if c.is_petit():
                        self.petit_au_bout_taker = self.is_taker(winner)
                        break

            self.current_trick = []
            self.leader = winner

    def legal_cards(self, player: int) -> list[Card]:
        return legal_plays(self.hands[player], self.current_trick)


def run_deal_3p(
    deal: Deal3P,
    bidding: BiddingResult,
    get_play: Callable[[SingleDealState3P, int], Card],
    rng: random.Random | None = None,
    get_poignee: Callable[[SingleDealState3P, int], tuple[int, int] | None] | None = None,
    get_chelem: Callable[[Deal3P, BiddingResult], int | None] | None = None,
) -> tuple[int, int, int]:
    """
    Run one 3-player deal after deal and bidding.
    Handles chien/écart, optional poignée and chelem, Petit au Bout, Excuse exchanges, ½-point rule.
    """
    state = SingleDealState3P(deal, bidding)

    # Prise/Garde: taker takes chien and discards 6 (reuse 4p simple rule)
    if bidding.contract in (Contract.PRISE, Contract.GARDE):
        state.hands[state.taker].extend(state.chien)
        state.chien.clear()
        hand = state.hands[state.taker]
        discardable = [c for c in hand if not c.is_bout() and not (c.is_suit() and c.rank == 14)]
        if len(discardable) >= 6:
            for c in discardable[:6]:
                hand.remove(c)
        else:
            for _ in range(6):
                if hand:
                    hand.pop()

    # Poignée: before first card, each player (taker first then others) may announce 13/15/18 atouts.
    # Thresholds/points are determined by the callback; we just apply the points.
    if get_poignee is not None:
        for player in [state.taker, (state.taker + 1) % 3, (state.taker + 2) % 3]:
            if state.poignee_points > 0:
                break
            out = get_poignee(state, player)
            if out is not None:
                num_atouts, points = out
                # num_atouts is unused here but included for parity with 4p callback shape
                state.poignee_points = points
                state.poignee_taker_side = state.is_taker(player)

    # Chelem: if announced, announcer leads
    if get_chelem is not None:
        announcer = get_chelem(deal, bidding)
        if announcer is not None:
            state.chelem_announcer = announcer
            state.leader = announcer

    # Play 24 tricks (3 players, 4x4 distribution)
    for _ in range(24):
        for _ in range(3):
            player = state.current_player()
            legal = state.legal_cards(player)
            card = get_play(state, player)
            if card not in legal:
                raise ValueError(f"Illegal play {card}; legal {legal}")
            state.play_card(player, card)

    # If an Excuse exchange was pending and never resolved, give Excuse to that camp
    if state.pending_excuse is not None:
        _, pend_side = state.pending_excuse
        if pend_side:
            state.taker_tricks.append(EXCUSE)
        else:
            state.defense_tricks.append(EXCUSE)
        state.pending_excuse = None

    taker_final = list(state.taker_tricks)
    defense_final = list(state.defense_tricks)
    if state.contract == Contract.GARDE_SANS:
        taker_final = taker_final + state.chien
    elif state.contract == Contract.GARDE_CONTRE:
        defense_final = defense_final + state.chien

    taker_pts_half = points_in_cards(taker_final, use_half_points=True)
    num_bouts = count_bouts_in_cards(taker_final)
    base = deal_base_score_3p(taker_pts_half, num_bouts, state.contract)

    # Poignée benefit: side that announced gets the prime if they also win, else they lose it
    poignee_benefit_taker = None
    if state.poignee_points > 0 and state.poignee_taker_side is not None:
        poignee_benefit_taker = (state.poignee_taker_side and base > 0) or (
            not state.poignee_taker_side and base < 0
        )

    final_score = apply_primes(
        base,
        petit_au_bout_taker=state.petit_au_bout_taker,
        poignee_taker_side=poignee_benefit_taker,
        poignee_points=state.poignee_points or 0,
        chelem_points=state.chelem_points,
        contract=state.contract,
    )
    return mark_3p_with_taker(final_score, state.taker)


def play_one_deal_3p(
    get_bid: Callable[[int, list[tuple[int, int | None]]], int | None],
    get_play: Callable[[SingleDealState3P, int], Card],
    dealer: int = 0,
    rng: random.Random | None = None,
    get_poignee: Callable[[SingleDealState3P, int], tuple[int, int] | None] | None = None,
    get_chelem: Callable[[Deal3P, BiddingResult], int | None] | None = None,
) -> tuple[tuple[int, int, int], Deal3P | None, BiddingResult | None]:
    """
    Deal, bid, play one 3-player deal. If everyone passes, returns (0,0,0), deal, None.
    Optional get_poignee and get_chelem for poignée and chelem announcements.
    """
    if rng is None:
        rng = random.Random()
    deck = make_deck_78()
    deal = deal_3p(deck=deck, rng=rng)
    deal = Deal3P(hands=deal.hands, chien=deal.chien, dealer=dealer)
    bidding = run_bidding_3p(deal.dealer, get_bid)
    if bidding is None:
        return (0, 0, 0), deal, None
    scores = run_deal_3p(
        deal,
        bidding,
        get_play,
        rng,
        get_poignee=get_poignee,
        get_chelem=get_chelem,
    )
    return scores, deal, bidding


def run_match_3p(
    num_deals: int,
    get_bid: Callable[[int, list[tuple[int, int | None]]], int | None],
    get_play: Callable[[SingleDealState3P, int], Card],
    rng: random.Random | None = None,
    get_poignee: Callable[[SingleDealState3P, int], tuple[int, int] | None] | None = None,
    get_chelem: Callable[[Deal3P, BiddingResult], int | None] | None = None,
) -> tuple[tuple[int, int, int], list[tuple[int, int, int]]]:
    """
    Run a 3-player match. Dealer rotates 0->1->2->0. Redistributes if everyone passes.
    """
    if rng is None:
        rng = random.Random()
    totals = [0, 0, 0]
    per_deal: list[tuple[int, int, int]] = []
    dealer = 0
    played = 0
    while played < num_deals:
        scores, _deal, bidding = play_one_deal_3p(
            get_bid,
            get_play,
            dealer=dealer,
            rng=rng,
            get_poignee=get_poignee,
            get_chelem=get_chelem,
        )
        if bidding is None:
            continue
        per_deal.append(scores)
        for i in range(3):
            totals[i] += scores[i]
        played += 1
        dealer = next_dealer_3p(dealer)
    return (totals[0], totals[1], totals[2]), per_deal


# ---- 5 players: full support (same contracts, 3x3 deal, ½-point scoring, 1v4 or 2v3) ----


class SingleDealState5P:
    """State for one 5-player deal (Excuse, Petit au Bout, Poignée, Chelem)."""

    def __init__(self, deal: Deal5P, bidding: BiddingResult, partner: int | None):
        self.hands = [list(h) for h in deal.hands]
        self.chien = list(deal.chien)
        self.dealer = deal.dealer
        self.taker = bidding.taker
        self.partner = partner
        self.contract = bidding.contract
        self.taker_tricks: list[Card] = []
        self.defense_tricks: list[Card] = []
        self.current_trick: list[tuple[int, Card]] = []
        self.leader: int = first_to_play_5p(deal.dealer)
        # Trick counters (for Petit au Bout / Chelem)
        self.trick_count: int = 0
        self.taker_trick_count: int = 0
        self.defense_trick_count: int = 0
        self.petit_au_bout_taker_side: bool | None = None
        # Excuse: when Excuse is played and that camp has no tricks yet, we delay the exchange
        self.pending_excuse: tuple[int, bool] | None = None  # (player who played Excuse, is_attack_side)
        # Poignée: (points value 20/30/40, announced by attack/defense side)
        self.poignee_points: int = 0
        self.poignee_attack_side: bool | None = None
        # Chelem: who announced (leads first), and outcome for scoring
        self.chelem_announcer: int | None = None
        self.chelem_points: int = 0

    def current_player(self) -> int:
        return (self.leader + len(self.current_trick)) % 5

    def is_attack_side(self, player: int) -> bool:
        if self.partner is None:
            return player == self.taker
        return player == self.taker or player == self.partner

    def play_card(self, player: int, card: Card) -> None:
        hand = self.hands[player]
        if card not in hand:
            raise ValueError(f"Card {card} not in hand")
        hand.remove(card)
        self.current_trick.append((player, card))

        if len(self.current_trick) == 5:
            winner = trick_winner(self.current_trick)
            trick_cards = list(self.current_trick)

            # Increment trick counters (for Chelem / Petit au Bout)
            self.trick_count += 1
            if self.is_attack_side(winner):
                self.taker_trick_count += 1
            else:
                self.defense_trick_count += 1

            # Distribute cards: Excuse does not go to winner; it goes to Excuse-player's camp (with possible exchange)
            excuse_player: int | None = None
            for p, c in trick_cards:
                if c.is_excuse():
                    excuse_player = p
                    break

            # Add non-Excuse cards to winner's camp
            for _, c in trick_cards:
                if c.is_excuse():
                    continue
                if self.is_attack_side(winner):
                    self.taker_tricks.append(c)
                else:
                    self.defense_tricks.append(c)

            # Handle Excuse (if played in this trick, or pending from a previous trick)
            if excuse_player is not None:
                excuse_attack_side = self.is_attack_side(excuse_player)
                from_pile = self.taker_tricks if excuse_attack_side else self.defense_tricks
                to_pile = self.defense_tricks if excuse_attack_side else self.taker_tricks
                if from_pile:
                    from_pile.append(EXCUSE)
                    low = _lowest_value_card([c for c in from_pile if not c.is_excuse()])
                    if low is not None:
                        from_pile.remove(low)
                        to_pile.append(low)
                else:
                    self.pending_excuse = (excuse_player, excuse_attack_side)
            elif self.pending_excuse is not None:
                _, pend_side = self.pending_excuse
                if self.is_attack_side(winner) == pend_side:
                    from_pile = self.taker_tricks if pend_side else self.defense_tricks
                    to_pile = self.defense_tricks if pend_side else self.taker_tricks
                    from_pile.append(EXCUSE)
                    low = _lowest_value_card([c for c in from_pile if not c.is_excuse()])
                    if low is not None:
                        from_pile.remove(low)
                        to_pile.append(low)
                    self.pending_excuse = None

            # Petit au Bout: if Petit is in the last trick (15th)
            if self.trick_count == 15:
                for _, c in trick_cards:
                    if c.is_petit():
                        self.petit_au_bout_taker_side = self.is_attack_side(winner)
                        break

            self.current_trick = []
            self.leader = winner

    def legal_cards(self, player: int) -> list[Card]:
        return legal_plays(self.hands[player], self.current_trick)


def run_deal_5p(
    deal: Deal5P,
    bidding: BiddingResult,
    partner: int | None,
    get_play: Callable[[SingleDealState5P, int], Card],
    rng: random.Random | None = None,
    get_poignee: Callable[[SingleDealState5P, int], tuple[int, int] | None] | None = None,
    get_chelem: Callable[[Deal5P, BiddingResult], int | None] | None = None,
) -> tuple[int, int, int, int, int]:
    """
    Run one 5-player deal after deal and bidding.
    Handles chien/écart, optional poignée and chelem, Petit au Bout, Excuse exchanges, ½-point rule.

    partner: index of taker's partner (2 vs 3), or None if taker plays alone (1 vs 4).
    """
    state = SingleDealState5P(deal, bidding, partner)

    # Prise/Garde: taker takes chien and discards 3
    if bidding.contract in (Contract.PRISE, Contract.GARDE):
        state.hands[state.taker].extend(state.chien)
        state.chien.clear()
        hand = state.hands[state.taker]
        discardable = [c for c in hand if not c.is_bout() and not (c.is_suit() and c.rank == 14)]
        if len(discardable) >= 3:
            for c in discardable[:3]:
                hand.remove(c)
        else:
            for _ in range(3):
                if hand:
                    hand.pop()

    # Poignée: before first card, each player may announce according to thresholds defined by callback.
    if get_poignee is not None:
        order = [state.taker, (state.taker + 1) % 5, (state.taker + 2) % 5, (state.taker + 3) % 5, (state.taker + 4) % 5]
        for player in order:
            if state.poignee_points > 0:
                break
            out = get_poignee(state, player)
            if out is not None:
                num_atouts, points = out
                state.poignee_points = points
                state.poignee_attack_side = state.is_attack_side(player)

    # Chelem: if announced, announcer leads
    if get_chelem is not None:
        announcer = get_chelem(deal, bidding)
        if announcer is not None:
            state.chelem_announcer = announcer
            state.leader = announcer

    # Play 15 tricks (5 players, 3x3 distribution + chien 3)
    for _ in range(15):
        for _ in range(5):
            player = state.current_player()
            legal = state.legal_cards(player)
            card = get_play(state, player)
            if card not in legal:
                raise ValueError(f"Illegal play {card}; legal {legal}")
            state.play_card(player, card)

    # If an Excuse exchange was pending and never resolved, give Excuse to that camp
    if state.pending_excuse is not None:
        _, pend_side = state.pending_excuse
        if pend_side:
            state.taker_tricks.append(EXCUSE)
        else:
            state.defense_tricks.append(EXCUSE)
        state.pending_excuse = None

    taker_final = list(state.taker_tricks)
    defense_final = list(state.defense_tricks)
    if state.contract == Contract.GARDE_SANS:
        taker_final = taker_final + state.chien
    elif state.contract == Contract.GARDE_CONTRE:
        defense_final = defense_final + state.chien

    # Chelem primes (15 tricks = 60 cards on attack/defense side; chien belongs to one side)
    n_attack_tricks = len(taker_final) // 5
    n_defense_tricks = len(defense_final) // 5
    if n_attack_tricks == 15:
        state.chelem_points = CHELEM_ANNOUNCED if state.chelem_announcer is not None else CHELEM_NOT_ANNOUNCED
    elif n_defense_tricks == 15:
        state.chelem_points = -CHELEM_DEFENSE
    elif state.chelem_announcer is not None:
        state.chelem_points = CHELEM_ANNOUNCED_FAILED

    taker_pts_half = points_in_cards(taker_final, use_half_points=True)
    num_bouts = count_bouts_in_cards(taker_final)
    base = deal_base_score_3p(taker_pts_half, num_bouts, state.contract)

    # Poignée benefit: side that announced gets the prime if that side wins, else loses it
    poignee_benefit_attack = None
    if state.poignee_points > 0 and state.poignee_attack_side is not None:
        poignee_benefit_attack = (state.poignee_attack_side and base > 0) or (
            not state.poignee_attack_side and base < 0
        )

    final_score = apply_primes(
        base,
        petit_au_bout_taker=state.petit_au_bout_taker_side,
        poignee_taker_side=poignee_benefit_attack,
        poignee_points=state.poignee_points or 0,
        chelem_points=state.chelem_points,
        contract=state.contract,
    )
    return mark_5p_with_taker(final_score, state.taker, state.partner)


def play_one_deal_5p(
    get_bid: Callable[[int, list[tuple[int, int | None]]], int | None],
    get_play: Callable[[SingleDealState5P, int], Card],
    get_partner: Callable[[Deal5P, BiddingResult], int | None] | None = None,
    dealer: int = 0,
    rng: random.Random | None = None,
    get_poignee: Callable[[SingleDealState5P, int], tuple[int, int] | None] | None = None,
    get_chelem: Callable[[Deal5P, BiddingResult], int | None] | None = None,
) -> tuple[tuple[int, int, int, int, int], Deal5P | None, BiddingResult | None, int | None]:
    """
    Deal, bid, play one 5-player deal.

    - get_bid: bidding callback, same contract values as 4p/3p.
    - get_partner(deal, bidding) -> partner index (2 vs 3) or None (taker alone, 1 vs 4).
    - get_poignee: optional Poignée callback (8/10/13 atouts logic is up to the caller).
    - get_chelem: optional Chelem announcer callback.

    Returns (scores5, deal, bidding, partner_index). If everyone passes, returns (0,0,0,0,0), deal, None, None.
    """
    if rng is None:
        rng = random.Random()
    deck = make_deck_78()
    deal = deal_5p(deck=deck, rng=rng)
    deal = Deal5P(hands=deal.hands, chien=deal.chien, dealer=dealer)
    bidding = run_bidding_5p(deal.dealer, get_bid)
    if bidding is None:
        return (0, 0, 0, 0, 0), deal, None, None

    partner: int | None = None
    if get_partner is not None:
        partner = get_partner(deal, bidding)

    scores = run_deal_5p(
        deal,
        bidding,
        partner,
        get_play,
        rng,
        get_poignee=get_poignee,
        get_chelem=get_chelem,
    )
    return scores, deal, bidding, partner


def run_match_5p(
    num_deals: int,
    get_bid: Callable[[int, list[tuple[int, int | None]]], int | None],
    get_play: Callable[[SingleDealState5P, int], Card],
    get_partner: Callable[[Deal5P, BiddingResult], int | None] | None = None,
    rng: random.Random | None = None,
    get_poignee: Callable[[SingleDealState5P, int], tuple[int, int] | None] | None = None,
    get_chelem: Callable[[Deal5P, BiddingResult], int | None] | None = None,
) -> tuple[tuple[int, int, int, int, int], list[tuple[int, int, int, int, int]]]:
    """
    Run a 5-player match. Dealer rotates 0->1->2->3->4->0. Redistributes if everyone passes.

    get_partner(deal, bidding) chooses partner per deal (2 vs 3) or None (1 vs 4).
    """
    if rng is None:
        rng = random.Random()
    totals = [0, 0, 0, 0, 0]
    per_deal: list[tuple[int, int, int, int, int]] = []
    dealer = 0
    played = 0
    while played < num_deals:
        scores, _deal, bidding, _partner = play_one_deal_5p(
            get_bid,
            get_play,
            get_partner=get_partner,
            dealer=dealer,
            rng=rng,
            get_poignee=get_poignee,
            get_chelem=get_chelem,
        )
        if bidding is None:
            dealer = next_dealer_5p(dealer)
            continue
        per_deal.append(scores)
        for i in range(5):
            totals[i] += scores[i]
        played += 1
        dealer = next_dealer_5p(dealer)
    return (totals[0], totals[1], totals[2], totals[3], totals[4]), per_deal
