"""
Environment wrapper around the 4-player tarot engine for RL.

Design (first version):
- Single-agent view: one learning seat (e.g. player 0) per env instance.
- Episode = one match of N deals. Reward is given only at the end of the match
  and equals the total match score of the learning seat.
- At each step, the env exposes a decision point for the learning seat:
  - Bidding phase at the start of each deal.
  - Play phase whenever it's that player's turn to play a card.
- Other seats use a simple random policy for now; the API is structured so that
  we can later plug arbitrary policies/models per seat.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from .bidding import Contract, run_bidding_4p, run_bidding_3p, run_bidding_5p
from .deal import (
    Deal4P,
    Deal3P,
    Deal5P,
    deal_4p,
    deal_3p,
    deal_5p,
    petit_sec_4p,
    next_dealer_4p,
    next_dealer_3p,
    next_dealer_5p,
)
from .deck import Card, make_deck_78
from .env import (
    NUM_ACTIONS,
    NUM_BID_ACTIONS,
    NUM_CARD_ACTIONS,
    card_index,
    encode_bidding_observation_4p,
    encode_bidding_observation_3p,
    encode_bidding_observation_5p,
    encode_play_observation_4p,
    encode_play_observation_3p,
    encode_play_observation_5p,
    legal_action_mask_bidding,
    legal_action_mask_play_from_hand_and_legal_cards,
)
from .game import SingleDealState, SingleDealState3P, SingleDealState5P
from .play import legal_plays
from .scoring import (
    points_in_cards,
    deal_base_score,
    deal_base_score_3p,
    apply_primes,
    mark_4p_with_taker,
    mark_3p_with_taker,
    mark_5p_with_taker,
)
from .play import count_bouts_in_cards
from .deck import EXCUSE


@dataclass
class StepResult:
    """Container returned by TarotEnv4P.step/reset for clarity."""

    obs: List[float]
    reward: float
    done: bool
    info: dict
    legal_actions_mask: List[bool]


class TarotEnv4P:
    """
    4-player Tarot environment (single learning seat, full match episodes).

    Public API (minimal, Gym-like but without external dependency):
      - reset() -> StepResult          # start new match, first decision for learning seat
      - step(action: int) -> StepResult
    """

    def __init__(
        self,
        num_deals: int = 5,
        learning_player: int = 0,
        rng: Optional[random.Random] = None,
    ) -> None:
        assert 0 <= learning_player < 4
        self.num_deals = num_deals
        self.learning_player = learning_player
        self.rng = rng or random.Random()

        # Match state
        self._dealer: int = 0
        self._deal_index: int = 0
        self._totals: List[int] = [0, 0, 0, 0]

        # Current deal / play state
        self._deal: Optional[Deal4P] = None
        self._state: Optional[SingleDealState] = None
        self._bidding_result = None  # type: ignore[assignment]
        self._phase: str = "idle"  # "bidding", "play", "done"

    # ---- Public API ----

    def reset(self) -> StepResult:
        """Start a new match and return the first decision for the learning seat."""
        self._dealer = 0
        self._deal_index = 0
        self._totals = [0, 0, 0, 0]
        self._phase = "idle"
        return self._start_next_deal_or_finish_match()

    def step(self, action: int) -> StepResult:
        """
        Apply an action for the learning seat at the current decision point.

        - In bidding phase: `action` must be in [0, NUM_BID_ACTIONS).
        - In play phase: `action` must be a card action index in the global space.
        """
        if self._phase == "bidding":
            return self._step_bidding(action)
        if self._phase == "play":
            return self._step_play(action)
        # If phase is "done", any further step just re-emits terminal state
        return StepResult(
            obs=[],
            reward=0.0,
            done=True,
            info={"phase": "done"},
            legal_actions_mask=[False] * NUM_ACTIONS,
        )

    # ---- Internal helpers ----

    def _start_next_deal_or_finish_match(self) -> StepResult:
        """Either start the next deal (bidding) or finish the match."""
        # If we've already played num_deals deals, terminate match
        if self._deal_index >= self.num_deals:
            self._phase = "done"
            reward = float(self._totals[self.learning_player])
            return StepResult(
                obs=[],
                reward=reward,
                done=True,
                info={
                    "phase": "done",
                    "totals": tuple(self._totals),
                    "deals_played": self._deal_index,
                },
                legal_actions_mask=[False] * NUM_ACTIONS,
            )

        # Create a fresh 4p deal without Petit sec for any player
        while True:
            deck = make_deck_78()
            self.rng.shuffle(deck)
            deal = deal_4p(deck=deck, rng=self.rng)
            if not any(petit_sec_4p(hand) for hand in deal.hands):
                break
        # Attach correct dealer
        self._deal = Deal4P(hands=deal.hands, chien=deal.chien, dealer=self._dealer)
        self._state = None
        self._bidding_result = None
        self._phase = "bidding"

        # First decision for learning player is its bid; other bids will be sampled inside _step_bidding
        obs = encode_bidding_observation_4p(
            hand=self._deal.hands[self.learning_player],
            history=[],  # for now, learning seat does not see others' bids before choosing
            player_index=self.learning_player,
        )
        mask = legal_action_mask_bidding(history=[])
        return StepResult(
            obs=obs,
            reward=0.0,
            done=False,
            info={
                "phase": "bidding",
                "deal_index": self._deal_index,
                "dealer": self._dealer,
            },
            legal_actions_mask=mask,
        )

    def _step_bidding(self, action: int) -> StepResult:
        assert self._deal is not None
        # Decode learning player's chosen bid
        if not (0 <= action < NUM_BID_ACTIONS):
            raise ValueError(f"Invalid bidding action {action}")

        def learning_bid_from_action(a: int) -> Optional[int]:
            if a == 0:
                return None  # PASS
            # 1..4 map directly to Contract enum value
            return a

        chosen_bid_value: Optional[int] = learning_bid_from_action(action)

        # Implement bidding round using run_bidding_4p, plugging learning player's choice
        def get_bid(player: int, history: List[Tuple[int, int | None]]) -> Optional[int]:
            # Learning seat: return fixed chosen bid regardless of history
            if player == self.learning_player:
                return chosen_bid_value
            # Opponents: simple random policy among PASS or a random contract
            # For now, we don't enforce monotonicity; run_bidding_4p will just pick the max.
            options: List[Optional[int]] = [None, int(Contract.PRISE), int(Contract.GARDE)]
            # Occasionally allow higher contracts
            if self.rng.random() < 0.3:
                options.extend([int(Contract.GARDE_SANS), int(Contract.GARDE_CONTRE)])
            return self.rng.choice(options)

        self._bidding_result = run_bidding_4p(self._deal.dealer, get_bid)
        if self._bidding_result is None:
            # Everyone passed or effectively no taker: no score change, move to next deal
            self._advance_after_deal_zero_scores()
            return self._start_next_deal_or_finish_match()

        # There is a taker and contract: initialise play state and advance until it's learning player's turn
        self._state = SingleDealState(self._deal, self._bidding_result)
        self._phase = "play"
        return self._advance_play_until_learning_turn_or_deal_end()

    def _random_legal_card(self, hand: Sequence[Card], current_trick: Sequence[Tuple[int, Card]]) -> Card:
        legal = legal_plays(list(hand), list(current_trick))
        if not legal:
            raise RuntimeError("No legal plays available")
        return self.rng.choice(legal)

    def _advance_play_until_learning_turn_or_deal_end(self) -> StepResult:
        """Simulate other players until learning seat must act, or the deal ends."""
        assert self._state is not None
        state = self._state

        # 18 tricks in 4p; deal ends when all hands are empty
        while any(state.hands[p] for p in range(4)):
            current_player = state.current_player()
            if current_player == self.learning_player:
                # Learning seat must choose a card now: emit observation and legal mask
                legal_cards = state.legal_cards(self.learning_player)
                obs = encode_play_observation_4p(state, player_index=self.learning_player)
                mask = legal_action_mask_play_from_hand_and_legal_cards(
                    state.hands[self.learning_player],
                    legal_cards,
                )
                return StepResult(
                    obs=obs,
                    reward=0.0,
                    done=False,
                    info={
                        "phase": "play",
                        "deal_index": self._deal_index,
                        "dealer": self._dealer,
                        "current_trick_len": len(state.current_trick),
                    },
                    legal_actions_mask=mask,
                )

            # Opponent plays randomly among legal cards
            legal_cards = state.legal_cards(current_player)
            card = self._random_legal_card(state.hands[current_player], state.current_trick)
            if card not in legal_cards:
                raise RuntimeError("Random policy chose illegal card (internal bug)")
            state.play_card(current_player, card)

        # Deal is over: finalise scoring and move to next deal / match end
        self._finalise_scoring_for_current_deal()
        return self._start_next_deal_or_finish_match()

    def _step_play(self, action: int) -> StepResult:
        assert self._state is not None
        state = self._state
        # Decode card action index
        if not (NUM_BID_ACTIONS <= action < NUM_ACTIONS):
            raise ValueError(f"Invalid play action {action}")
        card_idx = action - NUM_BID_ACTIONS
        if not (0 <= card_idx < NUM_CARD_ACTIONS):
            raise ValueError(f"Invalid card index {card_idx}")

        # Find corresponding Card object in hand
        hand = state.hands[self.learning_player]
        chosen_card: Optional[Card] = None
        for c in hand:
            if card_index(c) == card_idx:
                chosen_card = c
                break
        if chosen_card is None:
            raise ValueError("Chosen card index not found in hand")

        legal = state.legal_cards(self.learning_player)
        if chosen_card not in legal:
            raise ValueError("Chosen card is not a legal move")

        state.play_card(self.learning_player, chosen_card)
        # Continue play until learning seat's next turn or deal end
        return self._advance_play_until_learning_turn_or_deal_end()

    def _finalise_scoring_for_current_deal(self) -> None:
        """Compute per-player scores for the deal and update match totals."""
        assert self._state is not None and self._bidding_result is not None and self._deal is not None
        state = self._state
        bidding = self._bidding_result

        # Handle pending Excuse if any (same logic as in run_deal_4p)
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

        taker_pts = points_in_cards(taker_final)
        num_bouts = count_bouts_in_cards(taker_final)
        base = deal_base_score(taker_pts, num_bouts, state.contract)

        # For now, we ignore Poignée and Chelem in the env scoring (they are optional extras).
        final_score = apply_primes(
            base,
            petit_au_bout_taker=state.petit_au_bout_taker,
            poignee_taker_side=None,
            poignee_points=0,
            chelem_points=0,
            contract=state.contract,
        )
        scores = mark_4p_with_taker(final_score, state.taker)
        for i in range(4):
            self._totals[i] += scores[i]

        self._advance_after_deal_zero_scores()

    def _advance_after_deal_zero_scores(self) -> None:
        """Advance dealer/deal counters after a deal (even if it scored 0)."""
        self._deal_index += 1
        self._dealer = next_dealer_4p(self._dealer)
        self._deal = None
        self._state = None
        self._bidding_result = None
        self._phase = "idle"


__all__ = ["TarotEnv4P", "TarotEnv3P", "TarotEnv5P", "StepResult"]


class TarotEnv3P:
    """
    3-player Tarot environment (single learning seat, full match episodes).

    Same interface as TarotEnv4P, but using 3-player rules and scoring.
    """

    def __init__(
        self,
        num_deals: int = 5,
        learning_player: int = 0,
        rng: Optional[random.Random] = None,
    ) -> None:
        assert 0 <= learning_player < 3
        self.num_deals = num_deals
        self.learning_player = learning_player
        self.rng = rng or random.Random()

        self._dealer: int = 0
        self._deal_index: int = 0
        self._totals: List[int] = [0, 0, 0]

        self._deal: Optional[Deal3P] = None
        self._state: Optional[SingleDealState3P] = None
        self._bidding_result = None  # type: ignore[assignment]
        self._phase: str = "idle"

    def reset(self) -> StepResult:
        self._dealer = 0
        self._deal_index = 0
        self._totals = [0, 0, 0]
        self._phase = "idle"
        return self._start_next_deal_or_finish_match()

    def step(self, action: int) -> StepResult:
        if self._phase == "bidding":
            return self._step_bidding(action)
        if self._phase == "play":
            return self._step_play(action)
        return StepResult(
            obs=[],
            reward=0.0,
            done=True,
            info={"phase": "done"},
            legal_actions_mask=[False] * NUM_ACTIONS,
        )

    def _start_next_deal_or_finish_match(self) -> StepResult:
        if self._deal_index >= self.num_deals:
            self._phase = "done"
            reward = float(self._totals[self.learning_player])
            return StepResult(
                obs=[],
                reward=reward,
                done=True,
                info={"phase": "done", "totals": tuple(self._totals), "deals_played": self._deal_index},
                legal_actions_mask=[False] * NUM_ACTIONS,
            )

        deck = make_deck_78()
        self.rng.shuffle(deck)
        deal = deal_3p(deck=deck, rng=self.rng)
        self._deal = Deal3P(hands=deal.hands, chien=deal.chien, dealer=self._dealer)
        self._state = None
        self._bidding_result = None
        self._phase = "bidding"

        obs = encode_bidding_observation_3p(
            hand=self._deal.hands[self.learning_player],
            history=[],
            player_index=self.learning_player,
        )
        mask = legal_action_mask_bidding(history=[])
        return StepResult(
            obs=obs,
            reward=0.0,
            done=False,
            info={"phase": "bidding", "deal_index": self._deal_index, "dealer": self._dealer},
            legal_actions_mask=mask,
        )

    def _step_bidding(self, action: int) -> StepResult:
        assert self._deal is not None
        if not (0 <= action < NUM_BID_ACTIONS):
            raise ValueError(f"Invalid bidding action {action}")

        def learning_bid_from_action(a: int) -> Optional[int]:
            if a == 0:
                return None
            return a

        chosen_bid_value: Optional[int] = learning_bid_from_action(action)

        def get_bid(player: int, history: List[Tuple[int, int | None]]) -> Optional[int]:
            if player == self.learning_player:
                return chosen_bid_value
            options: List[Optional[int]] = [None, int(Contract.PRISE), int(Contract.GARDE)]
            if self.rng.random() < 0.3:
                options.extend([int(Contract.GARDE_SANS), int(Contract.GARDE_CONTRE)])
            return self.rng.choice(options)

        self._bidding_result = run_bidding_3p(self._deal.dealer, get_bid)
        if self._bidding_result is None:
            self._advance_after_deal_zero_scores()
            return self._start_next_deal_or_finish_match()

        self._state = SingleDealState3P(self._deal, self._bidding_result)
        self._phase = "play"
        return self._advance_play_until_learning_turn_or_deal_end()

    def _random_legal_card(self, hand: Sequence[Card], current_trick: Sequence[Tuple[int, Card]]) -> Card:
        legal = legal_plays(list(hand), list(current_trick))
        if not legal:
            raise RuntimeError("No legal plays available")
        return self.rng.choice(legal)

    def _advance_play_until_learning_turn_or_deal_end(self) -> StepResult:
        assert self._state is not None
        state = self._state

        while any(state.hands[p] for p in range(3)):
            current_player = state.current_player()
            if current_player == self.learning_player:
                legal_cards = state.legal_cards(self.learning_player)
                obs = encode_play_observation_3p(state, player_index=self.learning_player)
                mask = legal_action_mask_play_from_hand_and_legal_cards(
                    state.hands[self.learning_player],
                    legal_cards,
                )
                return StepResult(
                    obs=obs,
                    reward=0.0,
                    done=False,
                    info={
                        "phase": "play",
                        "deal_index": self._deal_index,
                        "dealer": self._dealer,
                        "current_trick_len": len(state.current_trick),
                    },
                    legal_actions_mask=mask,
                )

            legal_cards = state.legal_cards(current_player)
            card = self._random_legal_card(state.hands[current_player], state.current_trick)
            if card not in legal_cards:
                raise RuntimeError("Random policy chose illegal card (internal bug)")
            state.play_card(current_player, card)

        self._finalise_scoring_for_current_deal()
        return self._start_next_deal_or_finish_match()

    def _step_play(self, action: int) -> StepResult:
        assert self._state is not None
        state = self._state
        if not (NUM_BID_ACTIONS <= action < NUM_ACTIONS):
            raise ValueError(f"Invalid play action {action}")
        card_idx = action - NUM_BID_ACTIONS
        if not (0 <= card_idx < NUM_CARD_ACTIONS):
            raise ValueError(f"Invalid card index {card_idx}")

        hand = state.hands[self.learning_player]
        chosen_card: Optional[Card] = None
        for c in hand:
            if card_index(c) == card_idx:
                chosen_card = c
                break
        if chosen_card is None:
            raise ValueError("Chosen card index not found in hand")

        legal = state.legal_cards(self.learning_player)
        if chosen_card not in legal:
            raise ValueError("Chosen card is not a legal move")

        state.play_card(self.learning_player, chosen_card)
        return self._advance_play_until_learning_turn_or_deal_end()

    def _finalise_scoring_for_current_deal(self) -> None:
        assert self._state is not None and self._bidding_result is not None and self._deal is not None
        state = self._state

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

        # Ignore Poignée and Chelem primes in env scoring for now (as in 4p env).
        final_score = base
        scores = mark_3p_with_taker(final_score, state.taker)
        for i in range(3):
            self._totals[i] += scores[i]

        self._advance_after_deal_zero_scores()

    def _advance_after_deal_zero_scores(self) -> None:
        self._deal_index += 1
        self._dealer = next_dealer_3p(self._dealer)
        self._deal = None
        self._state = None
        self._bidding_result = None
        self._phase = "idle"


class TarotEnv5P:
    """
    5-player Tarot environment (single learning seat, full match episodes).

    For now, the taker always plays alone (no explicit partner logic yet).
    """

    def __init__(
        self,
        num_deals: int = 5,
        learning_player: int = 0,
        rng: Optional[random.Random] = None,
    ) -> None:
        assert 0 <= learning_player < 5
        self.num_deals = num_deals
        self.learning_player = learning_player
        self.rng = rng or random.Random()

        self._dealer: int = 0
        self._deal_index: int = 0
        self._totals: List[int] = [0, 0, 0, 0, 0]

        self._deal: Optional[Deal5P] = None
        self._state: Optional[SingleDealState5P] = None
        self._bidding_result = None  # type: ignore[assignment]
        self._phase: str = "idle"

    def reset(self) -> StepResult:
        self._dealer = 0
        self._deal_index = 0
        self._totals = [0, 0, 0, 0, 0]
        self._phase = "idle"
        return self._start_next_deal_or_finish_match()

    def step(self, action: int) -> StepResult:
        if self._phase == "bidding":
            return self._step_bidding(action)
        if self._phase == "play":
            return self._step_play(action)
        return StepResult(
            obs=[],
            reward=0.0,
            done=True,
            info={"phase": "done"},
            legal_actions_mask=[False] * NUM_ACTIONS,
        )

    def _start_next_deal_or_finish_match(self) -> StepResult:
        if self._deal_index >= self.num_deals:
            self._phase = "done"
            reward = float(self._totals[self.learning_player])
            return StepResult(
                obs=[],
                reward=reward,
                done=True,
                info={"phase": "done", "totals": tuple(self._totals), "deals_played": self._deal_index},
                legal_actions_mask=[False] * NUM_ACTIONS,
            )

        deck = make_deck_78()
        self.rng.shuffle(deck)
        deal = deal_5p(deck=deck, rng=self.rng)
        self._deal = Deal5P(hands=deal.hands, chien=deal.chien, dealer=self._dealer)
        self._state = None
        self._bidding_result = None
        self._phase = "bidding"

        obs = encode_bidding_observation_5p(
            hand=self._deal.hands[self.learning_player],
            history=[],
            player_index=self.learning_player,
        )
        mask = legal_action_mask_bidding(history=[])
        return StepResult(
            obs=obs,
            reward=0.0,
            done=False,
            info={"phase": "bidding", "deal_index": self._deal_index, "dealer": self._dealer},
            legal_actions_mask=mask,
        )

    def _step_bidding(self, action: int) -> StepResult:
        assert self._deal is not None
        if not (0 <= action < NUM_BID_ACTIONS):
            raise ValueError(f"Invalid bidding action {action}")

        def learning_bid_from_action(a: int) -> Optional[int]:
            if a == 0:
                return None
            return a

        chosen_bid_value: Optional[int] = learning_bid_from_action(action)

        def get_bid(player: int, history: List[Tuple[int, int | None]]) -> Optional[int]:
            if player == self.learning_player:
                return chosen_bid_value
            options: List[Optional[int]] = [None, int(Contract.PRISE), int(Contract.GARDE)]
            if self.rng.random() < 0.3:
                options.extend([int(Contract.GARDE_SANS), int(Contract.GARDE_CONTRE)])
            return self.rng.choice(options)

        self._bidding_result = run_bidding_5p(self._deal.dealer, get_bid)
        if self._bidding_result is None:
            self._advance_after_deal_zero_scores()
            return self._start_next_deal_or_finish_match()

        # For now, taker plays alone (partner=None)
        partner = None
        self._state = SingleDealState5P(self._deal, self._bidding_result, partner)
        self._phase = "play"
        return self._advance_play_until_learning_turn_or_deal_end()

    def _random_legal_card(self, hand: Sequence[Card], current_trick: Sequence[Tuple[int, Card]]) -> Card:
        legal = legal_plays(list(hand), list(current_trick))
        if not legal:
            raise RuntimeError("No legal plays available")
        return self.rng.choice(legal)

    def _advance_play_until_learning_turn_or_deal_end(self) -> StepResult:
        assert self._state is not None
        state = self._state

        while any(state.hands[p] for p in range(5)):
            current_player = state.current_player()
            if current_player == self.learning_player:
                legal_cards = state.legal_cards(self.learning_player)
                obs = encode_play_observation_5p(state, player_index=self.learning_player)
                mask = legal_action_mask_play_from_hand_and_legal_cards(
                    state.hands[self.learning_player],
                    legal_cards,
                )
                return StepResult(
                    obs=obs,
                    reward=0.0,
                    done=False,
                    info={
                        "phase": "play",
                        "deal_index": self._deal_index,
                        "dealer": self._dealer,
                        "current_trick_len": len(state.current_trick),
                    },
                    legal_actions_mask=mask,
                )

            legal_cards = state.legal_cards(current_player)
            card = self._random_legal_card(state.hands[current_player], state.current_trick)
            if card not in legal_cards:
                raise RuntimeError("Random policy chose illegal card (internal bug)")
            state.play_card(current_player, card)

        self._finalise_scoring_for_current_deal()
        return self._start_next_deal_or_finish_match()

    def _step_play(self, action: int) -> StepResult:
        assert self._state is not None
        state = self._state
        if not (NUM_BID_ACTIONS <= action < NUM_ACTIONS):
            raise ValueError(f"Invalid play action {action}")
        card_idx = action - NUM_BID_ACTIONS
        if not (0 <= card_idx < NUM_CARD_ACTIONS):
            raise ValueError(f"Invalid card index {card_idx}")

        hand = state.hands[self.learning_player]
        chosen_card: Optional[Card] = None
        for c in hand:
            if card_index(c) == card_idx:
                chosen_card = c
                break
        if chosen_card is None:
            raise ValueError("Chosen card index not found in hand")

        legal = state.legal_cards(self.learning_player)
        if chosen_card not in legal:
            raise ValueError("Chosen card is not a legal move")

        state.play_card(self.learning_player, chosen_card)
        return self._advance_play_until_learning_turn_or_deal_end()

    def _finalise_scoring_for_current_deal(self) -> None:
        assert self._state is not None and self._bidding_result is not None and self._deal is not None
        state = self._state

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

        # Chelem primes (copied from run_deal_5p)
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

        scores = mark_5p_with_taker(final_score, state.taker, state.partner)
        for i in range(5):
            self._totals[i] += scores[i]

        self._advance_after_deal_zero_scores()

    def _advance_after_deal_zero_scores(self) -> None:
        self._deal_index += 1
        self._dealer = next_dealer_5p(self._dealer)
        self._deal = None
        self._state = None
        self._bidding_result = None
        self._phase = "idle"

