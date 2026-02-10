# Tarot Solver

Tarot game engine (FFT official rules), RL training, and tournaments for 3, 4, and 5 players.

- **Rules:** `docs/rules.md` (extracted from `Règlement Tarot.pdf`).
- **Plan:** `GAMEPLAN.md`.

---

## Project layout

| Path | Description |
|------|-------------|
| `src/tarot/` | Engine, envs, RL training, tournaments, GA, league orchestration. |
| `src/tarot_gui/` | Placeholder desktop GUI (PySide6) with tabs for Dashboard, League, Agents, Play, Settings. |
| `docs/rules.md` | FFT rules checklist for 4p, 3p, 5p, scoring, match. |
| `tests/` | Unit tests for engine, envs, tournaments, GA, RL, league, and CLI helpers. |

---

## Setup

```bash
cd "Tarot Solver"

# (Recommended) Create and activate a virtualenv
python -m venv .venv
.\.venv\Scripts\activate  # Windows

# Install package + dev/RL/GUI extras
python -m pip install -e ".[dev,rl,gui]"
```

---

## Usage

### One deal (4 players)

```python
from tarot import play_one_deal_4p, SingleDealState, Contract
import random

rng = random.Random(42)

def get_bid(player, history):
    return Contract.PRISE if player == 1 else None

def get_play(state: SingleDealState, player: int):
    return rng.choice(state.legal_cards(player))

scores, deal, bidding = play_one_deal_4p(get_bid, get_play, rng=rng)
# scores: (s0, s1, s2, s3), sum = 0; bidding is None if everyone passed or Petit sec
```

### Match (N deals, rotating dealer)

```python
from tarot import run_match_4p, Contract
import random

def get_bid(player, history):
    return Contract.PRISE if player == 1 else None
def get_play(state, player):
    return random.choice(state.legal_cards(player))

totals, per_deal_scores = run_match_4p(5, get_bid, get_play, rng=random.Random(99))
# totals: (t0, t1, t2, t3); per_deal_scores: list of (s0,s1,s2,s3) per played deal
```

### Poignée and Chelem (optional callbacks)

```python
# get_poignee(state, player) -> None or (num_atouts, points) e.g. (10, 20), (13, 30), (15, 40)
def get_poignee(state, player):
    n = len([c for c in state.hands[player] if c.is_trump()])
    if any(c.is_excuse() for c in state.hands[player]):
        n += 1
    if n >= 10:
        return (10, 20)  # simple poignée
    return None

# get_chelem(deal, bidding) -> None or player_index (that player leads first)
def get_chelem(deal, bidding):
    return None  # or e.g. bidding.taker to announce

scores, _, _ = play_one_deal_4p(
    get_bid, get_play, rng=rng,
    get_poignee=get_poignee,
    get_chelem=get_chelem,
)
```

---

## Engine features (Phase 1)

- **4 players:** Deal (3×3, Chien 6), bidding (Prise, Garde, Garde sans, Garde contre), chien/écart, 18 tricks, FFT scoring.
- **Excuse:** Correct handling (return to owner’s camp, give low card to trick winner; pending if camp has no tricks yet).
- **Match:** `run_match_4p(N, ...)` — N deals, rotating dealer, total points per player.
- **Poignée:** Optional callback before play (10/13/15 atouts → 20/30/40 pts to winning camp).
- **Chelem:** Optional callback (announcer leads); +400/+200/−200 and defense slam.
- **Petit au Bout:** Applied in scoring.
- **3 players:** Deal (4×4, Chien 6), bidding as in 4p, chien/écart, 24 tricks, ½-point scoring, full Excuse handling, Petit au Bout, Poignée, Chelem, and match scoring via `run_match_3p`.
-- **5 players:** Deal (3×3, Chien 3), bidding as in 4p, chien/écart, 15 tricks, ½-point scoring, full Excuse handling, Petit au Bout, Poignée, Chelem. Supports **1 vs 4** (taker alone) and **2 vs 3** (taker + partner via callback), with FFT-style scoring (defense as 4p; attackers 2/3–1/3 when there is a partner).

---

## League training, CLI, GUI

### CLI commands

After installation you get a `tarot` CLI with:

- `tarot train-ppo-4p` – run custom PPO training for a single 4p learning seat.
- `tarot eval-4p` – evaluate a checkpoint vs random opponents.
- `tarot league-4p` – run a simple 4p league generation loop (tournaments + GA, PPO hooks ready).

There is also a convenience league runner in Python:

```python
from tarot.league import LeagueConfig, run_league_generation
from tarot.tournament import Population, Agent
import random

pop = Population()
for i in range(8):
    pop.add(Agent(id=f"A{i}", name=f"A{i}", player_counts=[4]))

cfg = LeagueConfig(player_count=4, deals_per_match=3, rounds_per_generation=2)
new_pop, summary = run_league_generation(pop, cfg, rng=random.Random(0))
print(summary)
```

### Tests

You can run the full test suite with:

```bash
python tests.py
```

This script ensures dev/RL deps are installed and then runs `pytest`.

### GUI

An early, non-functional desktop GUI is available:

```bash
python run.py
```

This will create `.venv` (if needed), install `tarot-solver` with dev/RL/GUI
extras, and start a PySide6 window with placeholder tabs:

- **Dashboard** – overview of league runs (placeholder buttons).
- **League** – configuration and controls for league-based training (placeholders).
- **Agents** – list + Hall of Fame area (placeholders).
- **Play** – custom tables, spectate, play-vs-AI (placeholders).
- **Settings** – global settings (placeholders).

---

## License

Private / unlicensed unless stated otherwise.
