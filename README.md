## Tarot Solver

Tarot Solver is a Python project that lets you **play, simulate and train agents** for French Tarot using the **official FFT rules** for 3, 4, and 5 players.

The focus is on:

- **Accurate rules** for deals, bidding, scoring and special cases (Excuse, Petit au Bout, Poignées, Chelem, etc.).
- **Experiments and AI**: reinforcement learning (RL), tournaments, and league-style training.
- **A future desktop GUI** to explore games and agents more easily.

The desktop GUI is **currently under construction**.  
Right now it is only a **placeholder** with non-functional tabs; all core functionality lives in the Python engine and CLI tools.

- **Rules reference:** `docs/rules.md` (extracted from `Règlement Tarot.pdf`).
- **High-level plan:** `GAMEPLAN.md`.

---

## Project layout (quick overview)

| Path | Description |
|------|-------------|
| `src/tarot/` | Core engine, environments, RL training loops, tournaments, genetic algorithm (GA), and league orchestration. |
| `src/tarot_gui/` | **Work-in-progress** desktop GUI (PySide6); currently a non-functional shell with placeholder tabs. |
| `docs/rules.md` | Detailed FFT rules checklist for 3, 4, and 5 players, scoring and match rules. |
| `tests/` | Unit tests for engine, environments, tournaments, GA, RL, league and CLI helpers. |

---

## Setup (development)

From the project root:

```bash
# (Recommended) Create and activate a virtualenv
python -m venv .venv
.\.venv\Scripts\activate  # Windows

# Install package + dev/RL/GUI extras
python -m pip install -e ".[dev,rl,gui]"
```

This installs the core engine plus extras needed for **development**, **RL experiments**, and the **work-in-progress GUI**.

---

## Basic usage (engine only)

### Example: play a single deal (4 players)

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

### Example: run a short match (N deals, rotating dealer)

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

### Optional features: Poignée and Chelem callbacks

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

## Engine features

- **4 players**
  - Deal (3×3, Chien 6), bidding (Prise, Garde, Garde sans, Garde contre), chien/écart, 18 tricks, FFT scoring.
  - Correct **Excuse** handling (return to owner’s camp, give low card to trick winner; pending if camp has no tricks yet).
  - **Matches** via `run_match_4p(N, ...)` — N deals with rotating dealer and total points per player.
- **3 players**
  - Deal (4×4, Chien 6), same bidding as 4 players, chien/écart, 24 tricks, ½‑point scoring.
  - Full Excuse handling, Petit au Bout, Poignée, Chelem, and match scoring via `run_match_3p`.
- **5 players**
  - Deal (3×3, Chien 3), same bidding as 4 players, chien/écart, 15 tricks, ½‑point scoring.
  - Full Excuse handling, Petit au Bout, Poignée, Chelem.
  - Supports **1 vs 4** (taker alone) and **2 vs 3** (taker + partner via callback), with FFT‑style scoring (defense as 4p; attackers 2/3–1/3 when there is a partner).
- **Extras**
  - **Poignée:** Optional callback before play (10/13/15 atouts → 20/30/40 pts to winning camp).
  - **Chelem:** Optional callback (announcer leads); +400/+200/−200 and defense slam.
  - **Petit au Bout:** Applied in scoring.

---

## League training, CLI, and GUI status

### CLI commands

After installation you get a `tarot` command‑line interface with:

- `tarot train-ppo-4p` – run custom PPO training for a single 4‑player learning seat.
- `tarot eval-4p` – evaluate a checkpoint vs random opponents.
- `tarot league-4p` – run a simple 4‑player league generation loop (tournaments + GA, PPO hooks ready).

You can also run leagues directly from Python:

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

This script ensures dev/RL dependencies are installed and then runs `pytest`.

### GUI (work in progress)

An **early, non-functional desktop GUI** is available for preview only:

```bash
python run.py
```

This will create `.venv` (if needed), install `tarot-solver` with dev/RL/GUI
extras, and start a PySide6 window with **placeholder** tabs:

- **Dashboard** – overview of league runs (placeholder buttons, not wired up yet).
- **League** – configuration and controls for league‑based training (placeholders).
- **Agents** – list and Hall of Fame area (placeholders).
- **Play** – custom tables, spectate, play‑vs‑AI (placeholders).
- **Settings** – global settings (placeholders).

The GUI is **under active construction** and is **not yet usable** for real games or training; use the Python API and CLI for now.

---

## License

Private / unlicensed unless stated otherwise.
