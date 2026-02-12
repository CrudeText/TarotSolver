# Tarot Solver

Tarot Solver is a Python project for **playing, simulating and training agents** in French Tarot using the **official FFT rules** for 3, 4, and 5 players.

**Focus:** Accurate rules, reinforcement learning, tournaments, league-style training, and a desktop GUI to manage experiments and play vs AI.

---

## Development status

| Component | Status |
|-----------|--------|
| **Game engine** | ✅ Complete — 3, 4, 5 players, full rules (Excuse, Petit au Bout, Poignée, Chelem) |
| **RL training** | ✅ Complete — Custom PPO, checkpointing, `tarot train-ppo-4p` |
| **Tournaments & league** | ✅ Complete — ELO, GA, `run_league_generation`, `tarot league-4p` |
| **Population persistence** | ✅ Complete — Import/export, augmentation helpers |
| **Desktop GUI** | 🚧 In progress — League tab functional (groups, population tools); other tabs placeholder |

**Planned:** Wire League tab to run training, add export, live metrics and charts; then Agents, Play vs AI, and spectate.

---

## Setup

From the project root:

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
python -m pip install -e ".[dev,rl,gui]"
```

**Quick start (GUI):**

```bash
python run.py
```

Creates `.venv` if needed, installs deps, and launches the PySide6 GUI. Skips install when the package is already present.

---

## Project layout

| Path | Description |
|------|-------------|
| `src/tarot/` | Engine, envs, RL, tournaments, GA, league orchestration, persistence |
| `src/tarot_gui/` | Desktop GUI (PySide6) — League tab with population management |
| `docs/` | Rules, implementation plans |
| `tests/` | Test suite |

---

## Usage

### Engine (Python)

See the API for `play_one_deal_4p`, `run_match_4p`, `run_league_generation`, etc. Rules reference: `docs/rules.md`.

### CLI

- `tarot train-ppo-4p` — PPO training
- `tarot eval-4p` — Evaluate checkpoint vs random
- `tarot league-4p` — League generation loop (tournaments + GA)

### GUI

- **League** — Build populations as groups (Add random, Import, Augment, Clear), expand groups to edit agents, configure league structure and GA (forms not yet wired).
- **Dashboard, Agents, Play, Settings** — Placeholders for later phases.

---

## Tests

```bash
python tests.py
```

---

## References

- **Rules:** `docs/rules.md` (from `Règlement Tarot.pdf`)
- **Plan:** `GAMEPLAN.md`
- **License:** MIT — see `LICENSE`
