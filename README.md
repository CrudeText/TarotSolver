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
| **Desktop GUI** | 🚧 In progress — League tab (groups, config, export); Dashboard run controls (Start/Pause/Cancel, LeagueRunWorker); Agents, Play, Settings placeholder |

**Planned:** Dashboard live metrics and charts, export during run; then Agents, Play vs AI, and spectate.

---

## Classic workflow

1. **Build a population** — Add groups of agents (random, imported, or mutated from selection). Optionally mark some as reference agents (no GA reproduction).
2. **Configure league** — Set player count, deals per match, league style (ELO or bracket), GA parameters.
3. **Run** — Start the league; tournaments run, ELO updates, GA evolves the population. Pause or let it complete.
4. **Export** — Save the population (full or filtered by ELO) for later runs.
5. **Play & explore** — Browse agents, promote best to Hall of Fame, spectate games, or play vs AI.

---

## High-level features (built or planned)

| Feature | Description |
|---------|-------------|
| **Agent navigation** | Browse agents by generation, traits, ELO; view lineage and training history. |
| **Hall of Fame** | Best-ever agents as benchmarks; use them as opponents or in custom tables. |
| **ELO rating** | Per-player-count and global ELO; margin-aware updates after each match. |
| **Tournament running** | ELO-based (round-robin, stratified, Swiss) or bracket-style; parallel tables. |
| **League training** | Tournament rounds + optional PPO fine-tuning + GA evolution, generation by generation. |
| **Play vs AI** | Pick your seat and opponents; play a full game against trained agents. |
| **Spectate** | Watch games unfold; optional policy overlays (top-k actions). |
| **Export / Import** | Save and load populations (full or filtered); merge or replace. |
| **Personality traits** | Descriptive metadata per agent (bidding, play style, context); used for filtering and GA; conditioning optional later. |

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

- **League** — Build populations as groups (Add random, Import, Augment, Clear), expand groups to edit agents, configure league structure and GA. Export now saves full population.
- **Dashboard** — Run controls (Start, Pause at next generation, Cancel); ELO min/mean/max; project required to start. Charts and ranking table planned.
- **Agents, Play, Settings** — Placeholders for later phases.

#### League tab (screenshots)

Top of the League Parameters tab:

![League Parameters tab — top](images/League%20Parameters%20Example.png)

Bottom of the League Parameters tab:

![League Parameters tab — bottom](images/Population%20Generator%20Example.png)

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
