# League Tab — Implementation Plan

Step-by-step plan to implement the League tab as described in GAMEPLAN §8.1. Tasks are ordered by dependency; later steps assume earlier ones are done.

---

## Phase 0 — Backend foundations

### Step 0.1: Add `can_use_as_ga_parent` to Agent

- Add `can_use_as_ga_parent: bool = True` to `Agent` in `tournament.py`.
- Update `ga.py` so `next_generation` selects parents only from agents with `can_use_as_ga_parent=True`.
- Add tests to ensure non-parent agents participate in matches but not in selection.

### Step 0.2: Population serialization (import/export)

- Add `population_to_dict(pop: Population) -> dict` and `population_from_dict(d: dict) -> Population` in a new module (e.g. `tarot/persistence.py`) or in `tournament.py`.
- Serialize: agent id, name, player_counts, ELOs, generation, traits, checkpoint_path, arch_name, parents, matches_played, total_match_score, **can_use_as_ga_parent**.
- Include metadata: export timestamp, league config snapshot (optional).
- Decide file format (JSON) and schema version for future compatibility.
- Add tests for round-trip (export → import → same population).

### Step 0.3: Population augmentation helpers

- Add `generate_random_agents(n: int, player_counts: List[int], rng, trait_bounds?) -> List[Agent]` — create N agents with random traits, no checkpoint.
- Add `mutate_from_base(base_agents: List[Agent], n: int, mutation_prob, mutation_std, rng) -> List[Agent]` — create N mutated children from base.
- Add `clone_agents(agents: List[Agent], n_per_agent: int) -> List[Agent]` — clone agents (same traits, new ids); no checkpoint copy unless needed.
- Integrate with existing `mutate_agent` / GA logic where possible.

### Step 0.4: League run with pause/cancel (backend)

- Refactor `run_league_generation` or add a generator/iterator that yields after each generation with current population + summary.
- Add a way to signal “pause at next generation” and “cancel” (e.g. threading.Event or callback).
- Ensure metrics (elo_min, elo_mean, elo_max, ranking) are available per generation for live updates.

---

## Phase 1 — League tab layout and population table

### Step 1.1: Tab structure

- Replace the League placeholder with a structured layout:
  - **Top:** Population management (table + tools).
  - **Middle:** Collapsible groups for League structure, GA parameters.
  - **Bottom:** Run controls + live metrics area.
  - **Separate section or dialog:** Export options.
- Use `QScrollArea` if needed; keep sections in logical order.

### Step 1.2: Population table (basic)

- Add `QTableWidget` (or `QTableView` + model) with columns: **GA parent** (checkbox), **Name**, **ID**, **Generation**, **ELO global**, **Checkpoint**, **Actions** (Delete button).
- Use delegate or custom widget for checkbox and delete button.
- Connect delete to remove selected row(s) from the in-memory population.

### Step 1.3: Population model / state

- Create a `LeagueTabState` or similar to hold:
  - Current `Population`.
  - Whether a run is active, paused, or idle.
  - Last run summary (elos, etc.).
- Sync table ↔ Population: table reflects state; edits update state.

---

## Phase 2 — Population management actions

**Layout:** All population tools (Add random, Import, Augment, Clear) go in the tools panel to the **right of the table**, same line level.

### Step 2.1: Add random agents

- Add “Add random” button and optional “Count” spinbox.
- On click: call `generate_random_agents`, add to population, refresh table.
- Validate population size constraints if any.

### Step 2.2: Import population

- Add “Import” button → file dialog (.json).
- On select: call `population_from_dict`, merge or replace (user choice via dialog).
- Refresh table and state.

### Step 2.3: Augment from selection

- Add “Augment from selection” flow:
  - User selects rows (or agents) in table.
  - Dialog: target size, mutation params (prob, std), clone count.
  - Call `mutate_from_base` and/or `clone_agents`, add to population.
- If selection empty, show message or disable button.

### Step 2.4: Clear / New population

- Add “New” or “Clear” button to start from empty population.
- Confirm if population is non-empty.

### Step 2.5: GA-parent checkbox behaviour

- Ensure checkbox column is editable and updates `Agent.can_use_as_ga_parent`.
- Persist when exporting; restore when importing.

---

## Phase 3 — League structure and GA parameters

### Step 3.1: League structure form

- Add form widgets for:
  - Player count (3 / 4 / 5).
  - Deals per match.
  - Matches per generation.
  - League style: combobox (ELO round-robin, ELO stratified, Swiss, Bracket single-elim, Bracket double-elim, etc.).
- Map to `LeagueConfig` and tournament logic. If bracket matchmaking is not yet in backend, add placeholder and implement in a later step.

### Step 3.2: GA parameters form

- Add form for:
  - Elite fraction, mutation prob, mutation std.
  - Fitness weights (global ELO, per-count, avg score).
  - Number of generations.
  - Cloning options (elite count, clones from base).
- Map to `LeagueConfig` and `GAConfig`.

### Step 3.3: Wire config to backend

- Build `LeagueConfig` and `GAConfig` from form values when starting a run.
- Validate (e.g. population size ≥ 4 for 4p, etc.) before run.

---

## Phase 4 — Run controls and live metrics

### Step 4.1: Run buttons

- **Start:** Disable config edits, start league run in worker thread, enable Pause/Cancel.
- **Pause at next generation:** Set flag; worker checks after each generation and pauses.
- **Cancel:** Signal worker to stop; handle partial state.
- On pause/finish: re-enable config (or keep disabled until user explicitly resets), enable Export button.

### Step 4.2: Worker thread for league run

- Use `QThread` or `QThreadPool` + `QRunnable` to run the league loop off the main thread.
- Emit signals after each generation: `(population, summary)`.
- Main thread updates state and UI from signals.

### Step 4.3: Live metrics display

- Add labels for ELO min, mean, max (updated from `summary`).
- Add ranking table: best N agents by ELO, ELO gain (requires storing previous ELO for comparison).
- Make table sortable; optionally filter by column.

### Step 4.4: Charts (optional, can defer)

- Add line chart for ELO min/mean/max over generations (e.g. with `pyqtgraph` or `matplotlib` embedded).
- Add fitness-over-generations if desired.
- Ensure charts update when new generation completes.

---

## Phase 5 — Export

### Step 5.1: Export dialog / panel

- Add “Export population” section with:
  - **When:** checkbox “Export at each generation”, “Every X generations” (spinbox), or “On demand only”.
  - **What:** radio or combo: Full / Selected agents / Filtered.
  - For Filtered: filters for ELO range, fitness range, generation range.

### Step 5.2: Export actions

- **On-demand export (button):** Enabled when paused or finished. Opens save dialog; exports current population (full, selected, or filtered) via `population_to_dict` → JSON.
- **Automatic export:** If “at each gen” or “every X gens” is set, call export logic after the relevant generations; use auto-generated filename or user-provided prefix.

### Step 5.3: Export file format

- Write JSON with schema version and metadata (timestamp, config, filters used).
- Ensure `population_from_dict` can read it for Import.

---

## Phase 6 — Polish and integration

### Step 6.1: Bracket matchmaking (if not done)

- Implement bracket-style matchmaking in `tournament.py` if not present.
- Wire League style combobox to backend.

### Step 6.2: Hall of Fame / previous runs as source

- If Hall of Fame or “previous runs” storage exists, add them as sources for “Augment from selection” (e.g. pick from HOF list, then augment).

### Step 6.3: Presets and validation

- Add config presets (e.g. “Quick test”, “Full run”).
- Validate population size vs player count, generation count, etc.
- Show clear error messages on validation failure.

### Step 6.4: UX polish

- Disable inappropriate actions during run (e.g. Import, Add random).
- Add tooltips for parameters.
- Ensure tables are responsive (large populations).

---

## Suggested order of implementation

| Order | Step | Rationale |
|-------|------|-----------|
| 1 | 0.1 | Small, unblocks GA logic |
| 2 | 0.2 | Needed for Import/Export |
| 3 | 0.3 | Needed for Add random / Augment |
| 4 | 1.1, 1.2, 1.3 | Skeleton + table + state |
| 5 | 2.1, 2.2, 2.4, 2.5 | Core population actions |
| 6 | 2.3 | Augment from selection |
| 7 | 3.1, 3.2, 3.3 | Config forms |
| 8 | 0.4, 4.1, 4.2 | Run with pause/cancel |
| 9 | 4.3, 4.4 | Live metrics + charts |
| 10 | 5.1, 5.2, 5.3 | Export |
| 11 | 6.x | Polish, bracket, HOF integration |

---

## Dependencies and files

- **Backend:** `tarot/tournament.py` (Agent), `tarot/ga.py` (selection), `tarot/league.py` (LeagueConfig, run_league_generation), new `tarot/persistence.py` (optional).
- **GUI:** `tarot_gui/main.py` or split into `tarot_gui/league_tab.py` (recommended as the tab grows).
