# Dashboard tab — step-by-step implementation tasks

Step-by-step tasks for each Dashboard block. Run log is inside the Run box (Save / Load run log).

**Reference:** GAMEPLAN §8.1.4 (Dashboard tab).

---

## Current state (last session summary)

**Implemented and tested (107 tests pass):**

- **Run box:** Start/Pause/Cancel, status line (Generation X of Y, Elapsed, ETA with "calculating…" for first gen), run log path/name row, Save/Load run log; RunLogManager; multiple loaded logs.
- **Run log:** JSONL schema, per-generation entries with per-agent snapshots and optional `game_metrics` (deals, petit_au_bout, grand_slem); auto-save to user path/name; Load adds to loaded list; `run_log_loaded` signal updates Charts area.
- **ELO block:** Min/mean/max/std summary; time series chart (elo_min, elo_mean, elo_max vs generation); "—" when no data.
- **Compute block:** Time used, Time left (ETA), Avg time/gen; cleared on run finish.
- **RL performance block:** Top-N table (N configurable 1–50, default 10); columns Name, ELO, Δ (high risers from run log), W/L deal, W/L match, Avg, Gen. Backend: Agent has `deals_won`, `deals_played`, `matches_won`; tournament updates them from game deal outcomes.
- **Game metrics block:** Scope League | Generation | Last N; displays Deals, Petit au bout, Grand schlem. Backend: game returns deal outcome (taker_made, petit, chelem); league aggregates per round; run log entry includes `game_metrics`.
- **Export block:** Placeholder "—" with tooltip "Configure in League Parameters".
- **Charts area:** ELO evolution chart (same QPainter style as League tab); checkboxes per loaded log to include in chart; banner "Viewing saved run(s) + current"; generation slider to cap view; `set_current_entries` / `set_loaded_logs`; tooltip on chart.

**Not yet implemented (optional / later):**

- Origin chart and per-trait variance over time (RL block).
- "Cards" in game metrics (only deals, petit au bout, grand schlem so far).
- Reference line e.g. "Random (4p) ≈ 25%" in RL block.
- Zoom/pan on charts; richer chart tooltips (e.g. value at hover).
- Event overlay on charts.

**Key files:** `src/tarot_gui/dashboard_blocks.py`, `src/tarot_gui/main.py`, `src/tarot_gui/run_log.py`, `src/tarot_gui/league_tab.py` (RunSectionWidget), `src/tarot/tournament.py` (Agent W/L, run_round return metrics), `src/tarot/game.py` (DealOutcome), `src/tarot/league.py` (game_metrics in summary), `src/tarot/persistence.py` (Agent serialization). Tests: `tests/test_dashboard_blocks.py`, `tests/test_run_log.py`, `tests/test_tournament.py`, `tests/test_gui_league.py`, `tests/test_engine.py`.

---

## Decisions locked (from product decisions)

- **Generations total:** From League Parameters currently loaded (League tab).
- **ETA:** `(total_gens - current_gen) * avg_time_per_gen`; first gens "—" or "calculating…".
- **Compute block:** Time used, time left, **average time per generation**.
- **Save run log:** Enabled when run log data exists. **Load:** **Multiple logs** for comparison; checkbox per loaded log to add to charts.
- **Run log state:** **Dedicated Dashboard state object** (e.g. RunLogManager) holds current run log + loaded logs; MainWindow creates it; Dashboard/Run box use it.
- **ELO block:** Graph = ELO of agents/groups over time; side metrics = std, min, max. **Time series first**, optional small snapshot histogram.
- **Top-N:** Default 10, **user-configurable**. **W/L:** User can view **deal-level and match-level** both.
- **High risers:** **From run log:** per-agent ELO stored each gen; Dashboard/analysis computes Δ ELO. Works for current run and loaded logs.
- **Origin:** Origin chart from start; **per-trait variance over time** chart as well.
- **Game metrics:** Implement **petit au bout** and **grand schlem**; scope filter **League | Generation | Last N generations**. Optionally later: animation/event overlay on charts.
- **Compute:** CPU/GPU later.
- **Export placeholder:** "—" / N/A with tooltip "Configure in League Parameters".
- **Charts:** Same library as League tab; banner "Viewing saved run: …"; checkboxes to add loaded logs to charts.
- **Run log content:** Start **per-agent**; option to save summary only; eventually **Settings** controls what is saved.
- **Loaded log navigation:** **Sliders** when data doesn’t show elapsed time (e.g. generation slider).
- **Run log format:** **JSONL** (one line per generation). **Auto-save:** each gen append to a **user-defined path and filename** (user configures where and under what name); user can also "Save run log" to another path (e.g. export).

---

## Block 1 — Run box (controls + status + run log)

**Current state:** ✅ Done. Start/Pause/Cancel, status line (Gen X of Y, Elapsed, ETA), run log path/name + Save/Load; RunLogManager; multiple loaded logs; `run_log_loaded` signal.

### Tasks

1. **Status line**
   - **Task:** Show "Generation X of Y", "Elapsed: …", "ETA: …" (or "—" when not running).
   - **Locked:** Y = generations total from League Parameters (League tab). ETA = `(total_gens - current_gen) * avg_time_per_gen`. Update on generation done only.

2. **Run log buttons**
   - **Task:** Wire Save/Load; file dialog; support **multiple** loaded logs; checkbox per log to include in charts.
   - **Locked:** Save enabled when run log data exists. Multiple logs loadable for comparison. Run log state lives in a **dedicated state object** (e.g. RunLogManager) created by MainWindow and used by Dashboard/Run box.

---

## Block 2 — ELO (observational metrics)

**Current state:** ✅ Done. Min/mean/max/std summary; time series chart (min/mean/max ELO vs generation); "—" when no population.

### Tasks

1. **ELO summary**
   - **Task:** Show min, mean, max, **std** for current population or selected generation (when viewing loaded log).
   - **Locked:** Spread = standard deviation. Show "—" when no population.

2. **ELO chart**
   - **Task:** Main chart = **time series** (ELO of agents/groups over time). Optional small **snapshot histogram** on the side.
   - **Locked:** Time series first; same chart library as League tab.

---

## Block 3 — RL performance

**Current state:** ✅ Done. Top-N table (configurable N), columns Name, ELO, Δ, W/L deal, W/L match, Avg, Gen; high risers from run log. Backend: Agent deals_won/deals_played/matches_won; tournament updates from game outcomes. Not done: origin chart, per-trait variance, reference "Random (4p) ≈ 25%".

### Tasks

1. **Top-N table**
   - **Task:** Table: name/id, ELO, W/L (deal and/or match level), avg score, generation, group/origin. N default 10, **user-configurable** (e.g. spinbox).
   - **Locked:** N configurable; user can switch view to deal-level or match-level W/L.

2. **W/L and reference**
   - **Task:** Backend: add `deals_won`, `deals_played`, `matches_won`, `matches_played` on Agent; update in tournament. UI: reference e.g. "Random (4p) ≈ 25%".
   - **Locked:** Both deal-level and match-level data shown (user choice).

3. **High risers**
   - **Task:** Indicate agents with largest ELO gain vs previous generation (e.g. Δ ELO column or badge). Compute from run log: per-agent ELO is stored each gen; Dashboard/analysis computes Δ ELO.
   - **Locked:** From run log (option A). Works for current run and any loaded log with per-agent data.

4. **Diversity / origin**
   - **Task:** **Origin chart** from start; **per-trait variance over time** chart.
   - **Locked:** As above.

---

## Block 4 — Game metrics

**Current state:** ✅ Done. Deals, petit au bout, grand schlem; scope League | Generation | Last N. Backend: game deal outcome; league aggregates; run log `game_metrics`. "Cards" not added.

### Tasks

1. **Metrics and scope**
   - **Task:** Deals, cards, **petit au bout**, **grand schlem**. Scope filter: **League | Generation | Last N generations**.
   - **Locked:** Implement petit au bout and grand schlem in backend; all three scope options. Optionally later: event overlay on charts.

2. **Update frequency**
   - Refresh on generation_done and on scope/generation change (no sub-gen live update).

---

## Block 5 — Compute

**Current state:** ✅ Done. Time used, ETA, avg time per gen. No CPU/GPU yet.

### Tasks

1. **Time used / time left / avg time per gen**
   - **Task:** Compute block shows elapsed, ETA, and **average time per generation** (Run box keeps one-line status).
   - **Locked:** As above. CPU/GPU later.

---

## Block 6 — Export

**Current state:** ✅ Placeholder. "—" with tooltip "Configure in League Parameters". Real values when League Parameters export/HOF is wired.

### Tasks

1. **Placeholder metrics**
   - **Task:** Until League Parameters defines export/HOF: show "—" or "N/A" with tooltip **"Configure in League Parameters"**. Later: saved agents count, next export gen, HOF count.
   - **Locked:** Placeholder as above; implement real values when League Parameters export/HOF is defined.

---

## Block 7 — Charts area

**Current state:** ✅ Done. ELO evolution from current run + loaded logs; checkboxes per loaded log; banner when viewing saved run(s); generation slider; same chart library (QPainter). Optional: zoom/pan, richer tooltips, fitness/diversity series.

### Tasks

1. **Charts and data**
   - **Task:** ELO evolution, fitness, diversity over time. Data from current run log or loaded log(s); **checkboxes** to add each loaded log to charts; **banner** "Viewing saved run: …" when viewing loaded data.
   - **Locked:** Same chart library as League tab. Multiple loaded logs; checkbox per log to include in charts.

2. **Interaction**
   - Tooltips at least; zoom/pan if library supports it. Generation selection when viewing loaded log: **sliders** when data has no elapsed time.

---

## Run log (inside Run box) — schema and persistence

**Current state:** ✅ Done. JSONL schema; per-generation entries with per-agent snapshots and optional `game_metrics`; Save/Load; multiple loaded logs; auto-save to user path/name; generation slider in Charts area.

### Tasks

1. **Run log schema**
   - **Task:** Per-generation entries + per-agent snapshots (start with per-agent; option for summary-only; eventually Settings controls what is saved). **Format: JSONL** (one line per generation).
   - **Locked:** JSONL; start per-agent; option to save summary only; later Settings for "what is saved".

2. **When to write to disk**
   - **Locked:** **Auto-save each generation**: append one JSONL line to the user-defined path/filename. User configures **where** to save and **under what name** (e.g. in Settings or in Run box / project). "Save run log" button can still export/copy to another path.

3. **Loaded log navigation**
   - **Locked:** Use **sliders** when data doesn’t show elapsed time (e.g. generation slider).

---

## Suggested implementation order

1. ~~Run box: status line; run log Save/Load + schema.~~ ✅
2. ~~Run log: in-memory structure, Save/Load, multiple loaded logs, checkboxes for charts.~~ ✅
3. ~~ELO block: min/mean/max/std; time series chart.~~ ✅
4. ~~Compute block: time used, ETA, avg time per gen.~~ ✅
5. ~~Charts: ELO evolution; data from run log(s); banner; sliders for loaded log.~~ ✅
6. ~~RL performance: Top-N, W/L deal/match, high risers from run log.~~ ✅ (origin chart, per-trait variance: later)
7. ~~Game metrics: deals, petit au bout, grand schlem; scope League / Generation / Last N.~~ ✅
8. ~~Export block: placeholder.~~ ✅
9. Polish: origin chart, per-trait variance, event overlay on charts, zoom/pan, "cards" metric (optional later).

---

## Run log path and name (user-defined)

**Locked:** Auto-save each generation to a **user-defined path and filename**. The user configures:

- **Where** to save log data (directory path).
- **Under what name** (filename, e.g. run_2025-02-22.jsonl or a user-entered name).

Implementation: add a way for the user to set this before or when starting a run (e.g. in Settings, in the Run box, or in project options). When a run starts, the RunLogManager uses this path/name and appends one JSONL line per generation. The Save run log button can still export the current in-memory log to another path (e.g. copy for sharing).
