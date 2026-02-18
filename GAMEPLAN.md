# Tarot Solver — Game Plan

A project to train reinforcement learning models to play Tarot (3, 4, and 5 players), run tournaments, and provide a GUI for management and play.

---

## 1. Project overview

| Component | Description |
|-----------|-------------|
| **Game engine** | Full implementation of official Tarot rules (French règlement) for 3, 4, and 5 players. |
| **Model interface** | Plug-and-play API: one model can be used for any player count and any seat. |
| **RL training** | Models trained with only observable information (no hidden info from other players). |
| **Tournament system** | Configurable tournaments (round-robin, elimination, etc.) with full rules. |
| **GUI** | Manage models, configure tournaments, run them, spectate games, play vs. models. |

**Reference:** `Règlement Tarot.pdf` (official French rules in this folder).

---

## 2. Phases (high-level)

| Phase | Name | Main deliverables | Status |
|-------|------|-------------------|--------|
| **1** | Game engine | Core rules, game state, legal moves, scoring; 4 players first, extensible to 3/5. | ✅ Implemented (v0.1.0). |
| **2** | Model interface | Observation/action API, env wrapper, plug-and-play contract. | ✅ Implemented (v0.1.0). |
| **3** | Baseline & validation | Random/rule-based agents, tests, human-playable CLI or minimal UI. | ✅ Implemented (random agent + tests). |
| **4** | RL training pipeline | Training loop, reward design, checkpointing, evaluation. | ✅ Implemented (custom PyTorch PPO, CLI). |
| **5** | Tournament system / league | Tables, matchmaking, GA, league orchestration. | ✅ Core backend implemented (CLI + JSON logs). |
| **6** | GUI | Desktop GUI for league, agents, play vs AI, spectate. | 🚧 League tab functional (population, project, config); Dashboard/Agents/Play/Settings placeholder. |

Phases 1–5 are implemented at the backend level; Phase 6 has a functional
League tab (population management, project save/load, league and GA config)
and placeholder tabs for Dashboard, Agents, Play, and Settings.

---

## 3. Phase 1 — Game engine (detailed)

### 3.1 Scope from the règlement

- **Deck:** 78 cards (22 trumps + 56 suited cards: 14 per suit).
- **Players:** 3, 4, or 5 (with different rules for 5, e.g. discard, “partner” or team). **First version: 4 only.**
- **Deal:** Depends on player count (number of cards per player, dog size, etc.).
- **Bidding:** Contrat (e.g. Petite, Garde, Garde sans, Garde contre); who is the taker (preneur).
- **Dog (chien):** Handling of the dog (shown or not, when, how it affects score).
- **Play:** Trick-taking rules (follow suit, trump, excuses, etc.).
- **Scoring:** Points per trick, bonus/penalties, end-of-deal and match scoring.

**Rules reference:** See **`docs/rules.md`** for a concise checklist extracted from `Règlement Tarot.pdf` (4p, 3p, 5p, scoring, match). Tick off items there as the engine is implemented.

### 3.2 Proposed engine structure

- **Language:** Python recommended (easy integration with RL libs and GUI).
- **Modules (suggested):**
  - `deck.py` — Card, suits, trumps, deck building.
  - `deal.py` — Deal cards, form dog, deal order (player-count aware for future 3/5).
  - `bidding.py` — Bidding round, valid bids, contract types.
  - `play.py` — Trick-taking loop, legal moves, winner of trick.
  - `scoring.py` — Trick points, contract multiplier, end-of-deal score.
  - `game.py` — Orchestrates deal → bid → (dog) → play → score; single deal and **match** (N deals).
- **Extensibility:** Parameterise player count and deal/bid/play rules so adding 3 and 5 players later is mostly config + rule variants, not a rewrite.

### 3.3 Decisions (locked in)

- **5-player variant:** Official règlement variant when we add 5 players.
- **Match vs. single deal:** **Match play.** The number of deals per match is a **tournament parameter** (user chooses). Points are counted as in the official game; after the last deal of a match, **total points per model** determine the **ranking for that match**.
- **Rules scope (first version):** **4 players only** at first. Architecture must make it **relatively easy** to add 3- and 5-player rules later (e.g. player-count–aware deal/bidding/play modules).

---

## 4. Phase 2 — Model interface (plug-and-play)

### 4.1 Goals

- **Same model, any setup:** One trained agent can play as any player index (0..n-1) in 3-, 4-, or 5-player games.
- **No cheating:** The model receives only what a human would see at that moment (own hand, current trick, number of cards in others’ hands, visible dog if applicable, bidding history, etc.).
- **Clear contract:** Inputs (observations) and outputs (actions) are defined so that any implementation (random, rules, RL) can be swapped in.

### 4.2 One model for 3/4/5 vs one model per player count

Play style and win certainty differ a lot by player count: **3p** is relatively easy to win with a good hand, **4p** is harder, and **5p** is easier again and adds **teamwork** (2 vs 3). So we need to choose: one model that plays in any configuration, or separate models per game setup.

| | **One model for 3, 4, and 5 players** | **One model per game setup (3p, 4p, 5p)** |
|---|----------------------------------------|-------------------------------------------|
| **Feasible?** | Yes. Use a **fixed-size observation** (padding for max players/hand size) and include **player count** (and in 5p: team/partner) in the observation. Legal-move masking already handles different hand sizes and rules. One policy can be conditioned on “game type”. | Yes. Separate env and policy per player count; no padding or mode switching. |
| **Pros** | Single checkpoint; “one agent” in the GUI for any table; shared representation (cards, tricks, bidding) across modes. | **Specialized strategy** per mode (solo vs team, risk level); simpler training per model; easier to interpret (“this is our 4p model”). |
| **Cons** | One policy must learn **three different games** (solo 3p, solo 4p, team 5p). Risk of being “average at all” rather than “strong at one”. Training needs a mix of 3p/4p/5p and possibly curriculum. | Three (or more) checkpoints; GUI must pick the right model for the table (can be automatic). No direct transfer across counts. |

**Recommendation:** Prefer **one model per player count** (3p, 4p, 5p) for **stronger play and clearer personalities**, given how different the games are. The GUI can still offer “plug and play” by **auto-selecting** the right model when the user picks an agent and the table is 3/4/5 (e.g. one “agent” in the UI = one 3p model + one 4p model + one 5p model bundled). Optionally, later: **shared backbone + small per-mode heads** (one “model” in the UI, internal specialization) for a compromise between single checkpoint and specialization.

**Decision (to lock in when implementing):** Support **per–player-count models** as the default; keep observation/action interface **unified** so a single code path can load either “one policy for all” or “one policy per count” without changing the rest of the stack.

### 4.3 Observation (what the model sees)

The observation is **phase-dependent** (bidding vs play). All information below is **visible to a human** in that situation — no cheating.

**During bidding:**
- **Own hand** (cards held).
- **Current match context:**
  - **Points of each player** so far in the match (so the model can be more or less aggressive depending on whether it is leading or trailing).
  - **Games played / games left** in the match (e.g. “deal 3 of 5”) so it can take bigger risks on the last deal or play safely when ahead with few deals left.
- **Bidding history so far:** who has bid what (and in what order). E.g. if someone bid before the current player and bid high, that can signal a strong hand; the model can use this when deciding its own bid.

**During play:**
- **Own hand** (current cards; for the taker, after taking the dog: hand including dog cards and knowledge of what was discarded).
- **Full history of cards already played** in this deal (all tricks so far are visible, so the model can infer what remains in each opponent’s hand).
- **Current trick:** cards played this trick, in order; who leads; whose turn it is.
- **Count of cards remaining** in each opponent’s hand (derivable from play history, but can be provided explicitly).
- **Dog (chien):**
  - If the dog has been **revealed** (by rules): its contents.
  - If **this player is the taker** and has taken the dog: what was in the dog, what they **discarded** and what they **kept** (so they know their own full hand and what they gave up).
- **Bidding outcome (parameters from bidding phase):** contract type, who is the taker, who (if any) is the taker’s partner in 5p. Also **who bid what** during the auction (e.g. a player who bid high but didn’t win may have a strong hand — useful for the defense).
- **Running points this deal:** current points for taker vs defense in this deal, if needed for in-game decisions.

**Format:** These can be encoded as fixed-size vectors (one-hot or index-based for cards, scalars for points and game index) or as a **structured** observation (e.g. dict with keys for hand, played_cards, dog, bidding_result, match_scores, games_played/total). Same observation contract across 3/4/5 players via padding or a “player count” field; see §4.5 (flat vs structured).

### 4.4 Actions

- **Bidding:** Discrete: `PASS` or bid (Petite/Prise, Garde, Garde sans, Garde contre) — only when it’s the player’s turn to bid.
- **Playing a card:** Discrete: choose **one card identity** from the global 78-card action space; **only legal moves** are exposed by the engine via masking.

**Decision (locked in):**

- **Single global action space + phase indicator.**
  - We use a **unified discrete action index** where:
    - `0..4` encode bidding actions (`PASS`, `PRISE`, `GARDE`, `GARDE_SANS`, `GARDE_CONTRE`).
    - `5..82` encode **card actions**, using the same `card_index` mapping as the deck/observation (one index per card in the 78-card deck).
  - The observation (or `info`) includes a **phase flag** (bidding vs play); for each step, the environment:
    - Computes a **legal-action mask** over the whole action space.
    - During **bidding**, only indices `0..4` can be legal.
    - During **play**, only indices `5..82` that correspond to cards in hand and legal per the rules can be legal.
- **No separate action spaces per phase** at the environment interface level. Internally, policies are free to implement multi-head architectures if desired (e.g. separate bidding vs play logits), but the env exposes a **single masked discrete space** for compatibility with standard RL libraries.

### 4.5 Observation format: flat vs structured (pros & cons)

| | **Flat vector** | **Structured (e.g. dict + sets/lists)** |
|---|-----------------|----------------------------------------|
| **Pros** | Simple, works with MLPs and standard RL libs; single `observation_space`; easy to mask. | Natural for “sets of cards” and variable-length trick; better fit for attention/transformers later; clearer semantics. |
| **Cons** | Fixed size, padding; less interpretable; may need redesign for 3/5 players. | Slightly more complex env; some RL libs expect flat; may need a “flatten” view for baseline algorithms. |

**Decision (locked in):** **Start with a flat vector** for Phase 2. Keep observation *construction* in a **dedicated module** so that switching to a structured view (e.g. dict) later remains straightforward and does not require rewriting the rest of the stack.

### 4.6 Decisions (locked in)

- **Legal actions:** **Only legal moves** are exposed (environment uses action masking). The model never sees or chooses illegal actions.
- **Reward:** **End of match only** (no per-deal or per-trick reward). Rationale: in Tarot, the goal is not to maximize points every deal; the model must learn to minimize loss, play defensively, take more risk when behind and play conservatively when ahead. Optionally, a **bonus reward at the end of a tournament** (e.g. for winning the final or ranking) can be added for training.

### 4.7 Personality traits (“model personalities”)

- **Goal:** Give each model **tangible, human-readable parameters** so users can quickly understand and compare play styles.
- **Decision (locked in):** Traits are **descriptive by default** — metadata stored with the agent, used for display, filtering, and population/GA augmentation. They do **not** affect policy decisions unless we explicitly add conditioning later.
- **Rationale:** Conditioning (traits as policy input) adds training complexity and hyperparameters. Descriptive traits are simpler, require no policy changes, and still support filtering (e.g. “more defensive agents”), population diversity analysis, and Hall of Fame comparison. Conditioning can be added later for a subset of traits if we want explicit style control.

**Suggested trait list (extensible):**
- **Bidding:** aggressiveness, hand optimism, bluff tendency
- **Play:** trump usage (cut early vs hold), petit-au-bout risk, defensive focus, partner cooperation (5p)
- **Context:** adaptability (to match score), patience

**Representation:** A small **trait vector** attached to each model (e.g. 0–1 per trait). Traits are set via GA mutation, manual input, or (later) post-hoc statistics from observed play.
- **Usage:** Metadata for display, filtering, tournament setup, and GA augmentation. Optionally, **conditioning** (traits as policy input) can be added later for selected traits if we want explicit style control.
- **Modularity:** Implement traits via a small **registry/config** so adding a new trait is: (1) define name, range, default; (2) add GUI representation (label, tooltip, slider); (3) optionally wire into observation for conditioning.

---

## 5. Phase 3 — Baseline & validation

- **Random agent:** Picks uniformly among legal moves.
- **Rule-based agent (optional):** Simple heuristics (e.g. follow suit, play low when possible) to have a non-trivial baseline.
- **Tests:** Unit tests for rules (e.g. legal moves in tricky situations, scoring examples from the règlement).
- **Human play (optional):** Minimal CLI or very simple UI to play one full deal, to validate feel and rules.

---

## 6. Phase 4 — RL training (custom PyTorch)

### 6.1 Overall approach

- **Algorithm (decision):** Custom **PPO-style actor–critic in PyTorch**, not SB3.
- **Policy architecture (first version):**
  - Shared MLP backbone on the **flat observation** (dim 116 in bidding, 412 in play; we pass the appropriate vector).
  - Two small heads:
    - **Policy head:** outputs logits over the global action space (size 83).
    - **Value head:** scalar value estimate for the current state (match-return baseline).
  - Legal-action masking is applied **outside** the network: before sampling we set logits for illegal actions to a large negative value.
- **Seat & episode definition:**
  - **Single learning seat per env** (e.g. player 0 in 4p); other seats use configurable opponent policies (random at first, later other models).
  - **Episode = one match** of N deals (configurable). We use `TarotEnv4P`/`TarotEnv3P`/`TarotEnv5P` as the environment layer; one PPO run initially targets 4p only.

### 6.2 Environment & rollout collection

- **Environment wrapper:**
  - `TarotEnv4P` (already implemented) acts as the “Gym-like” env; a thin adapter provides `reset() -> obs, mask` and `step(action) -> next_obs, reward, done, info, next_mask`.
  - Each env step returns:
    - `obs`: flat observation.
    - `legal_actions_mask`: boolean mask over 83 actions.
    - `reward`: 0 at intermediate decision points, **final match score** at the terminal step.
- **Rollout buffer:**
  - For each step, we store: `(obs, mask, action, log_prob, value, reward, done)`.
  - After collecting T steps (or K episodes), we compute **advantages** and **returns** using GAE(λ).

### 6.3 Losses and updates

- **Policy loss:** standard clipped PPO objective on the masked action distribution.
- **Value loss:** MSE between predicted value and Monte-Carlo return.
- **Entropy bonus:** encourages exploration; weighted term added to the total loss.
- **Update schedule:**
  - Collect rollouts for some number of matches.
  - Run several epochs of minibatch PPO updates on that data.
  - Iterate until a target number of environment steps / matches is reached.

### 6.4 Checkpointing and integration with tournaments

- **Checkpoint format (decision):**
  - Each trained model is saved as a PyTorch `state_dict` plus a small JSON/yaml metadata blob containing:
    - `arch_name` (e.g. `"tarot_mlp_v1"`), observation dim, action dim.
    - Training config snapshot (PPO hyperparameters, seed, etc.).
- **Agent wiring:**
  - `Agent.checkpoint_path` and `Agent.arch_name` reference a saved checkpoint.
  - A **policy registry/loader** maps `(arch_name, checkpoint_path)` → `Policy` instance implementing `act(obs, mask)`.
  - Tournaments and GA rounds (`run_round_with_policies`) construct policies via this loader, so **the same trained model** is used for both evaluation and further training.

### 6.5 Extensions (later, still Phase 4)

- **Multi-agent / self-play:** extend the trainer so the same policy can control multiple seats, or multiple policies can be trained jointly, while preserving the current observation/action encodings.
- **Personality conditioning (optional):** extend the network input with the personality trait vector so a single checkpoint can represent multiple “styles” (see §4.7). Traits remain descriptive by default.
- **Tournament/ELO signal in training:** optionally add a **slow-timescale reward** term computed from tournament/ELO performance (Phase 5) to bias learning towards agents that do well in long-run leagues, not just isolated matches.
- **Compute & deployment:** keep the implementation **CPU-first but GPU-aware**; the GUI will expose a simple “device” setting to select CPU vs GPU when launching training.

---

## 7. Phase 5 — League training via tournaments + GA

In Phase 5, **tournaments become the main way to train agents**. PPO is used as
a local optimiser for promising agents, but the outer loop is a **league**:
repeated tournament rounds, optional PPO fine-tuning, and GA-based evolution.

### 7.1 League structure (outer loop)

- **Population:** A `Population` of `Agent`s. Each `Agent` bundles:
  - Checkpoint path + arch name (policy implementation).
  - Per–player-count ELOs and global ELO.
  - Traits and generation/parents metadata.
- **Generation-based or continuous:** We support:
  - **Generation mode:** For each generation `g`, run:
    1. Tournament rounds to update Elo and match stats.
    2. Optional PPO fine-tunes for top agents (Phase 4 trainer).
    3. GA `next_generation` to create children and elites for generation `g+1`.
  - **Continuous league:** A long-lived population where tournament rounds and
    occasional PPO/GA updates run indefinitely.

### 7.2 Tournament phase (evaluation)

- **Match definition:** As in Phase 1, a match is a full Tarot match (N deals);
  official scoring applies and total points per player determine match ranking.
- **Structure:**
  - Multiple **tables** run in parallel; each table plays a match using
    `run_match_for_table` with policies built via `policy_for_agent`.
  - After each match, we:
    - Record per-agent match scores.
    - Update per-count and global ELOs using `update_elo_pairwise` (margin-aware).
  - After each **round** (all tables finished), `run_round_with_policies` has:
    - Updated ELOs/global rankings.
    - Updated match statistics for fitness computation.
- **Matchmaking:**
  - **Early rounds:** random tables to decorrelate and explore.
  - **Later rounds:** ELO-stratified or Swiss-style pairing (similar-strength).

### 7.3 PPO refinement phase (inner loop, optional)

- For a configurable subset of agents (e.g. top K by fitness/ELO), we:
  - Load their checkpoints into `TarotPPOTrainer` (Phase 4).
  - Run a **short PPO training budget**:
    - Fixed number of updates or env steps.
    - Opponents sampled from the current population (or hall-of-famers).
  - Save updated checkpoints, either:
    - In-place for the same Agent, or
    - As new child Agents with `parents=[...]` and incremented generation.
- This ties local gradient-based learning directly to **tournament success**.

### 7.4 GA evolution phase

- Using GA helpers (`GAConfig`, `compute_fitness`, `next_generation`):
  - **Fitness:** Combine ELOs and average match scores:
    - `fitness = w_global * elo_global + sum_c w_c * elo_c + alpha * avg_match_score`.
  - **Selection:**
    - Keep a configurable elite fraction unchanged (hall-of-fame candidates).
    - Sample parents for the rest based on fitness.
  - **Mutation:**
    - Perturb traits (aggressiveness/defensiveness, etc.).
    - Later: mutate hyperparameters and, optionally, architecture choices.
  - **Hall of fame:**
    - Maintain a small set of best-ever Agents across generations as:
      - Fixed benchmarks in tournaments.
      - Options for custom tables / play vs AI.

### 7.5 Storage and metrics

- **Results & logs:**
  - Per-match results (scores, participants, player count).
  - Per-round standings and ELO updates.
  - Optionally, full game logs for selected matches (e.g. finals, “case studies”).
- **Metrics for GUI:**
  - Per-agent ELO trajectories (3p/4p/5p and global).
  - Fitness over generations.
  - Trait distributions among top agents.
  - Lineage graphs (parents/children) for selected agents.

---

## 8. Phase 6 — GUI (league-centric)

- **Tech (decision):** **Desktop** application (PySide6).
- **Layout (Option 4 — target viewport):** Designed for **1080p** (1920×1080). Window opens **fullscreen windowed** (maximized); minimum size 1920×1080. Fixed section heights so layout is consistent across displays; on WQHD and above the same layout scales (no stretch). League tab content height 1000px: Project 52px, Population 400px, arrow 14px, flow boxes (Tournament, Fitness, Mutation/Clone, Next Generation) 526px each, four equal-width boxes in one row.
- **Top-level sections:**
  - **Dashboard:** Overview of current league, best agents, recent matches. **Run controls** (Start, Pause, Cancel) and **ELO metrics** live here, plus charts area.
  - **League Parameters:** Population (top), then a row of four boxes: **Tournament**, **Fitness**, **Mutation / Clone**, **Next Generation** (name; was "Next Gen"). Each box same width and height.
  - **Agents & Hall of Fame:** Browse agents, see generations, traits, ELOs.
  - **Custom Tables & Play:** Ad-hoc matches, spectate, play vs AI.
  - **Settings:** Compute, storage, defaults.

### 8.1 League Parameters tab (training configuration)

The **League Parameters** tab is organised into: **Population management**, **Descriptive charts** (placeholder), **League structure**, **GA parameters**, and **Export**.

**Run controls** (Start, Pause, Cancel) and **live ELO metrics** are in the **Dashboard** tab, together with the charts area (Phase 4). Export options remain in League Parameters.

At the **bottom** of the League Parameters board: a **single row of four equal-sized boxes** — **Tournament** | **Fitness** | **Mutation / Clone** | **Next Generation** — with descriptive charts and controls inside each.
- **Fitness box:** Weights (global ELO, avg score) on one line; formula; **fitness graph** (fitness vs ELO) with **multiple curves** for avg_score = 0, 25, 50, 75, 100; legend as small **"Average Score"** table (colour + value). Elite % and bar; Selection checkbox.
- **Mutation / Clone box:** Compact top rows (Mutation % param, Mut. % bar, Mut. std) then **Mutation distribution** graph (bell curve, Δ trait vs density) taking most of the height.
- **Generation flow diagram** (if present): Population → Tournament → Fitness → Mutation/Clone → Next Gen.

#### 8.1.1 Population management

- **Layout:** Tools row (Count, Add random, Import, Augment from selection, Clear selected) **above** the agent groups table, aligned with the table’s left edge. **Left:** pie chart, metric/count table, "Group by" dropdown. **Right:** agent groups table.
- **Population table:** One row per **group**; columns: **Select** (checkbox for Augment/Clear selected), Expand, Color, GA parent, Fixed ELO, Clone only, Play in league, Group name, # agents, Source, ELO (min/mean/max), Actions (Delete). Row height 32px; Expand and Delete buttons fixed 52×20 for fit.
- **Population pie chart:** Pie + metrics table (GA agents, Fixed ELO, Clone only, Play in league, Total) + "Group by" (Group name / GA status / Play in league). Sits to the **left** of the table.
- **Tools:** Count spinbox, Add random, Import, Augment from selection (uses **checked** rows), Clear selected (removes checked groups), Clear (all). Selection is by row checkboxes, not table selection.
- **Per-group flag checkboxes** (with column header tooltips): **GA parent**, **Fixed ELO**, **Clone only**, **Play in league** — as above.

#### 8.1.2 League structure parameters (Tournament box)

- **Categories (checkbox-controlled):**
  - **Core** (always visible, no checkbox): **Rules** dropdown ("3 Players (FFT Rules)", "4 Players (FFT Rules)", "5 Players (FFT Rules)"), **Matchmaking** (ELO-based / Random), **Deals/match** and **Matches/gen** on one row each (pairs on same line).
  - **ELO tuning** (checkbox, unchecked by default): ELO K-factor and ELO margin scale on one line. When unchecked, parameters stay visible but **greyed out**.
  - **PPO fine-tuning** (checkbox, unchecked by default): PPO top-K and PPO updates/agent on one line. When unchecked, parameters **greyed out**.
- Same pattern for **Fitness** (Weights always visible; **Selection** checkbox for Elite % + bar) and **Next Generation** (Generations always visible; **Export** checkbox — when unchecked, export options **greyed out**).

#### 8.1.3 GA parameters

- **Elite fraction (%):** Top fraction preserved unchanged each generation.
- **Mutation prob (%):** Probability of mutating each trait when creating children.
- **Mutation std:** Standard deviation for Gaussian perturbation of traits (e.g. 0.1 = small random changes). Used when mutating; see GA visual in charts area.
- **Generations:** Number of generations to run.
- Fitness calculation (weights for global ELO, per-count ELO, average match score).
- Optional: PPO budget (which agents get fine-tuned, steps/updates).

#### 8.1.4 Run controls and live metrics (Dashboard tab)

- **Location:** Run controls live in the **Dashboard** tab (not League Parameters).
- **Buttons:** Start | Pause at next generation | Cancel.
- **Live metrics (updated during training):**
  - ELO average, min, max.
  - Ranking table (best agents by ELO, ELO gain, etc.).
  - Charts (ELO evolution, fitness, diversity).
- Tables and graphs are interactive during training (sort, filter, zoom, drill down).

#### 8.1.5 Export

- **Checkbox-controlled:** Export is toggled by a checkbox. When unchecked, Export when, Every N gens, Export what, and Export now are greyed out.
- **When to export:**
  - Option: export at each generation.
  - Option: export every X generations.
  - On pause or finish: button to export the current population.
- **What to export:**
  - Full population.
  - Selected agents (manual selection).
  - Filtered subset (by ELO range, fitness, generation, etc.).
- Exported populations can be reimported as “Import from previous run” in Population management.

#### 8.1.6 League Parameters — backend wiring status

**Control → Backend mapping (League Parameters tab):**

| Section | Control | Backend / Use | Status |
|---------|---------|---------------|--------|
| Project | New, Open, Save, Save As | project_save, project_load | Wired |
| Project | Export JSON, Import JSON | project_export_json, project_import_json | Wired |
| Population | Add random, Import, Augment, Clear | LeagueTabState.groups | Wired |
| Tournament | Rules (3/4/5 players) | LeagueConfig.player_count | Wired |
| Tournament | Matchmaking (ELO-based / Random) | LeagueConfig.matchmaking_style | Wired |
| Tournament | Deals/match, Matches/gen | LeagueConfig.deals_per_match, rounds_per_generation | Wired |
| Tournament | ELO tuning checkbox | When unchecked: defaults (K=32, margin=50). When checked: use spins | Wired |
| Tournament | PPO fine-tuning checkbox | When unchecked: ppo_top_k=0, ppo_updates_per_agent=0 | Wired |
| Fitness | Weight (global ELO), Weight (avg score) | LeagueConfig.fitness_weight_* | Wired |
| Reproduction | Elite %, Mutated %, Cloned %, Mut. std, Trait prob | GAConfig | Wired |
| Next Gen | Generations spin | run_league_generations(num_generations=...) via LeagueRunWorker | Wired |
| Next Gen | Export checkbox | Greys params; used when run loop exports | Run not wired |
| Next Gen | Export when / Export what | Auto-export during run loop | Run not wired |
| Next Gen | Export now button | population_to_json (full population) | Wired |

**Done:** Run controls (Start/Pause/Cancel) connected; `LeagueRunWorker` (QThread) runs `run_league_generations()` off main thread; per-generation update (state, project save, log append, UI refresh); Start enabled only when project loaded.

**TODOs:** Export during run: respect Export checkbox and Export when/what in generation callback. Export now: always full population; Export what combo ignored. Checkbox state (ELO/PPO) not persisted in projects.

### 8.2 Agents & Hall of Fame

- **Agents list:**
  - Columns: name, id, generation, parents, ELO_3p/4p/5p, global ELO, traits, checkpoint status.
  - Filters by generation, trait ranges, hall-of-fame status, player count strength.
- **Agent detail:**
  - Lineage (parents/children, generation).
  - ELO/fitness history over generations.
  - Training history: which PPO runs updated this agent.
  - Actions: mark as hall-of-famer, freeze as benchmark, create custom table.
- **Hall of Fame:**
  - Highlight best-ever agents with quick access to:
    - Start tournament vs current population.
    - Create “play vs AI” tables.

### 8.3 Custom Tables, Spectate, and Play vs AI

- **Custom tables:**
  - Editor to pick player count, deals per match, and per-seat agents:
    - Any Agent, hall-of-famer, random baseline, or human.
  - Run modes:
    - Batch simulation (stats only).
    - Spectate: show tricks/bids over time.
- **Spectate / Replays:**
  - Replay browser with filters (agents, generation, tournament).
  - Viewer with timeline of plays and optional policy overlays (top-k action probs).
- **Play vs AI:**
  - Choose seat, opponents, and table preset.
  - Game UI for human play, with optional “explain move” overlays from the policy.

### 8.4 Settings

- **Compute:** default device, max concurrent PPO jobs.
- **Storage:** checkpoint retention, replay logging level.
- **Projects folder:** Base directory where all projects are stored (default: e.g. `D:/1 - Project Data/TarotSolver`). New projects are created as subdirectories here. Set in Settings tab.
- **League defaults:** reusable presets for league configuration.
- **UI:** theme, chart smoothing, refresh intervals.

### 8.5 Project management & experiment tracking

#### 8.5.0 File menu and New Project dialog

- **File button:** Single "File" button in the Project bar (replaces separate New, Open, Save, Save As, Export JSON, Import JSON buttons). Opens a dropdown menu with:
  - New Project → opens New Project dialog
  - Open Project → opens same dialog (list existing, open selected)
  - Save, Save As, Export JSON, Import JSON → same behavior as before
- **New Project dialog:**
  - Shows projects folder (read-only; configured in Settings).
  - List of existing projects (subdirs containing `project.json`) in that folder. Double-click or select + Open to load.
  - Project name input for creating a new project. Create button validates name, creates `{projects_folder}/{name}` if available, saves initial state, loads the project.
  - Validation: reject empty/invalid names; reject names that already exist.

#### Implementation status (Phase 1 — done)

- **Project directory format:** Projects live under the configurable projects folder. Layout per project:
  - `project.json` — groups, league config, GA config, generation index, last summary.
  - `checkpoints/` — agent checkpoints (saved directly into project during league run).
  - `logs/league_run.jsonl` — per-generation ELO metrics (JSON Lines).
- **All paths relative** within the project directory.
- **Export to single JSON:** For easy sharing; includes population, config, logs. Checkpoint paths stored as relative.
- **GUI:** File menu (New Project, Open Project, Save, Save As, Export JSON, Import JSON) in League tab. New/Open use the New Project dialog with name input and existing projects list.
- **Log persistence:** `run_league_generations` accepts `log_path` and appends each generation.
- **Checkpoints:** League run accepts `checkpoint_base_dir`; PPO saves into project directory.
- **No RNG reproducibility** in this phase.

#### 8.5.1 Project save/load

- **Project file/directory:** A single project bundles everything needed to restore the full state:
  - Population (groups + agents with ELOs, traits, checkpoint references).
  - League config (player count, deals/match, rounds/gen, matchmaking, GA params).
  - PPO/training config (if used).
  - Generation index (last completed generation).
  - Random seed (for reproducibility).
  - Log data reference or embedded logs (see §8.5.3).
  - Run report (see §8.5.7).
- **Load:** Restore full UI state (groups, config, charts data) and allow resuming.
- **Save as:** Branch experiments by saving a copy under a new name without overwriting the current project.

#### 8.5.2 Resume training

- **Pause at generation boundary:** When the user pauses at the end of a generation, store the current population and generation index.
- **Resume:** Continue from the next generation with the same population and config (same RNG state if reproducibility is enabled).
- **GUI and CLI:** Both support resume (e.g. `tarot league-4p --resume project.json`).

#### 8.5.3 Log data persistence

- **During league/training runs:** Append to `logs/league_run.jsonl` (JSON Lines):
  - Per generation: ELO min/mean/max, agent count, timestamp.
  - `run_league_generations(log_path=...)` writes automatically.
- **Stored in project:** Log file at `project_dir/logs/league_run.jsonl`.
- **Load for charts:** `load_league_log(project_dir)` returns entries for ELO-over-time, etc.

#### 8.5.4 Auto-save

- **Configurable interval:** Save project state periodically (e.g. every N generations or every N minutes).
- **Configurable retention:** Keep the last K auto-save checkpoints (e.g. 3) to avoid unbounded disk use.
- **Crash recovery:** On restart, offer to resume from the most recent auto-save.

#### 8.5.5 Experiment comparison

- **Multiple runs per project (or linked projects):** Store/load several runs (different configs, populations, or seeds).
- **Side-by-side comparison:** Compare ELO curves, final populations, and config differences across runs.
- **Export:** Optionally export comparison tables or overlay charts.

#### 8.5.6 Agent lineage and history

- **Parent–child tracking:** Use existing `parents` metadata to track lineage across generations.
- **Lineage browser:** View agent family tree (ancestors and descendants).
- **History:** Per-agent summary of generations participated, ELO trajectory, and PPO updates (when applicable).
- **Export:** Optional lineage graph (e.g. DOT/Graphviz or interactive HTML).

#### 8.5.7 Run report (included in project)

- **Report generation:** Produce a run report (PDF/HTML/Markdown) for a project or run:
  - Config snapshot (league, GA, PPO if used).
  - ELO progression (min/mean/max over generations).
  - Top agents and lineage snippets.
  - Charts (ELO curves, optional loss curves).
- **Stored in project:** The report is saved as part of the project files (e.g. `report.html` or `report.pdf` in the project directory).
- **Regenerate on demand:** Option to refresh the report when the project is updated.

---

## 9. Summary of decisions

| # | Topic | Decision |
|---|--------|----------|
| 1 | 5-player | Official règlement variant when we add 5 players. |
| 2 | Match vs. deal | Match play; # of deals per match = user parameter; total points → match ranking. |
| 3 | Rules scope | 4 players first; architecture extensible to 3 and 5. |
| 4 | Observation | Start flat; observation built in a dedicated module for easy switch to structured later. |
| 5 | Legal actions | Only legal moves (masking). |
| 6 | Reward | End of match only; optional tournament bonus. |
| 7 | Compute | GPU when available, else CPU; user can switch in GUI. |
| 8 | Storage | Results only by default; user can choose to save specific games (e.g. finals) in advance. |
| 9 | GUI priority | Start simple (tournaments, results, training, metrics); spectate and play-vs-AI later. |
| 10 | Action space | Separate bidding phase and playing phase (two distinct decision steps). |
| 11 | GUI tech | Desktop. |
| 12 | Personality traits | Models have modular, human-readable traits; **descriptive by default** (metadata only); conditioning optional later. |
| 13 | One model vs per–player-count | Agents can support multiple player counts; we track **per–player-count ELOs** and a **global ELO** so that a single agent can be viewed as a “global player” while still seeing which variants it excels at. |

---

## 12. Genetic algorithm & population-based training (high-level)

- **Individual:** One `Agent` (policy + metadata) that can support one or more player counts (3/4/5).
- **Ratings:**
  - For each agent we keep:
    - `elo_3p`, `elo_4p`, `elo_5p` (when applicable).
    - `elo_global` (aggregated across counts), used as the default ranking / fitness component.
  - ELO updates:
    - Applied **per player count** after each match using multi-player pairwise comparisons.
    - **Score margin–aware:** the match-score difference influences how far the Elo “result” is from a pure 0/1 (big wins vs strong opponents → larger adjustments).
- **Fitness (for evolution):**
  - Default fitness = function of ELOs and scores, e.g.:
    - `fitness = w_global * elo_global + sum_c w_c * elo_c + alpha * avg_match_score`.
  - All weights (`w_global`, per–player-count weights, `alpha`) are **configurable** (GUI-controlled) to let the user shift focus between 3p/4p/5p or between raw ELO and average scores.
- **Selection & reproduction:**
  - **Selection:** rank-based or tournament selection using the fitness values, with:
    - **Elites:** top fraction preserved unchanged each generation.
    - Parameters like elite fraction, tournament size, selection temperature exposed as config (GUI-tunable).
  - **Reproduction:**
    - Children are created from one parent (asexual; *two-parent crossover is a future enhancement*):
      - Copy architecture/checkpoint references.
      - Record `parents` and increment `generation`.
      - **Mutate**: Traits (e.g. aggressiveness/defense sliders) via small random perturbations. *Optional: mutation bias (non-zero mean) is a future enhancement.*
      - Later: hyperparameters and, when integrated, model weights.
    - A fraction of offspring may be exact clones of elites (configurable).
    - A small **hall of fame** stores the best-ever agents (by global ELO/fitness) across generations for analysis and as potential fixed opponents.
- **Future reproduction enhancements (optional):**
    - **Mutation bias:** Non-zero mean for trait perturbation (e.g. bias toward higher/lower values).
    - **Two-parent crossover:** Agent/agent reproduction with trait crossover from two parents.
    - **Per-trait mutation controls:** Per-trait mutation probability or std.
- **Matchmaking during GA:**
  - Multiple strategies are supported and selectable in the GUI:
    - **Random tables** at the very start (to quickly de-correlate and spread ELOs).
    - **ELO-stratified tables** in later rounds (similar-strength agents grouped together for finer ordering).
    - Hybrid schedules (e.g. a few random rounds, then ELO-based rounds) can be configured.
  - All of this runs on top of the `TarotEnv*` match-level environments; the genetic algorithm layer never sees game internals, only match results and ratings.

---

## 10. Pre-start: elements to decide or clarify

Before or as we start implementation, the following should be agreed or clarified so we don’t have to rework later. All pre-start items are now resolved (see Summary of decisions and Next steps).

### Must resolve before Phase 1 (game engine)

| Item | Why it matters |
|------|----------------|
| **Rules checklist** | The plan refers to `Règlement Tarot.pdf` but there is no extracted checklist yet. The engine needs exact 4p rules: cards per player, dog size, deal order, bidding order, which contracts (Petite, Garde, Garde sans, Garde contre?), when the dog is shown, scoring formula, handling of Excuse, etc. **Options:** (a) Create `RULES_CHECKLIST.md` (or `docs/rules.md`) from the PDF first and tick off as we implement, or (b) Implement in parallel and document rules as we go. |
| **Match scoring** | “Points as in the official game” — confirm whether we use raw deal points summed over the match, or the règlement’s conversion (e.g. deal points × contract multiplier → “game points”) and then sum game points per player. Affects both engine and tournament ranking. |
| **Tie-breaking in a match** | If two (or more) players have the same total points at the end of a match, how do we rank them? (Règlement may define this; if not, we need a rule.) |

### Should resolve early (Phase 1–2) but not blocking

| Item | Why it matters |
|------|----------------|
| **Observation format (flat vs structured)** | §4.5 leaves this “TBD at Phase 2”. Recommendation is “start flat”. We can lock this when starting Phase 2; Phase 1 is independent. |
| **Action space (single vs separate bid/play)** | One global action space vs separate “bid” and “play card” phases affects the env API. Can be decided at the start of Phase 2. |
| **GUI tech** | §8: “Desktop or web? Your preference?” — Choosing early influences project layout and dependencies; can still be decided when starting Phase 6. |
| **GUI priority** | “Minimal first vs spectate/play-vs-AI early” (Summary #9) — Affects order of GUI features; can be set when we start Phase 6. |

### Can defer until later phases

| Item | When to decide |
|------|----------------|
| **Tournament matchmaking** | “Top half by score” vs “bracket-based” (and whether we support both) — Phase 5. |
| **Personality traits: learned vs fixed** | Whether traits are learned (emergent) or user-set conditioning — Phase 4 / trait design. |
| **Project structure & tooling** | Repo layout, Python version, dependency manager (pip/poetry/conda) — Can be fixed when creating the repo at start of Phase 1. |

### Summary: minimum to unblock Phase 1

1. **Rules source:** Either create a short **rules checklist** from the PDF (4p: deal, dog, bidding, play, scoring) or agree to document rules as we implement from the PDF.
2. **Match scoring:** Confirm how match total is computed (raw sum of deal points vs game-point conversion).
3. **Tie-breaking:** Define how to rank players when match points are tied.

Once these are clear, Phase 1 can start; the rest can be decided as we reach each phase.

---

## 11. Next steps

1. ~~You answer the questions above.~~ **Done.**
2. ~~Update this game plan with your choices.~~ **Done.**
3. ~~Resolve pre-start items (§10).~~ **Done** (rules in `docs/rules.md`, official match scoring, tie-breaking noted).
4. ~~Phase 1 (game engine, 3/4/5 players).~~ **Done.** Engine in `src/tarot/`: deck, deal, bidding, play, scoring, game; Excuse, match, poignée, chelem. See README.
5. ~~Phase 2 (model interface, observation/action, env wrapper).~~ **Done.** Flat obs + global action space + `TarotEnv3P/4P/5P` with tests.
6. ~~Phase 3 (baselines & validation).~~ **Done.** `RandomAgent`, engine/env/tournament tests, simple CLI/usage examples.
7. ~~Phase 4 (RL training).~~ **Done.** Custom PyTorch PPO trainer, checkpointing, `tarot train-ppo-4p` / `tarot eval-4p`, `NNPolicy` loader.
8. ~~Phase 5 (tournaments + league + GA).~~ **Done (backend).** `run_round_with_policies`, GA helpers, `run_league_generation`, `tarot league-4p` CLI with JSON logs.
9. **Phase 6 (GUI).**
   - Current: PySide6 app with tabs for Dashboard, League Parameters, Agents, Play, Settings. **League Parameters** tab is functional (layout, population, config, export now). **Dashboard** run controls are wired:
     - **Option 4 layout:** Target 1080p; min window 1920×1080; fullscreen windowed (maximized). Fixed heights: Project 52, Population 400, arrow 14, four flow boxes 526px each.
     - Population: tools row above table; pie + metrics + Group by on left; agent groups table with Select checkbox column, Expand/Delete buttons 52×20, row height 32.
     - Tournament: Rules ("N Players (FFT Rules)"), Matchmaking; Core always visible; ELO tuning and PPO fine-tuning with checkboxes (params greyed when unchecked).
     - Fitness: weights on one line; formula; multi-curve fitness graph with "Average Score" table legend; Selection checkbox.
     - Mutation/Clone: compact param rows, mutation distribution graph; Next Generation: Generations, Export checkbox (params greyed when unchecked).
     - **Run controls (Dashboard):** `LeagueRunWorker` (QThread) runs `run_league_generations()`; Start/Pause/Cancel connected; per-generation state update, project save, log append, table refresh; Start enabled only when project loaded.
   - Next GUI tasks:
     - Dashboard: live metrics (Gen X/Y, N agents, status line), charts (ELO evolution, fitness, diversity), ranking table, overview.
     - Export during run: respect Export checkbox and Export when/what.
     - Implement basic Agents list from league logs plus Hall-of-Fame view.
     - Add a simple Play flow using existing engine/env policies (e.g. run one match, show textual summary).
10. **Project management & experiment tracking (§8.5).**
    - Done: Project save/load (directory + export to JSON), log persistence, GUI actions.
    - Planned: Resume training (§8.5.2), auto-save (§8.5.4), experiment comparison (§8.5.5), agent lineage (§8.5.6), run report (§8.5.7).

### Dashboard — next session ideas

Planned work for the Dashboard tab (see §8.1.4 Run controls and live metrics). Suggested order:

1. ~~**Run controls wiring (highest priority)**~~ **Done.**
   - ~~Implement `LeagueRunWorker` (QThread) to run `run_league_generations()` off the main thread.~~
   - ~~Connect Start / Pause / Cancel to worker. Use `LeagueRunControl` for Cancel; Pause = stop iterating after current generation.~~
   - ~~Emit signal per generation → update state, save project, append log, refresh UI.~~
   - ~~Require loaded project before Start is enabled.~~
2. **Live metrics**
   - Expand ELO display: min / mean / max (already present), add Gen X/Y, N agents.
   - Status line: Idle, Running, Paused, Completed.
   - Fixed layout for 1080p (compact run section, charts area below).
3. **Charts**
   - **ELO evolution:** X = generation, Y = ELO; plot min/mean/max curves from `load_league_log(project_dir)`. Update live during run.
   - **Fitness over generations:** requires extending log with fitness_min/mean/max per generation.
   - **Diversity:** ELO spread (max−min) or trait variance; may need log extension.
4. **Ranking table**
   - Top agents: rank, name, generation, global ELO, ELO 3p/4p/5p, fitness, matches, avg score.
   - Sort by ELO descending. Data from `state.build_population()`.
5. **Overview**
   - Summary: project name, generations completed, population size, best ELO, best agent.
   - Recent matches (later): requires match-result persistence.

**Data:** `load_league_log()` gives per-generation `elo_min/mean/max`, `num_agents`, `timestamp`. Agents have `elo_global`, `matches_played`, `total_match_score`, `generation`, etc.

---

*Document version: 1.13 — Dashboard run controls wired (LeagueRunWorker, Start/Pause/Cancel, per-gen save/refresh); next: live metrics, charts, ranking table, overview.*
