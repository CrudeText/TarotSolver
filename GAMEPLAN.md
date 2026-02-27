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
| **6** | GUI | Desktop GUI for league, agents, play vs AI, spectate. | 🚧 League tab + Dashboard tab functional; Agents/Play/Settings placeholder. |

Phases 1–5 are implemented at the backend level; Phase 6 has a functional
League tab (population management, project save/load, league and GA config)
and a functional Dashboard tab (Run/File/Export row with run controls and inline compute, a Statistics block (ELO/Fitness/Origin) with integrated game metrics, an RL performance table, and run log wiring). Agents, Play, and Settings remain placeholders.

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
  - **Fitness:** Combine ELOs and average match scores (e.g. `fitness = a×ELO^b + c×avg_score^d`). Agents are **ranked by fitness** (descending); this ordering is used for selection (see §7.4.1).
  - **Selection & reproduction (count-based):**
    - Three counts sum to GA-eligible slots: **Sexual offspring**, **Mutated**, **Cloned**.
    - **Eliminated:** The **worst x by fitness** (x = sexual offspring count) are eliminated; their slots are filled by sexual offspring (two parents from the elite pool).
    - **Elite pool:** The top `(slots - x)` agents by fitness (the non-eliminated). This pool is used for: cloning, mutation parents, and sexual-reproduction parents.
    - **Cloned:** That many slots are filled by exact copies of agents chosen from the elite pool.
    - **Mutated:** That many slots are filled by mutated children of a single parent from the elite pool (roulette or uniform per config).
    - **Sexual offspring:** That many slots are filled by combining two parents from the elite pool; parent selection and trait combination are configurable via the **gearbox** (see §7.4.1).
  - **Hall of fame:**
    - Maintain a small set of best-ever Agents across generations as:
      - Fixed benchmarks in tournaments.
      - Options for custom tables / play vs AI.

#### 7.4.1 Sexual reproduction (design)

- **Eliminated:** The **worst x by fitness** (x = sexual offspring count). These are the non-elite; they are removed to make room for x offspring.
- **Parent pool:** The **elite** = top `(slots - x)` agents by fitness (survivors). Parents for sexual offspring are drawn from this pool only.
- **Parameters (gearbox / settings icon next to “Sexual offspring”):**
  - **Parent selection – with replacement:** Yes / No (whether the same parent can be drawn more than once per offspring or per generation).
  - **Parent selection – weighting:** Fitness-weighted (roulette) / Uniform (random among elite).
  - **Trait combination:** Average (per-trait mean of the two parents) / Crossover (per-trait randomly take one parent’s value). Both options are implemented; user chooses in the gearbox.
- **Order of operations:** (1) Rank eligible agents by fitness (descending). (2) Elite pool = top `(slots - sexual_offspring_count)` agents. (3) Fill clone_count slots with clones from elite pool; mutate_count slots with mutants (one parent from elite); sexual_offspring_count slots with sexual offspring (two parents from elite, combined by selected method).
- **UI:** An **arrow** under the Reproduction selection bar points **to the right**, with the label **“Fitness”** underneath, to indicate that the bar is ordered by fitness (worst → best left to right).

#### 7.4.2 Reproduction counts and inter-generational population change

- **Free adjustment:** Sexual offspring, Mutated, and Cloned counts are adjusted **independently**. The only constraints are: each value at least 0, each at most GA-eligible slots, and the **total must not exceed** the agent population (slots). The total may be **less than** slots (bar not full).
- **Bar not full:** When the sum of the three counts is less than slots, the **gap (unassigned slots)** is shown as a **grey segment on the left** of the Selection bar, followed by sexual (red), mutated (green), cloned (purple). A **warning icon** appears to the right of the bar with a tooltip (e.g. "X slot(s) not assigned"). The next generation fills only the assigned slots (population can shrink).
- **Future: inter-generational population change.** We will at some point explore **inter-generational population change**, where the agent population count can **vary between generations** (e.g. growth or shrinkage based on reproduction counts, or external rules). The current UI already allows total less than slots, which is a first step.

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
- **Layout (Option C — viewport-bound):** Designed for **1080p** (1920×1080). Content width is bound to the viewport (no fixed 1920px) so there is no horizontal scroll. League tab: Project bar, Population block (with internal scroll for the table), arrow, then **two flow rows**: **Row 1** — Tournament | Next Generation (equal width); **Row 2** — Fitness | Reproduction (equal width). Fixed heights per flow row; population block has fixed height with table in an internal scroll area when many rows.
- **Top-level sections:**
  - **Dashboard:** Primary tab during a run. Run controls (Start, Pause, Cancel), ELO (with stability/re-centering), RL performance (W/L, top agents, diversity, origin), game metrics (scope: league/generation), compute, export/HOF placeholders, charts, and run log (save/load for analysis). See §8.1.4 for blocks and layout.
  - **League Parameters:** Population (top), then two flow rows: **Tournament** | **Next Generation** (row 1), **Fitness** | **Reproduction** (row 2). Each box has a fixed height; Reproduction replaces the earlier "Mutation/Clone" label.
  - **Agents & Hall of Fame:** Browse agents, see generations, traits, ELOs.
  - **Custom Tables & Play:** Ad-hoc matches, spectate, play vs AI.
  - **Settings:** Compute, storage, defaults.

### 8.0 Display and layout — general rule (all tabs)

**Reference:** Full study and options: **`docs/display-layout-study.md`**. **Option C is implemented** for all tabs (League, Dashboard, Agents, Play, Settings).

When implementing or changing any tab or GUI element, apply the following so that **all tabs appear in their entirety** at 1080p fullscreen windowed and scale to higher resolutions **without requiring horizontal scroll** (and minimize vertical scroll):

1. **Target viewport:** Design for a content area of **1920×1000** (width × height). Total content height per tab should be ≤ 1000px so that at 1080p maximized (client area ~1005px) no vertical scroll is needed.
2. **No forced horizontal scroll:** Tab content must **never** set a minimum width greater than the **current viewport width**. Bind content width (or max width) to the scroll area viewport width so the user never scrolls sideways for the main layout.
3. **Structure:** Each tab’s main content lives in a **QScrollArea** with an inner widget whose **maximum width** is set to the viewport width on resize. Only **vertical** scroll when content is tall.
4. **Fixed heights:** Use fixed heights for key sections (Project bar, Population block, flow boxes) so the layout is predictable and fits in 1000px; use stretch for at most one “fill” section.
5. **New widgets:** Prefer minimum sizes and expanding size policies; avoid any fixed width larger than the parent/viewport. Use stretch factors so content reflows within the available width.

Implementation checklist and Option C (recommended) details are in `docs/display-layout-study.md`.

### 8.1 League Parameters tab (training configuration)

The **League Parameters** tab is organised into: **Population management**, **Descriptive charts** (placeholder), **League structure**, **GA parameters**, and **Export**.

**Run controls** (Start, Pause, Cancel) and **live ELO metrics** are in the **Dashboard** tab, together with the charts area (Phase 4). Export options remain in League Parameters.

At the **bottom** of the League Parameters board: **two flow rows**. **Row 1:** **Tournament** | **Next Generation** (equal width). **Row 2:** **Fitness** | **Reproduction** (equal width). Fixed heights per row; content width bound to viewport.
- **Fitness box:** Weights (global ELO, avg score) on one line; formula; **fitness graph** (fitness vs ELO) with multiple curves; legend as small "Average Score" table. Selection bar (optional).
- **Reproduction box:** **Sexual offspring**, **Mutated**, **Cloned** as **integer counts**, user adjusts each freely (only constraints: each in [0, slots], total ≤ slots; total may be less). **Gear** icon to the left of the Sexual offspring spinbox opens the **gearbox** dialog. Selection bar: **gap (grey) on the left** when not full, then sexual (red), mutated (green), cloned (purple), each with count; warning icon when bar not full. **Under the bar:** an **arrow pointing right** with the label **“Fitness”** underneath to indicate ranking direction (worst → best). Mutation std and Trait prob below (no frame highlight).
- **Generation flow diagram** (if present): Population → Tournament → Fitness → Reproduction → Next Gen.

#### 8.1.1 Population management

- **Layout:** Tools row (Count, Add random, Import, Augment from selection, Clear selected) **above** the agent groups table, aligned with the table’s left edge. **Left:** pie chart, metric/count table, "Group by" dropdown. **Right:** agent groups table.
- **Population table:** One row per **group**; columns: **Select**, Expand, Color, GA parent, Fixed ELO, Clone only, Play in league, Group name, # agents, Source, ELO (min/mean/max), Actions (Delete). Columns use ResizeToContents except Group name (Stretch); min section size 24px. Table sits in an internal scroll area when there are many rows.
- **Population pie chart:** Pie + metrics table (GA agents, Fixed ELO, Clone only, Play in league, Total) + "Group by" (Group name / GA status / Play in league). Sits to the **left** of the table.
- **Tools:** Count spinbox, Add random, Import, Augment from selection (uses **checked** rows), Clear selected (removes checked groups), Clear (all). Selection is by row checkboxes, not table selection.
- **Per-group flag checkboxes** (with column header tooltips): **GA parent**, **Fixed ELO**, **Clone only**, **Play in league** — as above.

#### 8.1.2 League structure parameters (Tournament box)

- **Categories (checkbox-controlled):**
  - **Core** (always visible, no checkbox): **Rules** dropdown ("3 Players (FFT Rules)", "4 Players (FFT Rules)", "5 Players (FFT Rules)"), **Matchmaking** (ELO-based / Random), **Deals/match** and **Matches/gen** on one row each (pairs on same line). **Under Rules:** a **sideline warning** is shown when total agent count is not a multiple of the selected player count (3, 4, or 5), recommending to adjust the population so every agent can play every round; the message states how many agents will sit out each matchmaking phase.
  - **ELO tuning** (checkbox, **checked by default**): ELO K-factor and ELO margin scale on one line. When unchecked, parameters stay visible but **greyed out**.
  - **PPO fine-tuning** (checkbox, **checked by default**): PPO top-K and PPO updates/agent on one line. When unchecked, parameters **greyed out**.
- **Next Generation:** Generations always visible; **Export** section (Export when, Every N gens, Export what, Export now) is **always visible** — there is **no Export checkbox**; export options are never greyed out by a toggle.

#### 8.1.3 GA parameters (Reproduction box)

- **Reproduction counts:** **Sexual offspring**, **Mutated**, **Cloned** are **integer count** spinboxes. The user can **adjust each freely**; the only constraints are: no value below 0, no value above GA-eligible slots, and **total must not exceed** slots (total may be less). When the bar is not full (total &lt; slots), a **warning icon** appears to the right of the bar with a short tooltip (e.g. "X slot(s) not assigned"). Backend uses the counts as given (see §7.4.1 and §7.4.2).
- **Gearbox (settings icon next to Sexual offspring):** Dialog with tooltips on all parameters.
  - **Parent selection – weighting** (row 1): Fitness-weighted / Uniform.
  - **Parent selection – with replacement** (row 2): Checkbox.
  - **Trait combination** (row 3): Average / Crossover.
- **Selection bar:** When not full, **gap (grey) on the left**; then red/green/purple (sexual/mutated/cloned). Warning icon when not full. **Fitness arrow:** An arrow under the bar points right with the label “Fitness” underneath (worst → best).
- **Mutation std:** Standard deviation for Gaussian perturbation of traits when mutating. **Trait prob (%):** Probability of mutating each trait when creating mutated offspring.
- **Generations:** Number of generations to run (in Next Generation box).
- Fitness calculation (weights for global ELO, average match score) in Fitness box.
- Optional: PPO budget (Tournament box: PPO fine-tuning checkbox, top-K, updates/agent).

#### 8.1.4 Dashboard tab — blocks, metrics, and run log

**Goal:** The user can stay on the Dashboard for an entire run, with macro-level control and dense but clear metrics. All metrics are recorded in a **run log** that the user can save and later load in the same tab for analysis.

**Layout (Option C, target 1920×1000):** Single column, vertical scroll only. Fixed heights for each block so the tab fits in the viewport at 1080p. Order top to bottom:

| Order | Block | Purpose |
|-------|--------|--------|
| 1 | **Run bar** | Start | Pause at next generation | Cancel. One-line status: generation X of Y, elapsed time, ETA. Directly under the status line, an inline **Compute** section shows time used, ETA, and average time per generation. |
| 2 | **File box** | Project-aware file controls. **Load League Project** opens an existing project (same projects list as League Parameters, but without Create). When a project is loaded, the log auto-save directory is set to that project’s `logs/` folder and only the **log file name** is editable. **Browse Logs** scans all projects under the configured Projects folder and lists their `*.jsonl` run logs so the user can select and load past runs for comparison. Save / Load run log buttons remain here. |
| 3 | **ELO block** | ELO metrics only (observational): min/mean/max, spread, optional small chart. User stabilizes ELO themselves (e.g. fixed_ELO reference populations). |
| 4 | **RL performance block** | Top-N agents (ELO, W/L, avg score, high risers), diversity/origin summary. Reference for W/L (e.g. random 4p ≈ 25%). Placed on the same row as the ELO block with equal width. |
| 5 | **Game metrics block** | Deals, cards, petit au bout, grand schlem, etc. Scope filter: League (run) / Generation / (future: match). Updated frequently. |
| 6 | **Export block** | Placeholder: "—" / N/A with tooltip "Configure in League Parameters" until League Parameters defines HOF/export. |
| 7 | **Charts area** | ELO evolution, fitness, diversity (interactive; current run or loaded log). Same chart library as League tab. When logs loaded, checkbox to add each log to charts. |
| — | **Run log** | Lives in the **File** box: Save run log, Load run log, plus logs chosen via Browse Logs. **Multiple logs** can be loaded for comparison. Banner "Viewing saved run: …" when viewing loaded data. |

**Decisions (locked)**

- **Generations total:** From League Parameters currently loaded (League tab); all run parameters come from there.
- **ETA:** `(total_gens - current_gen) * avg_time_per_gen`; first few gens show "—" or "calculating…".
- **Compute block:** Shows time used, time left, and **average time per generation** (in addition to Run box status line).
- **Save run log:** Enabled when there is run log data (current run or loaded log). **Load:** Multiple logs can be loaded for comparison; each loaded log can be toggled on/off for charts via a checkbox.
- **ELO block:** Graph with ELO of agents/groups over time; side metrics: standard deviation, min, max. Time series first; optional small snapshot histogram on the side.
- **RL performance:** Top-N default 10, user-configurable. **W/L:** User can view data at **deal level** and **match level** (both supported). Origin chart from start; per-trait variance over time chart as well.
- **Game metrics:** Implement **petit au bout** and **grand schlem** counters in backend; optionally later add animation or "Event" overlay on charts. Scope filter: **League**, **Generation**, **Last N generations** (all three).
- **Compute:** CPU/GPU implemented later.
- **Export placeholder:** "—" / N/A with tooltip as above.
- **Charts:** Same library as League tab; experiment later if needed. Banner when viewing saved run; checkboxes to add loaded logs to charts.
- **Run log content:** Start with **per-agent** data in saved log; option to save summary only; eventually in **Settings** the user chooses exactly what is saved in a run log.
- **Loaded log UI:** When data doesn’t include elapsed time, use **sliders** (e.g. generation slider) to navigate.

**ELO block (observational only)**

- **Objective:** ELO is a central way to evaluate agent level. The Dashboard shows **metrics only**; the user stabilizes ELO (e.g. fixed_ELO reference populations).
- **Dashboard:** ELO graph (agents/groups over time); side metrics: min, max, std. Time series first; optional small snapshot histogram.

**W/L (wins / losses)**

- **Definition:** **Deal-level:** wins = deals where agent was on winning side; **match-level:** match won or lost. User can view **both** deal-level and match-level data in the UI.
- **Reference:** e.g. “Random (4p) ≈ 25%.” Backend: add per-agent counters (e.g. `deals_won`, `deals_played`, `matches_won`, `matches_played`); tournament layer updates them.

**Game metrics**

- **Content:** Deals, cards, **petit au bout**, **grand schlem**. Scope filter: **League** | **Generation** | **Last N generations**. Backend implements petit au bout and grand schlem counters; run log stores them.

**Compute**

- Time used, time left, **average time per generation**. CPU/GPU later.

**Export and Hall of Fame**

- Placeholder until League Parameters defines it: show "—" / N/A with tooltip "Configure in League Parameters".

**Run log (save / load / browse)**

- **During run:** Metrics written into in-memory run log. **Auto-save:** when a League project is loaded, the Dashboard sets auto-save to that project’s `logs/` folder and only the log file name is user-configurable. **Save:** the user can still export the current run log to any path. **Load/Browse:** the user can load **multiple** logs for comparison; Browse Logs scans all projects under the configured Projects folder and lists their log files; each loaded log can be toggled on/off for charts; banner "Viewing saved run: …".
- **Content:** Start with per-agent snapshots; option for summary-only; eventually Settings controls what is saved.
- **Run log state:** A dedicated **Dashboard state object** (e.g. RunLogManager) holds current run log and the list of loaded logs; MainWindow creates it and the Dashboard/Run/File boxes use it.
- **High risers:** Computed **from the run log** (per-agent ELO per gen stored in log; Dashboard/analysis computes Δ ELO). Works for current run and any loaded log.
- **File format:** **JSONL** (one line per generation); append-friendly for auto-save.
- **Auto-save:** On each generation, append one JSON line to `<project_dir>/logs/<log_file_name>` when a League project is loaded (Dashboard configures this automatically based on the current project and editable log file name). The user can still "Save run log" to a different path (e.g. copy/export).

**Summary: Run controls and live metrics**

- **Location:** Run controls live in the **Dashboard** tab (not League Parameters).
- **Buttons:** Start | Pause at next generation | Cancel.
- **Live metrics (updated during training):** ELO block, RL performance (ranking, W/L, high risers, diversity/origin), game metrics (with scope filter), compute, export placeholders, charts, run log save/load.
- Tables and graphs are interactive during training (sort, filter, zoom, drill down); same when viewing a loaded run log.

**Implementation order:** See **`docs/dashboard-implementation.md`** for step-by-step tasks and decisions per block. Run log lives inside the Run box (Save/Load buttons). Placeholders and layout are in place; implement in order: run bar (status line), ELO block, RL performance, game metrics, compute, export, charts, run log wiring.

#### 8.1.5 Export

- **No checkbox:** Export options are always visible in the Next Generation box (no toggle to grey them out).
- **When to export:** Export when combo: On demand only; Every generation; Every N generations (when the latter is selected, "Every N gens" spinbox is enabled).
- **What to export:** Full population; Top N by ELO; GA-eligible only (Export what combo).
- **Export now** button exports the current population to a JSON file (e.g. for reimport via Import in Population management).
- Export when / every N / what are persisted in project league UI state.

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
| Reproduction | Sexual offspring, Mutated, Cloned (counts), gearbox (parent replacement, weighting, trait combination), Mut. std, Trait prob | GAConfig (counts + sexual params) | Wired |
| Next Gen | Generations spin | run_league_generations(num_generations=...) via LeagueRunWorker | Wired |
| Next Gen | Export when / Every N gens / Export what | Persisted in league UI; auto-export during run loop when implemented | Run not wired |
| Next Gen | Export now button | population_to_json (full population) | Wired |

**Done:** Run controls (Start/Pause/Cancel) connected; `LeagueRunWorker` (QThread) runs `run_league_generations()` off main thread; per-generation update (state, project save, log append, UI refresh); Start enabled only when project loaded.

**TODOs:** Export during run: respect Export when/what in generation callback (no Export checkbox). Export now: full population from state; Export what combo can be used for future filtering. ELO/PPO checkbox state is persisted in league UI.

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
- **Selection & reproduction (see §7.4 and §7.4.1):**
  - **Rank by fitness** (descending). Worst x agents (x = sexual offspring count) are **eliminated**; their slots are filled by **sexual offspring** (two parents from the elite pool). Elite pool = top (slots − x). Remaining slots filled by **clones** and **mutants** from the elite pool.
  - **Sexual reproduction:** Two parents from the elite pool (fitness-weighted or uniform, with or without replacement); traits combined by **average** or **crossover** (gearbox options).
  - **Mutation:** One parent from elite pool; traits perturbed (Gaussian, configurable std and per-trait probability).
  - **Cloning:** Exact copy of an agent from the elite pool.
  - A small **hall of fame** stores the best-ever agents across generations.
- **Future reproduction enhancements (optional):**
    - **Mutation bias:** Non-zero mean for trait perturbation.
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
     - **Option C layout:** Content width bound to viewport; two flow rows — Tournament | Next Generation (row 1), Fitness | Reproduction (row 2). Population in scroll area with table scrolling internally when many rows.
     - Population: tools row above table; pie + metrics + Group by on left; agent groups table; columns ResizeToContents except Group name (Stretch).
     - Tournament: Rules ("N Players (FFT Rules)"), Matchmaking; **sideline warning** under Rules when agent count is not a multiple of player count; ELO tuning and PPO fine-tuning checkboxes **checked by default** (params greyed when unchecked).
     - Fitness: weights on one line; formula; multi-curve fitness graph. Reproduction: **Sexual offspring, Mutated, Cloned** as **counts** (sum = GA slots); gearbox (settings icon) for parent selection and trait combination; selection bar with **fitness arrow** (→ and “Fitness” label) under the bar; Mut. std, Trait prob.
     - Next Generation: Generations; Export section **always visible** (no Export checkbox) — Export when, Every N gens, Export what, Export now.
     - **Run controls (Dashboard):** `LeagueRunWorker` (QThread) runs `run_league_generations()`; Start/Pause/Cancel connected; per-generation state update, project save, log append, table refresh; Start enabled only when project loaded.
   - Next GUI tasks:
     - Dashboard: ✅ Live metrics (Gen X/Y, status line), ELO block, Compute block, RL Top-N table (W/L, Δ ELO), Game metrics (scope + deals/petit/grand schlem), Export placeholder, Charts area (ELO evolution, loaded logs, slider). Optional later: origin chart, per-trait variance, fitness/diversity series, zoom/pan.
     - Export during run: respect Export when/what in generation callback.
     - Implement basic Agents list from league logs plus Hall-of-Fame view.
     - Add a simple Play flow using existing engine/env policies (e.g. run one match, show textual summary).
10. **Project management & experiment tracking (§8.5).**
    - Done: Project save/load (directory + export to JSON), log persistence, GUI actions.
    - Planned: Resume training (§8.5.2), auto-save (§8.5.4), experiment comparison (§8.5.5), agent lineage (§8.5.6), run report (§8.5.7).

### Dashboard — next session ideas

Dashboard tab is implemented (see `docs/dashboard-implementation.md` for current state). Done: run controls, status line, run log Save/Load + JSONL + multiple loaded logs, ELO block (summary + time series), Compute block, RL performance (Top-N, W/L, Δ ELO), Game metrics (scope + deals/petit au bout/grand schlem), Export placeholder, Charts area (ELO evolution, loaded-log checkboxes, banner, generation slider). Backend: Agent has deals_won/deals_played/matches_won; game returns deal outcome; league summary includes game_metrics; run log entries include optional game_metrics.

**Optional / later:**
- Origin chart and per-trait variance over time (RL block).
- Fitness / diversity series in Charts area (log extension).
- Reference line e.g. "Random (4p) ≈ 25%" in RL block.
- Zoom/pan and richer tooltips on charts.
- "Cards" in game metrics; event overlay on charts.

**Other next-session GUI:** Export during run (respect Export when/what); Agents list + HOF view; simple Play flow.

---

*Document version: 1.17 — Dashboard tab implemented: run box (status, run log Save/Load), ELO block, Compute block, RL performance (Top-N, W/L, Δ ELO), Game metrics (scope + deals/petit/grand schlem), Export placeholder, Charts area (ELO evolution, loaded logs, slider). Backend: Agent W/L, game deal outcome, run log game_metrics. See docs/dashboard-implementation.md. Next: optional origin/variance charts; Export during run; Agents/Play.*
