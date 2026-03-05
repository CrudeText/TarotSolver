"""
League tab: population management (groups of agents), league structure, GA parameters, run controls.

Groups contain agents; the main table shows one row per group. Expand opens a detail dialog.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from tarot.ga import GAConfig
from tarot.project import (
    load_league_log,
    project_export_json,
    project_import_json,
    project_load,
    project_save,
)
from tarot_gui.project_dialog import NewProjectDialog, OpenProjectDialog, _list_existing_projects
from tarot_gui.themes import get_projects_folder
from tarot_gui.charts import (
    FitnessVisualWidget,
    ReproductionBarWidget,
    GroupSliceData,
    MutationDistWidget,
    PIE_COLORS,
    PopulationPieWidget,
)


class _ResizeFilter(QtCore.QObject):
    """Calls a callback when the filtered object is resized."""

    def __init__(self, parent: QtCore.QObject, on_resize: callable) -> None:
        super().__init__(parent)
        self._on_resize = on_resize

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Resize:
            self._on_resize()
        return super().eventFilter(obj, event)
from tarot.ga import compute_fitness
from tarot.league import LeagueConfig, LeagueRunControl, run_league_generations
from tarot.persistence import population_from_dict, population_to_json, load_population_from_directory
from tarot.population_helpers import clone_agents, generate_random_agents, mutate_from_base
from tarot.project import get_checkpoint_base_dir, get_log_path
from tarot.tournament import Agent, Population

from .run_log import RunLogManager
from .dashboard_blocks import ComputeBlockWidget, ExportBlockWidget


def _format_duration(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    if seconds < 0 or not isinstance(seconds, (int, float)):
        return "0:00"
    total = int(round(seconds))
    if total >= 3600:
        h, r = divmod(total, 3600)
        m, s = divmod(r, 60)
        return f"{h}:{m:02d}:{s:02d}"
    m, s = divmod(total, 60)
    return f"{m}:{s:02d}"


class RunStatus(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    PAUSED = "paused"


@dataclass
class Group:
    """A group of agents, shown as one row in the main table."""

    id: str
    name: str
    agents: List[Agent] = field(default_factory=list)
    source_group_id: Optional[str] = None
    source_group_name: Optional[str] = None
    color: int = 0x4A90D9  # RGB hex, used in pie chart and table swatch

    def all_can_use_as_ga_parent(self) -> bool:
        return all(a.can_use_as_ga_parent for a in self.agents)

    def set_all_can_use_as_ga_parent(self, value: bool) -> None:
        for a in self.agents:
            a.can_use_as_ga_parent = value

    def all_fixed_elo(self) -> bool:
        return all(a.fixed_elo for a in self.agents) if self.agents else False

    def set_all_fixed_elo(self, value: bool) -> None:
        for a in self.agents:
            a.fixed_elo = value

    def all_clone_only(self) -> bool:
        return all(a.clone_only for a in self.agents) if self.agents else False

    def set_all_clone_only(self, value: bool) -> None:
        for a in self.agents:
            a.clone_only = value

    def all_play_in_league(self) -> bool:
        return all(a.play_in_league for a in self.agents) if self.agents else True

    def set_all_play_in_league(self, value: bool) -> None:
        for a in self.agents:
            a.play_in_league = value

    def elo_min(self) -> float:
        return min(a.elo_global for a in self.agents) if self.agents else 0.0

    def elo_mean(self) -> float:
        if not self.agents:
            return 0.0
        return sum(a.elo_global for a in self.agents) / len(self.agents)

    def elo_max(self) -> float:
        return max(a.elo_global for a in self.agents) if self.agents else 0.0


# Name library for random groups (no duplicates when picking)
GROUP_NAMES = [
    "Marseille",
    "Rider-Waite",
    "Thoth",
    "Wild Unknown",
    "Morgan-Greer",
    "Cosmic Tarot",
    "Robin Wood",
    "Hanson-Roberts",
    "Universal Waite",
    "Aquarian",
    "Light Seer",
    "Modern Witch",
    "Shadowscapes",
    "Mystic Mondays",
    "Ethereal Visions",
    "Oriens",
    "Spiritual Tarot",
    "Star Spinner",
    "Luminous Void",
    "Numinous",
]

# Group ID counters
_group_counters: Dict[str, int] = {"rand": 0, "mut": 0, "imp": 0}


def _pick_random_group_name(used_names: set[str], rng: random.Random) -> str:
    """Pick a random name from GROUP_NAMES that is not in used_names. Fallback if exhausted."""
    available = [n for n in GROUP_NAMES if n not in used_names]
    if available:
        return rng.choice(available)
    # Fallback when all names used
    return f"Cohort {rng.randint(1000, 9999)}"


def _pick_group_color(used_colors: set[int], rng: random.Random) -> int:
    """Pick a color from PIE_COLORS not yet used. Cycle if all used."""
    hex_colors = [c[0] for c in PIE_COLORS]
    available = [c for c in hex_colors if c not in used_colors]
    if available:
        return rng.choice(available)
    return rng.choice(hex_colors)


def _next_group_id(prefix: str) -> str:
    c = _group_counters.get(prefix, 0)
    _group_counters[prefix] = c + 1
    return f"grp_{prefix}_{c}"


def _assign_group_agent_ids(agents: List[Agent], group_id: str) -> List[Agent]:
    """Assign agent IDs {group_id}_0, {group_id}_1, ... Returns new Agent instances."""
    result: List[Agent] = []
    for i, a in enumerate(agents):
        new_id = f"{group_id}_{i}"
        result.append(Agent(
            id=new_id,
            name=a.name,
            player_counts=list(a.player_counts),
            elo_3p=a.elo_3p,
            elo_4p=a.elo_4p,
            elo_5p=a.elo_5p,
            elo_global=a.elo_global,
            generation=a.generation,
            traits=dict(a.traits),
            checkpoint_path=a.checkpoint_path,
            arch_name=a.arch_name,
            parents=list(a.parents),
            can_use_as_ga_parent=a.can_use_as_ga_parent,
            fixed_elo=a.fixed_elo,
            clone_only=a.clone_only,
            play_in_league=a.play_in_league,
            matches_played=a.matches_played,
            total_match_score=a.total_match_score,
        ))
    return result


def _agent_id_belongs_to_group(agent_id: str, group_id: str) -> bool:
    """
    Heuristic mapping from agent id back to its original group.

    Agents created in the League tab are assigned ids with the pattern
    "{group_id}_<index>". During GA evolution, new ids are derived from the
    parent id by appending a suffix (for example "-c1" or "-clone0"), so the
    original group_id remains as a prefix.

    This helper treats any agent whose id starts with "{group_id}_" or
    "{group_id}-" as belonging to that group.
    """
    if not agent_id or not group_id:
        return False
    if agent_id == group_id:
        return True
    if agent_id.startswith(f"{group_id}_"):
        return True
    if agent_id.startswith(f"{group_id}-"):
        return True
    return False


@dataclass
class LeagueTabState:
    """State for the League tab. Groups hold agents; population is built from groups."""

    groups: List[Group] = field(default_factory=list)
    run_status: RunStatus = RunStatus.IDLE
    last_summary: Optional[Dict[str, float]] = None
    project_path: Optional[str] = None
    generation_index: int = 0
    hof_agents: List[Agent] = field(default_factory=list)  # Hall of Fame: snapshots of best agents

    def build_population(self) -> Population:
        """Build flat Population from all agents in all groups (for backend)."""
        pop = Population()
        for g in self.groups:
            for a in g.agents:
                pop.add(a)
        return pop

    def total_agents(self) -> int:
        return sum(len(g.agents) for g in self.groups)


# Main table: groups
GRP_COL_SELECT = 0
GRP_COL_EXPAND = 1
GRP_COL_COLOR = 2
GRP_COL_GA_PARENT = 3
GRP_COL_FIXED_ELO = 4
GRP_COL_CLONE_ONLY = 5
GRP_COL_PLAY_IN_LEAGUE = 6
GRP_COL_NAME = 7
GRP_COL_AGENTS = 8
GRP_COL_SOURCE = 9
GRP_COL_ELO = 10
GRP_COL_ACTIONS = 11
GRP_NUM_COLUMNS = 12

# Column header tooltips for the flag checkboxes
GRP_TOOLTIP_GA_PARENT = (
    "Allow this group to be used as a parent in the genetic algorithm. "
    "When unchecked, agents participate in tournaments but are excluded from reproduction."
)
GRP_TOOLTIP_FIXED_ELO = (
    "ELO ratings for these agents are not updated after matches. "
    "Use for reference or benchmark agents."
)
GRP_TOOLTIP_CLONE_ONLY = (
    "When used as a GA parent, only clone (no mutation). "
    "Use for preserving exact copies of strong agents."
)
GRP_TOOLTIP_PLAY_IN_LEAGUE = (
    "Include these agents in league tournament tables. "
    "When unchecked, agents are excluded from league play (e.g. for evaluation-only sets)."
)


# Group detail (agents) columns
AGT_COL_GA_PARENT = 0
AGT_COL_NAME = 1
AGT_COL_ID = 2
AGT_COL_GENERATION = 3
AGT_COL_ELO = 4
AGT_COL_CHECKPOINT = 5
AGT_COL_ACTIONS = 6
AGT_NUM_COLUMNS = 7


class GroupDetailDialog(QtWidgets.QDialog):
    """Dialog to view/edit agents within a group (expand)."""

    def __init__(self, group: Group, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._group = group
        self.setWindowTitle(f"Group: {group.name}")
        self.setMinimumSize(900, 400)
        layout = QtWidgets.QVBoxLayout(self)
        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(AGT_NUM_COLUMNS)
        self._table.setHorizontalHeaderLabels(
            ["GA parent", "Name", "ID", "Generation", "ELO", "Checkpoint", "Actions"]
        )
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(self._table)
        layout.addWidget(QtWidgets.QLabel("Edit names in the Name column; use Delete to remove agents."))
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.accept)
        layout.addWidget(btns)
        self._refresh_table()
        self._table.resizeColumnsToContents()

    def _refresh_table(self) -> None:
        self._table.setRowCount(0)
        for row, agent in enumerate(self._group.agents):
            self._table.insertRow(row)
            self._fill_row(row, agent)

    def _fill_row(self, row: int, agent: Agent) -> None:
        cb = QtWidgets.QCheckBox()
        cb.setChecked(agent.can_use_as_ga_parent)
        cb.stateChanged.connect(
            lambda s, a=agent: setattr(a, "can_use_as_ga_parent", s == QtCore.Qt.CheckState.Checked)
        )
        cell = QtWidgets.QWidget()
        ll = QtWidgets.QHBoxLayout(cell)
        ll.setContentsMargins(4, 2, 4, 2)
        ll.addWidget(cb)
        self._table.setCellWidget(row, AGT_COL_GA_PARENT, cell)

        name_item = QtWidgets.QTableWidgetItem(agent.name)
        name_item.setFlags(name_item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, AGT_COL_NAME, name_item)
        self._table.setItem(row, AGT_COL_ID, QtWidgets.QTableWidgetItem(agent.id))
        self._table.setItem(row, AGT_COL_GENERATION, QtWidgets.QTableWidgetItem(str(agent.generation)))
        self._table.setItem(row, AGT_COL_ELO, QtWidgets.QTableWidgetItem(f"{agent.elo_global:.0f}"))
        self._table.setItem(row, AGT_COL_CHECKPOINT, QtWidgets.QTableWidgetItem(agent.checkpoint_path or "—"))

        btn = QtWidgets.QPushButton("Delete")
        btn.setMinimumHeight(22)
        btn.clicked.connect(lambda checked=False, r=row: self._delete_row(r))
        cell_actions = QtWidgets.QWidget()
        al = QtWidgets.QHBoxLayout(cell_actions)
        al.setContentsMargins(4, 2, 4, 2)
        al.addWidget(btn)
        self._table.setCellWidget(row, AGT_COL_ACTIONS, cell_actions)

    def _delete_row(self, row: int) -> None:
        if 0 <= row < len(self._group.agents):
            self._group.agents.pop(row)
            self._refresh_table()

    def accept(self) -> None:
        """Save edited names back to agents."""
        for row in range(self._table.rowCount()):
            if row < len(self._group.agents):
                name_item = self._table.item(row, AGT_COL_NAME)
                if name_item:
                    self._group.agents[row].name = name_item.text()
        super().accept()


class SexualReproductionSettingsDialog(QtWidgets.QDialog):
    """Dialog to configure parent selection and trait combination for sexual offspring."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        with_replacement: bool,
        fitness_weighted: bool,
        trait_combination: str,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sexual reproduction parameters")
        layout = QtWidgets.QFormLayout(self)
        # Row 1: Parent selection – weighting (Fitness-weighted / Uniform)
        self._combo_weighting = QtWidgets.QComboBox()
        self._combo_weighting.addItem("Fitness-weighted", True)
        self._combo_weighting.addItem("Uniform", False)
        self._combo_weighting.setCurrentIndex(0 if fitness_weighted else 1)
        self._combo_weighting.setToolTip(
            "How parents are chosen from the elite pool. Fitness-weighted: roulette selection by fitness "
            "(fitter agents more likely to be picked). Uniform: each elite agent has equal probability."
        )
        layout.addRow("Parent selection – weighting:", self._combo_weighting)
        # Row 2: With replacement – checkbox
        self._check_replacement = QtWidgets.QCheckBox()
        self._check_replacement.setChecked(with_replacement)
        self._check_replacement.setToolTip(
            "When checked, the same parent can be drawn more than once per offspring (or across offspring). "
            "When unchecked, the two parents for each sexual offspring are always distinct."
        )
        layout.addRow("Parent selection – with replacement:", self._check_replacement)
        # Row 3: Trait combination
        self._combo_trait = QtWidgets.QComboBox()
        self._combo_trait.addItem("Average", "average")
        self._combo_trait.addItem("Crossover", "crossover")
        idx = 0 if (trait_combination or "average").lower() == "average" else 1
        self._combo_trait.setCurrentIndex(idx)
        self._combo_trait.setToolTip(
            "How the two parents' traits are combined. Average: each trait is the mean of the two parents' values. "
            "Crossover: for each trait, randomly take the value from one parent or the other."
        )
        layout.addRow("Trait combination:", self._combo_trait)
        layout.addRow(QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            accepted=self.accept,
            rejected=self.reject,
        ))

    def get_with_replacement(self) -> bool:
        return self._check_replacement.isChecked()

    def get_fitness_weighted(self) -> bool:
        return self._combo_weighting.currentData() is True

    def get_trait_combination(self) -> str:
        return self._combo_trait.currentData() or "average"


class AugmentDialog(QtWidgets.QDialog):
    """Dialog to configure augment-from-selection (mutation and clone counts)."""

    def __init__(self, parent: Optional[QtWidgets.QWidget], num_selected: int) -> None:
        super().__init__(parent)
        self.setWindowTitle("Augment from selection")
        layout = QtWidgets.QFormLayout(self)
        layout.addRow(QtWidgets.QLabel(f"Base: {num_selected} selected agent(s)."))
        self._spin_mutate = QtWidgets.QSpinBox()
        self._spin_mutate.setRange(0, 999)
        self._spin_mutate.setValue(4)
        layout.addRow("Mutated children to add:", self._spin_mutate)
        self._spin_mut_prob = QtWidgets.QDoubleSpinBox()
        self._spin_mut_prob.setRange(0.0, 1.0)
        self._spin_mut_prob.setValue(0.5)
        self._spin_mut_prob.setSingleStep(0.1)
        layout.addRow("Mutation prob:", self._spin_mut_prob)
        self._spin_mut_std = QtWidgets.QDoubleSpinBox()
        self._spin_mut_std.setRange(0.0, 1.0)
        self._spin_mut_std.setValue(0.1)
        self._spin_mut_std.setSingleStep(0.05)
        layout.addRow("Mutation std:", self._spin_mut_std)
        self._spin_clone = QtWidgets.QSpinBox()
        self._spin_clone.setRange(0, 99)
        self._spin_clone.setValue(0)
        layout.addRow("Clones per selected agent:", self._spin_clone)
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def mutate_count(self) -> int:
        return self._spin_mutate.value()

    def clone_count(self) -> int:
        return self._spin_clone.value()

    def mutation_prob(self) -> float:
        return self._spin_mut_prob.value()

    def mutation_std(self) -> float:
        return self._spin_mut_std.value()


def _population_to_single_group(pop: Population, generation_index: int) -> Group:
    """Convert a flat Population from the league backend into one Group for the UI."""
    agents = list(pop.agents.values())
    return Group(
        id="league_0",
        name=f"League (gen {generation_index})",
        agents=agents,
        color=0x4A90D9,
    )


class LeagueRunWorker(QtCore.QThread):
    """
    Runs run_league_generations() in a background thread.
    Emits generation_done(gen_idx, population, summary) after each generation,
    and finished(cancelled, paused) when the run ends.
    """

    generation_done = QtCore.Signal(int, object, object)  # gen_idx, Population, summary dict
    match_done = QtCore.Signal(int, int, object, object)  # gen_idx, round_idx, Population, per-round summary
    finished_run = QtCore.Signal(bool, bool)  # cancelled, paused

    def __init__(
        self,
        pop: Population,
        cfg: LeagueConfig,
        num_generations: int,
        project_path: str,
        control: LeagueRunControl,
        rng_seed: Optional[int] = None,
        device: Optional[str] = None,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._pop = pop
        self._cfg = cfg
        self._num_generations = num_generations
        self._project_path = project_path
        self._control = control
        self._rng_seed = rng_seed
        self._device = device
        self._pause_requested = False

    def request_pause(self) -> None:
        self._pause_requested = True

    def run(self) -> None:
        rng = random.Random(self._rng_seed)
        log_path = get_log_path(self._project_path)
        checkpoint_base_dir = str(get_checkpoint_base_dir(self._project_path))
        cancelled = False
        paused = False

        class _PauseRequested(Exception):
            """Internal sentinel to stop the run due to a pause request."""

        try:
            def _on_round(gen_idx: int, round_idx: int, round_summary: Dict[str, float]) -> None:
                # Emit per-match updates so the GUI can update ELO / RL blocks at match resolution.
                self.match_done.emit(gen_idx, round_idx, self._pop, round_summary)
                # If a pause was requested, stop after this round (match) instead of waiting
                # for the end of the whole generation.
                if self._pause_requested:
                    raise _PauseRequested()

            gen_iter = run_league_generations(
                self._pop,
                self._cfg,
                num_generations=self._num_generations,
                rng=rng,
                control=self._control,
                checkpoint_base_dir=checkpoint_base_dir,
                device=self._device,
                log_path=str(log_path),
                on_round=_on_round,
            )
            for new_pop, summary, gen_idx in gen_iter:
                self._pop = new_pop
                self.generation_done.emit(gen_idx, new_pop, summary)
                if self._control.cancel_requested.is_set():
                    cancelled = True
                    break
                if self._pause_requested:
                    paused = True
                    break
        except _PauseRequested:
            paused = True
        except Exception:
            cancelled = True
            raise
        finally:
            self.finished_run.emit(cancelled, paused)

    def run_sync(self) -> None:
        """Run in current thread (for tests). Does not emit signals in a queued way."""
        rng = random.Random(self._rng_seed)
        log_path = get_log_path(self._project_path)
        checkpoint_base_dir = str(get_checkpoint_base_dir(self._project_path))
        class _PauseRequested(Exception):
            """Internal sentinel to stop the run due to a pause request (sync mode)."""

        def _on_round(gen_idx: int, round_idx: int, round_summary: Dict[str, float]) -> None:
            self.match_done.emit(gen_idx, round_idx, self._pop, round_summary)
            if self._pause_requested:
                raise _PauseRequested()

        cancelled = False
        paused = False
        try:
            gen_iter = run_league_generations(
                self._pop,
                self._cfg,
                num_generations=self._num_generations,
                rng=rng,
                control=self._control,
                checkpoint_base_dir=checkpoint_base_dir,
                device=self._device,
                log_path=str(log_path),
                on_round=_on_round,
            )
            for new_pop, summary, gen_idx in gen_iter:
                self._pop = new_pop
                self.generation_done.emit(gen_idx, new_pop, summary)
                if self._control.cancel_requested.is_set():
                    cancelled = True
                    break
                if self._pause_requested:
                    paused = True
                    break
        except _PauseRequested:
            paused = True
        finally:
            self.finished_run.emit(cancelled, paused)


class RunSectionWidget(QtWidgets.QWidget):
    """Run controls (Start, Pause/Resume, Cancel), compute metrics, run log path and Save/Load. Placed in Dashboard tab."""

    start_clicked = QtCore.Signal()
    pause_clicked = QtCore.Signal()
    cancel_clicked = QtCore.Signal()
    run_log_loaded = QtCore.Signal(str)  # log_id when a run log file is loaded
    load_population_clicked = QtCore.Signal()

    def __init__(
        self,
        state: LeagueTabState,
        parent: Optional[QtWidgets.QWidget] = None,
        run_log_manager: Optional[RunLogManager] = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._run_log_manager = run_log_manager
        self._auto_save_dir: Optional[str] = None
        self._is_paused: bool = False

        layout = QtWidgets.QHBoxLayout(self)

        # --- Run box: Start / Pause/Resume / Cancel + compute ---
        run_group = QtWidgets.QGroupBox("Run")
        run_layout = QtWidgets.QVBoxLayout(run_group)
        buttons_row = QtWidgets.QHBoxLayout()
        self._btn_start = QtWidgets.QPushButton("Start")
        self._btn_pause = QtWidgets.QPushButton("Pause")
        self._btn_cancel = QtWidgets.QPushButton("Wipe run")
        self._btn_pause.setEnabled(False)
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.setVisible(False)
        self._btn_start.clicked.connect(self._on_start_clicked)
        self._btn_pause.clicked.connect(self._on_pause_clicked)
        self._btn_cancel.clicked.connect(self._on_cancel_clicked)
        buttons_row.addWidget(self._btn_start)
        buttons_row.addWidget(self._btn_pause)
        buttons_row.addWidget(self._btn_cancel)
        buttons_row.addStretch(1)
        run_layout.addLayout(buttons_row)

        # One-line status: generation X of Y, elapsed, ETA (hidden; Compute block shows timing)
        self._label_status = QtWidgets.QLabel("Status: —")
        self._label_status.setVisible(False)

        # Inline compute metrics (time used, ETA, avg/gen)
        self._compute_block = ComputeBlockWidget()
        run_layout.addWidget(self._compute_block)

        layout.addWidget(run_group, 2)

        # --- File box: log location + Save / Load + Load population ---
        file_group = QtWidgets.QGroupBox("File")
        file_layout = QtWidgets.QVBoxLayout(file_group)

        # Header row: Load League Project button + project status label
        header_row = QtWidgets.QHBoxLayout()
        self._btn_load_population = QtWidgets.QPushButton("Load League Project")
        self._btn_load_population.clicked.connect(self._on_load_population)
        header_row.addWidget(self._btn_load_population)
        self._label_project_name = QtWidgets.QLabel("No Project Loaded")
        header_row.addWidget(self._label_project_name)
        header_row.addStretch(1)
        file_layout.addLayout(header_row)

        file_layout.addWidget(QtWidgets.QLabel("Log file name (auto-saved per gen in project logs folder):"))

        log_path_row = QtWidgets.QHBoxLayout()
        self._edit_log_filename = QtWidgets.QLineEdit()
        self._edit_log_filename.setPlaceholderText("Log file name (e.g. league_run.jsonl)")
        self._edit_log_filename.setClearButtonEnabled(True)
        log_path_row.addWidget(self._edit_log_filename)
        file_layout.addLayout(log_path_row)
        self._edit_log_filename.textChanged.connect(self._on_log_path_changed)

        run_log_btn_row = QtWidgets.QHBoxLayout()
        self._btn_save_run_log = QtWidgets.QPushButton("Save run log")
        self._btn_save_run_log.clicked.connect(self._on_save_run_log)
        run_log_btn_row.addWidget(self._btn_save_run_log)
        self._btn_browse_log_dir = QtWidgets.QPushButton("Browse Logs")
        self._btn_browse_log_dir.clicked.connect(self._on_browse_log_dir)
        run_log_btn_row.addWidget(self._btn_browse_log_dir)
        run_log_btn_row.addStretch(1)
        file_layout.addLayout(run_log_btn_row)

        layout.addWidget(file_group, 3)

        # --- Export box: placeholder, aligned with Run and File on the right ---
        self._export_group = ExportBlockWidget()
        layout.addWidget(self._export_group, 1)

        self.update_run_log_buttons()

    def set_run_output(
        self,
        project_path: Optional[str],
        population: Optional[object],
        generation_index: int,
        fitness_config: Optional[object] = None,
    ) -> None:
        """Set the last run output for the Export block (called by MainWindow after each generation)."""
        self._export_group.set_run_output(
            project_path, population, generation_index, fitness_config
        )

    def clear_run_output(self) -> None:
        """Clear run output in the Export block (e.g. when a new run starts)."""
        self._export_group.clear_run_output()

    def _on_start_clicked(self) -> None:
        self.start_clicked.emit()

    def _on_pause_clicked(self) -> None:
        # When running: interpret as Pause. When already paused: interpret as Resume.
        if self._is_paused:
            self.start_clicked.emit()
        else:
            self.pause_clicked.emit()

    def _on_cancel_clicked(self) -> None:
        self.cancel_clicked.emit()

    def _on_browse_log_dir(self) -> None:
        """Browse existing run logs across all projects (from the configured Projects folder)."""
        if self._run_log_manager is None:
            return
        base = get_projects_folder()
        if not base:
            QtWidgets.QMessageBox.warning(
                self,
                "Projects folder not set",
                "No Projects folder is configured.\n\n"
                "Go to the Settings tab, set a Projects folder, then try again.",
            )
            return
        base_path = Path(base)
        if not base_path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Projects folder unavailable",
                "The Projects folder does not exist:\n"
                f"{base}\n\n"
                "Please go to the Settings tab, choose a valid Projects folder,\n"
                "and try again.",
            )
            return

        # Build list of logs: all *.jsonl under each project's logs/ subfolder.
        logs: list[tuple[str, Path]] = []
        for proj_name in _list_existing_projects(base_path):
            proj_dir = base_path / proj_name
            logs_dir = proj_dir / "logs"
            if not logs_dir.is_dir():
                continue
            for log_path in logs_dir.glob("*.jsonl"):
                logs.append((f"{proj_name} / {log_path.name}", log_path))

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Browse Logs")
        dlg.setMinimumSize(480, 320)
        vbox = QtWidgets.QVBoxLayout(dlg)
        label = QtWidgets.QLabel(
            "Select a run log to load into the Dashboard charts.\n"
            "Logs are discovered under the Projects folder (project_name/logs/*.jsonl)."
        )
        label.setWordWrap(True)
        vbox.addWidget(label)
        list_widget = QtWidgets.QListWidget()
        list_widget.setMinimumHeight(180)
        vbox.addWidget(list_widget)
        if not logs:
            item = QtWidgets.QListWidgetItem(
                "No logs found under the current Projects folder.\n"
                "Run a league and ensure logs are written to each project's 'logs' directory."
            )
            item.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            list_widget.addItem(item)
        else:
            for display, path in logs:
                item = QtWidgets.QListWidgetItem(display)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, str(path))
                list_widget.addItem(item)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_open = QtWidgets.QPushButton("Load selected")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_row.addWidget(btn_open)
        btn_row.addWidget(btn_cancel)
        vbox.addLayout(btn_row)

        def on_selection_changed() -> None:
            item = list_widget.currentItem()
            btn_open.setEnabled(bool(item and item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled))

        list_widget.itemSelectionChanged.connect(on_selection_changed)
        on_selection_changed()

        def on_open() -> None:
            item = list_widget.currentItem()
            if not item or not (item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled):
                return
            path_str = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if not path_str:
                return
            try:
                log_id = self._run_log_manager.load_from_path(path_str)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Load run log failed", str(e))
                return
            else:
                self.run_log_loaded.emit(log_id)
                dlg.accept()

        btn_open.clicked.connect(on_open)
        btn_cancel.clicked.connect(dlg.reject)
        list_widget.itemDoubleClicked.connect(lambda _item: on_open())
        dlg.exec()

    def _on_log_path_changed(self) -> None:
        if self._run_log_manager is None:
            return
        name_text = self._edit_log_filename.text().strip() or None
        self._run_log_manager.set_auto_save(self._auto_save_dir, name_text)

    def _on_save_run_log(self) -> None:
        """
        Save the current run log to the project logs folder using the name
        from the filename field above (auto-appending .jsonl if missing).
        """
        if self._run_log_manager is None or not self._run_log_manager.has_current_data():
            return
        if not self._auto_save_dir:
            QtWidgets.QMessageBox.warning(
                self,
                "No project loaded",
                "Cannot save run log because no League project is loaded.\n\n"
                "Load a League project first so the logs folder is known.",
            )
            return
        name_text = self._edit_log_filename.text().strip() or "league_run.jsonl"
        if not name_text.lower().endswith(".jsonl"):
            name_text += ".jsonl"
        path = Path(self._auto_save_dir) / name_text
        self._run_log_manager.save_to_path(str(path))

    def _on_load_population(self) -> None:
        self.load_population_clicked.emit()

    def update_run_log_buttons(self) -> None:
        """Enable Save when there is current run log data; call after generation_done or clear."""
        if self._run_log_manager is not None:
            self._btn_save_run_log.setEnabled(self._run_log_manager.has_current_data())
        else:
            self._btn_save_run_log.setEnabled(False)

    def update_run_status(
        self,
        gen_index: int,
        total_generations: int,
        elapsed_seconds: float,
        eta_seconds: Optional[float],
    ) -> None:
        """
        Update the status line (Generation X of Y, Elapsed, ETA).
        Called only when a generation completes (or when run finishes to clear).
        gen_index: 0-based index of the generation just completed (-1 when not running).
        total_generations: from League Parameters (get_num_generations()).
        elapsed_seconds: time since run start.
        eta_seconds: estimated seconds remaining (None for first gens or when not running).
        """
        # Status text is currently hidden from the UI; Compute block shows timing instead.
        if gen_index < 0 or total_generations <= 0:
            self._label_status.setText("Status: —")
            return
        x = gen_index + 1  # Generation X of Y (1-based display)
        y = total_generations
        elapsed_str = _format_duration(elapsed_seconds)
        if eta_seconds is None:
            eta_str = "calculating…" if gen_index < 1 else "—"
        else:
            eta_str = _format_duration(eta_seconds)
        self._label_status.setText(
            f"Generation {x} of {y}  |  Elapsed: {elapsed_str}  |  ETA: {eta_str}"
        )

    def set_buttons_running(self, running: bool, paused: bool = False) -> None:
        """
        Update Start / Pause / Cancel buttons for running / paused / idle states.

        - running=True: Start disabled, Pause enabled ("Pause"), Cancel hidden.
        - running=False, paused=True: Start disabled, Pause enabled ("Resume"), Cancel visible+enabled.
        - running=False, paused=False: idle; Start enabled if project loaded, Pause/Cancel disabled+hidden.
        """
        self._is_paused = paused
        if running:
            self._btn_start.setEnabled(False)
            self._btn_pause.setEnabled(True)
            self._btn_pause.setText("Pause")
            self._btn_cancel.setEnabled(False)
            self._btn_cancel.setVisible(False)
            return
        # Not running: paused or idle
        if paused:
            self._btn_start.setEnabled(False)
            self._btn_pause.setEnabled(True)
            self._btn_pause.setText("Resume")
            self._btn_cancel.setEnabled(True)
            self._btn_cancel.setVisible(True)
        else:
            self._btn_pause.setEnabled(False)
            self._btn_pause.setText("Pause")
            self._btn_cancel.setEnabled(False)
            self._btn_cancel.setVisible(False)
            self._btn_start.setEnabled(bool(self._state.project_path))

    def update_start_enabled(self) -> None:
        """Enable Start only when a project is loaded and not running (call when project or run state changes)."""
        if not self._btn_pause.isEnabled() and not self._is_paused:
            self._btn_start.setEnabled(bool(self._state.project_path))

    def showEvent(self, event: QtCore.QEvent) -> None:
        super().showEvent(event)
        self.update_metrics()
        self.update_start_enabled()
        self.update_run_log_buttons()

    def update_metrics(self) -> None:
        """ELO metrics now live in the Dashboard ELO block; nothing to do here."""
        return

    def configure_auto_save_for_project(self, project_path: Optional[str]) -> None:
        """Set the auto-save directory based on the current League project path."""
        if not project_path:
            self._auto_save_dir = None
            self._label_project_name.setText("No Project Loaded")
            if self._run_log_manager is not None:
                self._run_log_manager.set_auto_save(None, self._edit_log_filename.text().strip() or None)
            return
        logs_dir = Path(project_path) / "logs"
        self._auto_save_dir = str(logs_dir)
        self._label_project_name.setText(f"Project: {Path(project_path).name}")
        if self._run_log_manager is not None:
            filename = self._edit_log_filename.text().strip() or "league_run.jsonl"
            self._run_log_manager.set_auto_save(self._auto_save_dir, filename)

    def get_log_filename(self) -> str:
        return self._edit_log_filename.text().strip()

    def update_compute_metrics(
        self,
        elapsed_seconds: Optional[float],
        eta_seconds: Optional[float],
        avg_seconds_per_gen: Optional[float],
    ) -> None:
        self._compute_block.update_metrics(
            elapsed_seconds,
            eta_seconds,
            avg_seconds_per_gen,
        )

    def clear_compute_metrics(self) -> None:
        self._compute_block.clear_metrics()


class LeagueTabWidget(QtWidgets.QWidget):
    """League Parameters tab: groups table, config sections, export."""

    project_path_changed = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._state = LeagueTabState()
        self._league_params_dirty = False
        self._rng = random.Random()
        self._update_league_content_size: Optional[callable] = None
        self._setup_ui()
        self._refresh_table()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if self._update_league_content_size is not None:
            self._update_league_content_size()

    def _setup_ui(self) -> None:
        # Layout: Project, Population (table scrolls internally when many rows), then row1 = Tournament|Next Gen, row2 = Fitness|Reproduction
        _m = 8  # layout vertical margins
        PROJECT_HEIGHT = 96  # enough for project name + File button without cropping
        POPULATION_HEIGHT = 460  # enough for pie, insights, tools, and table fully visible
        ARROW_HEIGHT = 14
        FLOW_ROW_SPACING = 12
        FLOW_BOX_HEIGHT_ROW1 = 220  # Tournament (reduced height)
        FLOW_BOX_HEIGHT_ROW2 = 494  # Fitness, Reproduction (taller; ~30% increase from 380)
        FLOW_GRAPH_MIN_HEIGHT = 260  # Min height for Fitness line chart and Reproduction mutation-dist graph (identical)
        CONTENT_HEIGHT_1080P = (
            _m + PROJECT_HEIGHT + POPULATION_HEIGHT + ARROW_HEIGHT
            + FLOW_BOX_HEIGHT_ROW1 + FLOW_BOX_HEIGHT_ROW2 + FLOW_ROW_SPACING
        )  # ~1278

        self._scroll = QtWidgets.QScrollArea()
        scroll = self._scroll
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        self._league_content = QtWidgets.QWidget()
        content = self._league_content
        # Option C: content width bound to viewport in resize callback (no fixed 1920)
        content.setMinimumHeight(CONTENT_HEIGHT_1080P)
        layout = QtWidgets.QVBoxLayout(content)
        layout.setSpacing(0)
        layout.setContentsMargins(4, 4, 4, 4)

        proj_group = QtWidgets.QGroupBox("Project")
        proj_layout = QtWidgets.QVBoxLayout(proj_group)
        self._label_project = QtWidgets.QLabel("No Project Loaded")
        self._label_project.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._label_project.setMinimumWidth(200)
        self._label_project.setMinimumHeight(28)
        self._label_project.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #c0c0c0; padding: 8px;"
        )
        # No-project buttons (visible when no project loaded), underneath the label with spacing
        self._btns_no_project = QtWidgets.QWidget()
        no_proj_layout = QtWidgets.QHBoxLayout(self._btns_no_project)
        no_proj_layout.setContentsMargins(0, 0, 0, 0)
        btn_new = QtWidgets.QPushButton("New Project")
        btn_new.clicked.connect(self._on_new_project)
        btn_open = QtWidgets.QPushButton("Open Project")
        btn_open.clicked.connect(self._on_open_project)
        no_proj_layout.addWidget(btn_new)
        no_proj_layout.addSpacing(12)
        no_proj_layout.addWidget(btn_open)
        # File button (visible only when project loaded)
        self._btn_file = QtWidgets.QPushButton("File")
        self._btn_file.setFixedWidth(100)
        self._btn_file.setMinimumHeight(34)
        file_menu = QtWidgets.QMenu(self)
        act_new = file_menu.addAction("New Project")
        act_new.triggered.connect(self._on_new_project)
        act_open = file_menu.addAction("Open Project")
        act_open.triggered.connect(self._on_open_project)
        file_menu.addSeparator()
        self._act_save = file_menu.addAction("Save")
        self._act_save.triggered.connect(self._on_save_project)
        self._act_save_as = file_menu.addAction("Save As")
        self._act_save_as.triggered.connect(self._on_save_project_as)
        file_menu.addSeparator()
        self._act_export = file_menu.addAction("Export JSON")
        self._act_export.triggered.connect(self._on_export_project_json)
        act_import = file_menu.addAction("Import JSON")
        act_import.triggered.connect(self._on_import_project_json)
        self._btn_file.setMenu(file_menu)
        self._btn_file.setVisible(False)
        # Unsaved warning indicator (visible only when league params are dirty)
        self._btn_unsaved_warning = QtWidgets.QToolButton()
        self._btn_unsaved_warning.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning)
        )
        self._btn_unsaved_warning.setToolTip("Unsaved changes in league parameters")
        self._btn_unsaved_warning.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._btn_unsaved_warning.setVisible(False)
        # Save button (visible when project loaded)
        self._btn_save = QtWidgets.QPushButton("Save")
        self._btn_save.setFixedWidth(80)
        self._btn_save.setMinimumHeight(34)
        self._btn_save.setToolTip("Save project (league config and UI state)")
        self._btn_save.clicked.connect(self._on_save_project)
        self._btn_save.setVisible(False)
        # Row 1: project name / "No Project Loaded" label + warning (when dirty) + Save + File (when project loaded)
        proj_row1 = QtWidgets.QHBoxLayout()
        # Alignment is updated dynamically in _update_project_label() depending on whether
        # a project is loaded (left-aligned) or not (centered).
        proj_row1.addWidget(self._label_project, 1)
        proj_row1.addWidget(self._btn_unsaved_warning, 0)
        proj_row1.addWidget(self._btn_save, 0)
        proj_row1.addWidget(self._btn_file, 0)
        proj_layout.addLayout(proj_row1)
        # Row 2: New Project / Open Project (only when no project loaded), centered under the label
        self._no_project_row = QtWidgets.QWidget()
        proj_row2 = QtWidgets.QHBoxLayout(self._no_project_row)
        proj_row2.setContentsMargins(0, 4, 0, 0)
        proj_row2.addStretch(1)
        proj_row2.addWidget(self._btns_no_project, 0)
        proj_row2.addStretch(1)
        proj_layout.addWidget(self._no_project_row)
        proj_group.setFixedHeight(PROJECT_HEIGHT)
        self._project_group = proj_group
        self._project_height_when_loaded = PROJECT_HEIGHT

        # No-project view: centered project box (~1/3 viewport height)
        self._no_project_center = QtWidgets.QWidget()
        no_proj_center_layout = QtWidgets.QVBoxLayout(self._no_project_center)
        no_proj_center_layout.setContentsMargins(0, 0, 0, 0)
        no_proj_center_layout.addWidget(proj_group)
        self._no_project_view = QtWidgets.QWidget()
        no_proj_view_layout = QtWidgets.QVBoxLayout(self._no_project_view)
        no_proj_view_layout.setContentsMargins(0, 0, 0, 0)
        no_proj_view_layout.addStretch(1)
        no_proj_view_layout.addWidget(self._no_project_center)
        no_proj_view_layout.addStretch(1)
        # Project-loaded view: project bar at top, then content
        self._project_view = QtWidgets.QWidget()
        project_view_layout = QtWidgets.QVBoxLayout(self._project_view)
        project_view_layout.setContentsMargins(0, 0, 0, 0)
        project_view_layout.setSpacing(0)
        # proj_group will be inserted at 0 when switching to project view
        self._project_view_layout = project_view_layout

        self._content_container = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(self._content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout = content_layout

        pop_group = QtWidgets.QGroupBox("Population")
        pop_main = QtWidgets.QVBoxLayout(pop_group)

        # Row: pie + metrics (LEFT) | tools + table (RIGHT, tools over table left edge)
        pop_middle = QtWidgets.QHBoxLayout()
        pie_container = QtWidgets.QWidget()
        pie_layout = QtWidgets.QVBoxLayout(pie_container)
        pie_layout.setContentsMargins(0, 0, 0, 0)
        self._pie_widget = PopulationPieWidget()
        pie_layout.addWidget(self._pie_widget)
        # Insights table under pie: GA agents, Fixed ELO, Clone only, Play in league, Total
        self._pie_insights_table = QtWidgets.QTableWidget()
        self._pie_insights_table.setColumnCount(2)
        self._pie_insights_table.setHorizontalHeaderLabels(["Metric", "Count"])
        self._pie_insights_table.setRowCount(5)
        self._pie_insights_table.verticalHeader().setVisible(False)
        self._pie_insights_table.setMinimumHeight(120)
        self._pie_insights_table.verticalHeader().setDefaultSectionSize(28)
        self._pie_insights_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        for row, label in enumerate(["GA agents", "Fixed ELO", "Clone only", "Play in league", "Total"]):
            self._pie_insights_table.setItem(row, 0, QtWidgets.QTableWidgetItem(label))
            self._pie_insights_table.setItem(row, 1, QtWidgets.QTableWidgetItem("0"))
        self._pie_insights_table.setColumnWidth(0, 100)
        pie_layout.addWidget(self._pie_insights_table)
        filter_row = QtWidgets.QHBoxLayout()
        lbl_group_by = QtWidgets.QLabel("Group by:")
        lbl_group_by.setToolTip("How to group segments in the pie chart: by group name, GA status, or Play in league.")
        filter_row.addWidget(lbl_group_by)
        self._combo_pie_group_by = QtWidgets.QComboBox()
        self._combo_pie_group_by.addItems(["Group name", "GA status", "Play in league"])
        self._combo_pie_group_by.setItemData(0, "Show distribution by group", QtCore.Qt.ItemDataRole.ToolTipRole)
        self._combo_pie_group_by.setItemData(1, "Show GA-eligible vs Reference", QtCore.Qt.ItemDataRole.ToolTipRole)
        self._combo_pie_group_by.setItemData(2, "Show Play-in-league vs Not", QtCore.Qt.ItemDataRole.ToolTipRole)
        self._combo_pie_group_by.currentTextChanged.connect(self._update_pie_chart)
        filter_row.addWidget(self._combo_pie_group_by)
        filter_row.addStretch()
        pie_layout.addLayout(filter_row)
        pop_middle.addWidget(pie_container)

        # Right column: tools over table (aligned with table left edge)
        right_col = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)
        tools_row = QtWidgets.QHBoxLayout()
        lbl_count = QtWidgets.QLabel("Count:")
        lbl_count.setToolTip("Number of random agents to add when clicking Add random.")
        tools_row.addWidget(lbl_count)
        self._spin_add_random = QtWidgets.QSpinBox()
        self._spin_add_random.setRange(1, 999)
        self._spin_add_random.setValue(4)
        self._spin_add_random.setMinimumWidth(52)
        tools_row.addWidget(self._spin_add_random)
        btn_add_random = QtWidgets.QPushButton("Add random")
        btn_add_random.clicked.connect(self._on_add_random)
        tools_row.addWidget(btn_add_random)
        self._btn_import = QtWidgets.QPushButton("Import")
        import_menu = QtWidgets.QMenu(self)
        act_import_file = import_menu.addAction("From file...")
        act_import_file.triggered.connect(self._on_import)
        act_import_agents = import_menu.addAction("From agents folder")
        act_import_agents.triggered.connect(self._on_import_from_agents_folder)
        act_import_agents.setToolTip("Load all agents from this project's agents/ folder.")
        act_import_hof = import_menu.addAction("From Hall of Fame")
        act_import_hof.triggered.connect(self._on_import_from_hof)
        act_import_hof.setToolTip("Load all agents from this project's agents/Hall of Fame/ folder.")
        self._btn_import.setMenu(import_menu)
        tools_row.addWidget(self._btn_import)
        btn_augment = QtWidgets.QPushButton("Augment from selection")
        btn_augment.clicked.connect(self._on_augment_from_selection)
        tools_row.addWidget(btn_augment)
        btn_clear_selected = QtWidgets.QPushButton("Clear selected")
        btn_clear_selected.clicked.connect(self._on_clear_selected)
        tools_row.addWidget(btn_clear_selected)
        btn_clear = QtWidgets.QPushButton("Clear")
        btn_clear.clicked.connect(self._on_clear)
        tools_row.addStretch(1)
        right_layout.addLayout(tools_row)

        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(GRP_NUM_COLUMNS)
        self._table.setHorizontalHeaderLabels(
            [
                "Select",
                "Expand",
                "Color",
                "GA parent",
                "Fixed ELO",
                "Clone only",
                "Play in league",
                "Group name",
                "# agents",
                "Source",
                "ELO (min/mean/max)",
                "Actions",
            ]
        )
        model = self._table.model()
        model.setHeaderData(GRP_COL_SELECT, QtCore.Qt.Orientation.Horizontal, "Select for Augment / Clear selected", QtCore.Qt.ItemDataRole.ToolTipRole)
        model.setHeaderData(GRP_COL_COLOR, QtCore.Qt.Orientation.Horizontal, "Color used in pie chart", QtCore.Qt.ItemDataRole.ToolTipRole)
        model.setHeaderData(GRP_COL_GA_PARENT, QtCore.Qt.Orientation.Horizontal, GRP_TOOLTIP_GA_PARENT, QtCore.Qt.ItemDataRole.ToolTipRole)
        model.setHeaderData(GRP_COL_FIXED_ELO, QtCore.Qt.Orientation.Horizontal, GRP_TOOLTIP_FIXED_ELO, QtCore.Qt.ItemDataRole.ToolTipRole)
        model.setHeaderData(GRP_COL_CLONE_ONLY, QtCore.Qt.Orientation.Horizontal, GRP_TOOLTIP_CLONE_ONLY, QtCore.Qt.ItemDataRole.ToolTipRole)
        model.setHeaderData(GRP_COL_PLAY_IN_LEAGUE, QtCore.Qt.Orientation.Horizontal, GRP_TOOLTIP_PLAY_IN_LEAGUE, QtCore.Qt.ItemDataRole.ToolTipRole)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._table.horizontalHeader().setMinimumSectionSize(24)
        self._table.verticalHeader().setDefaultSectionSize(34)
        header = self._table.horizontalHeader()
        for col in range(GRP_NUM_COLUMNS):
            if col == GRP_COL_NAME:
                header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.Stretch)
            else:
                header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        # Table in scroll area: when few rows, no internal scroll; when many rows, scroll inside population box
        self._table_scroll = QtWidgets.QScrollArea()
        self._table_scroll.setWidgetResizable(True)
        self._table_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self._table_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self._table_scroll.setWidget(self._table)
        self._table_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        right_layout.addWidget(self._table_scroll, stretch=1)
        pop_middle.addWidget(right_col, stretch=1)
        pop_main.addLayout(pop_middle)
        pop_group.setFixedHeight(POPULATION_HEIGHT)
        content_layout.addWidget(pop_group, 0)

        def add_arrow() -> None:
            arr = QtWidgets.QLabel("▼")
            arr.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            arr.setFixedHeight(ARROW_HEIGHT)
            arr.setStyleSheet("font-size: 14px; font-weight: bold; color: #808080;")
            content_layout.addWidget(arr, 0)

        add_arrow()

        # Flow blocks: row1 = Tournament | Next Gen, row2 = Fitness (left) | Reproduction (right)
        flow_vertical = QtWidgets.QVBoxLayout()
        flow_vertical.setSpacing(FLOW_ROW_SPACING)
        flow_vertical.setContentsMargins(0, 0, 0, 0)
        flow_row1 = QtWidgets.QHBoxLayout()
        flow_row1.setSpacing(12)
        flow_row2 = QtWidgets.QHBoxLayout()
        flow_row2.setSpacing(12)

        # Tournament block with categories: Core (no checkbox) | ELO tuning | PPO
        tour_group = QtWidgets.QGroupBox("Tournament")
        tour_layout = QtWidgets.QVBoxLayout(tour_group)

        # Core: always visible, no checkbox. Pairs: Players|Style, Deals|Matches
        tour_core = QtWidgets.QFrame()
        tour_core.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        core_layout = QtWidgets.QFormLayout(tour_core)
        core_layout.setSpacing(4)
        self._combo_player_count = QtWidgets.QComboBox()
        player_model = QtGui.QStandardItemModel()
        for n in [3, 4, 5]:
            item = QtGui.QStandardItem(f"{n} Players (FFT Rules)")
            item.setData(n, QtCore.Qt.ItemDataRole.UserRole)
            player_model.appendRow(item)
        self._combo_player_count.setModel(player_model)
        self._combo_player_count.setCurrentIndex(1)
        self._combo_player_count.setMinimumWidth(140)
        self._combo_player_count.currentIndexChanged.connect(self._update_tournament_insights)
        self._combo_player_count.currentIndexChanged.connect(self._update_sideline_warning)
        self._combo_league_style = QtWidgets.QComboBox()
        self._combo_league_style.addItems(["ELO-based", "Random"])
        self._combo_league_style.setMinimumWidth(130)
        self._combo_league_style.setItemData(0, "Match agents of similar ELO strength.", QtCore.Qt.ItemDataRole.ToolTipRole)
        self._combo_league_style.setItemData(1, "Shuffle agents randomly.", QtCore.Qt.ItemDataRole.ToolTipRole)
        core_row1 = QtWidgets.QHBoxLayout()
        lbl_rules = QtWidgets.QLabel("Rules:")
        lbl_rules.setToolTip("Game rules: 3, 4, or 5 players per table (FFT rules).")
        core_row1.addWidget(lbl_rules)
        core_row1.addWidget(self._combo_player_count)
        # Warning icon: fixed-size frame (always reserves space); tooltip on frame so it shows on hover
        self._sideline_warning_frame = QtWidgets.QFrame()
        self._sideline_warning_frame.setFixedSize(26, 26)
        sideline_icon_layout = QtWidgets.QHBoxLayout(self._sideline_warning_frame)
        sideline_icon_layout.setContentsMargins(0, 0, 0, 0)
        sideline_icon_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._sideline_warning_icon = QtWidgets.QLabel("")
        self._sideline_warning_icon.setStyleSheet("color: #c9a227; font-size: 16px;")
        sideline_icon_layout.addWidget(self._sideline_warning_icon)
        core_row1.addWidget(self._sideline_warning_frame)
        core_row1.addSpacing(12)
        lbl_matchmaking = QtWidgets.QLabel("Matchmaking:")
        lbl_matchmaking.setToolTip("ELO-based: pair similar-strength agents. Random: random table assignment.")
        core_row1.addWidget(lbl_matchmaking)
        core_row1.addWidget(self._combo_league_style)
        core_row1.addStretch()
        core_layout.addRow(core_row1)
        self._spin_deals = QtWidgets.QSpinBox()
        self._spin_deals.setRange(1, 99)
        self._spin_deals.setValue(5)
        self._spin_deals.setMinimumWidth(52)
        self._spin_deals.valueChanged.connect(self._update_tournament_insights)
        self._spin_matches = QtWidgets.QSpinBox()
        self._spin_matches.setRange(1, 999)
        self._spin_matches.setValue(3)
        self._spin_matches.setMinimumWidth(56)
        self._spin_matches.valueChanged.connect(self._update_tournament_insights)
        core_row2 = QtWidgets.QHBoxLayout()
        lbl_deals = QtWidgets.QLabel("Deals/match:")
        lbl_deals.setToolTip("Number of deals played in each match. More deals give more stable ELO updates.")
        core_row2.addWidget(lbl_deals)
        core_row2.addWidget(self._spin_deals)
        core_row2.addSpacing(12)
        lbl_matches = QtWidgets.QLabel("Matches/gen:")
        lbl_matches.setToolTip("Number of tournament rounds per generation. More rounds refine ELO rankings.")
        core_row2.addWidget(lbl_matches)
        core_row2.addWidget(self._spin_matches)
        core_row2.addStretch()
        core_layout.addRow(core_row2)
        self._combo_player_count.setToolTip("Game rules: 3, 4, or 5 players per table (FFT rules).")
        self._combo_league_style.setToolTip("ELO-based: pair similar-strength agents. Random: random table assignment.")
        self._spin_deals.setToolTip("Number of deals played in each match. More deals give more stable ELO updates.")
        self._spin_matches.setToolTip("Number of tournament rounds per generation. More rounds refine ELO rankings.")

        # ELO tuning: checkbox (on by default), params always visible but greyed when unchecked
        self._cb_elo_tuning = QtWidgets.QCheckBox("ELO tuning")
        self._cb_elo_tuning.setChecked(True)
        self._cb_elo_tuning.toggled.connect(self._update_tour_elo_enabled)
        self._tour_elo_frame = QtWidgets.QFrame()
        self._tour_elo_frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._tour_elo_frame.setEnabled(True)
        elo_layout = QtWidgets.QVBoxLayout(self._tour_elo_frame)
        self._spin_elo_k = QtWidgets.QDoubleSpinBox()
        self._spin_elo_k.setRange(1, 100)
        self._spin_elo_k.setValue(32)
        self._spin_elo_k.setMinimumWidth(52)
        self._spin_elo_k.setToolTip("Max ELO change per pairwise comparison. Higher = faster rating changes.")
        self._spin_elo_margin = QtWidgets.QDoubleSpinBox()
        self._spin_elo_margin.setRange(1, 200)
        self._spin_elo_margin.setValue(50)
        self._spin_elo_margin.setMinimumWidth(56)
        self._spin_elo_margin.setToolTip("Scale for score-diff to result mapping. Affects how big wins influence ELO.")
        # Row 1: ELO K-factor
        elo_row1 = QtWidgets.QHBoxLayout()
        lbl_elo_k = QtWidgets.QLabel("ELO K-factor:")
        lbl_elo_k.setToolTip("Max ELO change per pairwise comparison. Higher = faster rating changes.")
        elo_row1.addWidget(lbl_elo_k)
        elo_row1.addWidget(self._spin_elo_k)
        elo_row1.addStretch()
        # Row 2: ELO margin scale
        elo_row2 = QtWidgets.QHBoxLayout()
        lbl_elo_margin = QtWidgets.QLabel("ELO margin scale:")
        lbl_elo_margin.setToolTip("Scale for score-diff to result mapping. Affects how big wins influence ELO.")
        elo_row2.addWidget(lbl_elo_margin)
        elo_row2.addWidget(self._spin_elo_margin)
        elo_row2.addStretch()
        elo_layout.addLayout(elo_row1)
        elo_layout.addLayout(elo_row2)

        # ELO normalization toggle: keep mean ELO stable across generations when enabled
        self._cb_elo_normalize = QtWidgets.QCheckBox("Normalize ELO mean between generations")
        self._cb_elo_normalize.setChecked(True)
        self._cb_elo_normalize.setToolTip(
            "When checked, shift non-fixed ELOs after GA so the new generation's mean global ELO "
            "matches the previous generation's mean."
        )

        # PPO fine-tuning: checkbox (on by default), params always visible but greyed when unchecked
        self._cb_ppo = QtWidgets.QCheckBox("PPO fine-tuning")
        self._cb_ppo.setChecked(True)
        self._cb_ppo.toggled.connect(self._update_tour_ppo_enabled)
        self._tour_ppo_frame = QtWidgets.QFrame()
        self._tour_ppo_frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._tour_ppo_frame.setEnabled(True)
        ppo_layout = QtWidgets.QVBoxLayout(self._tour_ppo_frame)
        self._spin_ppo_top_k = QtWidgets.QSpinBox()
        self._spin_ppo_top_k.setRange(0, 999)
        self._spin_ppo_top_k.setValue(0)
        self._spin_ppo_top_k.setMinimumWidth(52)
        self._spin_ppo_top_k.setToolTip("0 = disabled. Top-K agents by fitness get PPO fine-tuning each generation.")
        self._spin_ppo_updates = QtWidgets.QSpinBox()
        self._spin_ppo_updates.setRange(0, 9999)
        self._spin_ppo_updates.setValue(0)
        self._spin_ppo_updates.setMinimumWidth(56)
        self._spin_ppo_updates.setToolTip("Number of PPO update steps per agent when top-K > 0.")
        # Row 1: PPO top-K
        ppo_row1 = QtWidgets.QHBoxLayout()
        lbl_ppo_k = QtWidgets.QLabel("PPO top-K:")
        lbl_ppo_k.setToolTip("0 = disabled. Top-K agents by fitness get PPO fine-tuning each generation.")
        ppo_row1.addWidget(lbl_ppo_k)
        ppo_row1.addWidget(self._spin_ppo_top_k)
        ppo_row1.addStretch()
        # Row 2: PPO updates/agent
        ppo_row2 = QtWidgets.QHBoxLayout()
        lbl_ppo_updates = QtWidgets.QLabel("PPO updates/agent:")
        lbl_ppo_updates.setToolTip("Number of PPO update steps per agent when top-K > 0.")
        ppo_row2.addWidget(lbl_ppo_updates)
        ppo_row2.addWidget(self._spin_ppo_updates)
        ppo_row2.addStretch()
        ppo_layout.addLayout(ppo_row1)
        ppo_layout.addLayout(ppo_row2)

        # Generations: total number of generations to run (moved from Next Generation box)
        gens_row = QtWidgets.QHBoxLayout()
        self._spin_generations = QtWidgets.QSpinBox()
        self._spin_generations.setRange(1, 9999)
        self._spin_generations.setValue(10)
        self._spin_generations.setToolTip("Total number of generations to run.")
        lbl_gens = QtWidgets.QLabel("Generations:")
        lbl_gens.setToolTip("Total number of generations to run.")
        gens_row.addWidget(lbl_gens)
        gens_row.addWidget(self._spin_generations)
        gens_row.addStretch()
        core_layout.addRow(gens_row)

        # Assemble Tournament box as three columns: left (core + generations), middle (ELO tuning),
        # right (PPO fine-tuning).
        tour_columns = QtWidgets.QHBoxLayout()

        left_col_widget = QtWidgets.QWidget()
        left_col_layout = QtWidgets.QVBoxLayout(left_col_widget)
        left_col_layout.setContentsMargins(0, 0, 0, 0)
        left_col_layout.addWidget(tour_core)
        left_col_layout.addStretch()

        middle_col_widget = QtWidgets.QWidget()
        middle_col_layout = QtWidgets.QVBoxLayout(middle_col_widget)
        middle_col_layout.setContentsMargins(0, 0, 0, 0)
        middle_col_layout.addWidget(self._cb_elo_tuning)
        middle_col_layout.addWidget(self._tour_elo_frame)
        middle_col_layout.addWidget(self._cb_elo_normalize)
        middle_col_layout.addStretch()

        right_col_widget = QtWidgets.QWidget()
        right_col_layout = QtWidgets.QVBoxLayout(right_col_widget)
        right_col_layout.setContentsMargins(0, 0, 0, 0)
        right_col_layout.addWidget(self._cb_ppo)
        right_col_layout.addWidget(self._tour_ppo_frame)
        right_col_layout.addStretch()

        tour_columns.addWidget(left_col_widget, 2)
        tour_columns.addWidget(middle_col_widget, 1)
        tour_columns.addWidget(right_col_widget, 1)
        tour_layout.addLayout(tour_columns)

        self._tour_insights = QtWidgets.QLabel("Tables/round: —  Matches/gen: —  Deals/agent: —")
        self._tour_insights.setWordWrap(True)
        self._tour_insights.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        tour_layout.addWidget(self._tour_insights)
        tour_group.setFixedHeight(FLOW_BOX_HEIGHT_ROW1)
        flow_row1.addWidget(tour_group, 1)

        # Fitness block: Weights (core, no checkbox) | Selection (checkbox)
        fit_group = QtWidgets.QGroupBox("Fitness")
        fit_layout = QtWidgets.QVBoxLayout(fit_group)
        # Fitness = a*ELO^b + c*avg_score^d. Line 1: a, b. Line 2: c, d.
        self._spin_fitness_a = QtWidgets.QDoubleSpinBox()
        self._spin_fitness_a.setRange(0, 100)
        self._spin_fitness_a.setValue(1.0)
        self._spin_fitness_a.setDecimals(1)
        self._spin_fitness_a.setSingleStep(0.1)
        self._spin_fitness_a.setToolTip("Coefficient for ELO term: a in a×ELO^b.")
        self._spin_fitness_b = QtWidgets.QDoubleSpinBox()
        self._spin_fitness_b.setRange(0.01, 5.0)
        self._spin_fitness_b.setValue(1.0)
        self._spin_fitness_b.setDecimals(2)
        self._spin_fitness_b.setSingleStep(0.01)
        self._spin_fitness_b.setToolTip("Exponent for ELO: b in a×ELO^b.")
        fit_row_elo = QtWidgets.QHBoxLayout()
        lbl_a = QtWidgets.QLabel("a (ELO coef):")
        lbl_a.setToolTip("Coefficient for ELO term: a in a×ELO^b.")
        fit_row_elo.addWidget(lbl_a)
        fit_row_elo.addWidget(self._spin_fitness_a)
        fit_row_elo.addSpacing(12)
        lbl_b = QtWidgets.QLabel("b (ELO exp):")
        lbl_b.setToolTip("Exponent for ELO: b in a×ELO^b.")
        fit_row_elo.addWidget(lbl_b)
        fit_row_elo.addWidget(self._spin_fitness_b)
        fit_row_elo.addStretch()
        fit_layout.addLayout(fit_row_elo)
        self._spin_fitness_c = QtWidgets.QDoubleSpinBox()
        self._spin_fitness_c.setRange(0, 100)
        self._spin_fitness_c.setValue(0.0)
        self._spin_fitness_c.setDecimals(1)
        self._spin_fitness_c.setSingleStep(0.1)
        self._spin_fitness_c.setToolTip("Coefficient for avg_score term: c in c×avg_score^d.")
        self._spin_fitness_d = QtWidgets.QDoubleSpinBox()
        self._spin_fitness_d.setRange(0.01, 5.0)
        self._spin_fitness_d.setValue(1.0)
        self._spin_fitness_d.setDecimals(2)
        self._spin_fitness_d.setSingleStep(0.01)
        self._spin_fitness_d.setToolTip("Exponent for avg_score: d in c×avg_score^d.")
        fit_row_score = QtWidgets.QHBoxLayout()
        lbl_c = QtWidgets.QLabel("c (avg_score coef):")
        lbl_c.setToolTip("Coefficient for avg_score term: c in c×avg_score^d.")
        fit_row_score.addWidget(lbl_c)
        fit_row_score.addWidget(self._spin_fitness_c)
        fit_row_score.addSpacing(12)
        lbl_d = QtWidgets.QLabel("d (avg_score exp):")
        lbl_d.setToolTip("Exponent for avg_score: d in c×avg_score^d.")
        fit_row_score.addWidget(lbl_d)
        fit_row_score.addWidget(self._spin_fitness_d)
        fit_row_score.addStretch()
        fit_layout.addLayout(fit_row_score)
        self._fitness_formula = QtWidgets.QLabel("Fitness = a×ELO^b + c×avg_score^d")
        self._fitness_formula.setStyleSheet("color: #b8b8b8; font-size: 14px; font-weight: 500; padding: 4px 0;")
        self._fitness_formula.setWordWrap(True)
        self._fitness_formula.setMinimumHeight(28)
        for spin in (self._spin_fitness_a, self._spin_fitness_b, self._spin_fitness_c, self._spin_fitness_d):
            spin.valueChanged.connect(self._update_fitness_formula)
        fit_layout.addWidget(self._fitness_formula)
        self._fitness_visual = FitnessVisualWidget()
        self._fitness_visual.setMinimumHeight(FLOW_GRAPH_MIN_HEIGHT)
        fit_layout.addWidget(self._fitness_visual)
        fit_group.setFixedHeight(FLOW_BOX_HEIGHT_ROW2)
        flow_row2.addWidget(fit_group, 1)

        # Reproduction block: Elite %, Clone %, Mutation %, Mut std, distribution graph
        MUT_TOP_ROW_HEIGHT = 42
        mut_group = QtWidgets.QGroupBox("Reproduction")
        mut_group.setFixedHeight(FLOW_BOX_HEIGHT_ROW2)
        mut_layout = QtWidgets.QVBoxLayout(mut_group)
        mut_layout.setSpacing(8)
        mut_layout.setContentsMargins(10, 8, 10, 10)

        def _rep_compact_row(w: QtWidgets.QWidget) -> QtWidgets.QWidget:
            """Wrap widget in a fixed-height row (plain widget, no frame background)."""
            wrapper = QtWidgets.QWidget()
            wrapper.setFixedHeight(MUT_TOP_ROW_HEIGHT)
            wrapper.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            lo = QtWidgets.QVBoxLayout(wrapper)
            lo.setContentsMargins(0, 0, 0, 0)
            lo.addWidget(w)
            return wrapper

        # 1. Sexual offspring, Mutated, Cloned as counts (must sum to GA-eligible slots)
        params_row = QtWidgets.QHBoxLayout()
        self._btn_sexual_settings = QtWidgets.QToolButton()
        self._btn_sexual_settings.setText("\u2699")  # gear
        self._btn_sexual_settings.setToolTip("Configure which parameters are combined in sexual reproduction.")
        self._btn_sexual_settings.clicked.connect(self._on_sexual_reproduction_settings)
        lbl_sexual = QtWidgets.QLabel("Sexual offspring:")
        lbl_sexual.setToolTip(
            "Number of slots filled by offspring from two parents. That many agents are eliminated and "
            "replaced; parents are chosen from the elite pool (fitness-weighted) and their parameters are combined."
        )
        self._spin_kept = QtWidgets.QSpinBox()
        self._spin_kept.setRange(0, 9999)
        self._spin_kept.setValue(1)
        self._spin_kept.setMinimumWidth(64)
        self._spin_kept.setToolTip(
            "Number of slots filled by sexual offspring (two parents from elite pool, parameters combined)."
        )
        self._spin_kept.valueChanged.connect(lambda: self._on_repro_count_changed("kept"))
        lbl_mutate = QtWidgets.QLabel("Mutated:")
        lbl_mutate.setToolTip("Number of offspring from mutation.")
        self._spin_mutate = QtWidgets.QSpinBox()
        self._spin_mutate.setRange(0, 9999)
        self._spin_mutate.setValue(0)
        self._spin_mutate.setMinimumWidth(64)
        self._spin_mutate.setToolTip("Number of offspring from mutation.")
        self._spin_mutate.valueChanged.connect(lambda: self._on_repro_count_changed("mutate"))
        lbl_clone = QtWidgets.QLabel("Cloned:")
        lbl_clone.setToolTip("Number of offspring from cloning elites.")
        self._spin_clone = QtWidgets.QSpinBox()
        self._spin_clone.setRange(0, 9999)
        self._spin_clone.setValue(0)
        self._spin_clone.setMinimumWidth(64)
        self._spin_clone.setToolTip("Number of offspring from cloning elites.")
        self._spin_clone.valueChanged.connect(lambda: self._on_repro_count_changed("clone"))
        params_row.addWidget(self._btn_sexual_settings)
        params_row.addWidget(lbl_sexual)
        params_row.addWidget(self._spin_kept)
        params_row.addSpacing(16)
        params_row.addWidget(lbl_mutate)
        params_row.addWidget(self._spin_mutate)
        params_row.addSpacing(16)
        params_row.addWidget(lbl_clone)
        params_row.addWidget(self._spin_clone)
        params_row.addStretch()
        params_w = QtWidgets.QWidget()
        params_w.setLayout(params_row)
        mut_layout.addWidget(_rep_compact_row(params_w), 0)
        # Sexual reproduction gearbox state (used by get_league_config; set by dialog and set_league_config)
        self._sexual_parent_with_replacement = True
        self._sexual_parent_fitness_weighted = True
        self._sexual_trait_combination = "average"
        # 2. Selection bar with color legend and Fitness arrow
        self._reproduction_bar_widget = ReproductionBarWidget()
        mut_layout.addWidget(self._reproduction_bar_widget, 0)
        # 3. Mutation std and Trait Mutation prob (no frame background; placed close to bar)
        mut_std_row = QtWidgets.QHBoxLayout()
        lbl_mut_std = QtWidgets.QLabel("Mutation std:")
        lbl_mut_std.setToolTip("Standard deviation of the Gaussian used to perturb traits.")
        self._spin_mut_std = QtWidgets.QDoubleSpinBox()
        self._spin_mut_std.setRange(0.01, 1.0)
        self._spin_mut_std.setValue(0.1)
        self._spin_mut_std.setDecimals(2)
        self._spin_mut_std.setSingleStep(0.01)
        self._spin_mut_std.setMinimumWidth(72)
        self._spin_mut_std.setToolTip("Standard deviation of the Gaussian used to perturb traits.")
        lbl_trait_prob = QtWidgets.QLabel("Trait mutation prob:")
        lbl_trait_prob.setToolTip("Probability that a trait is perturbed when creating a mutant offspring.")
        self._spin_trait_prob = QtWidgets.QDoubleSpinBox()
        self._spin_trait_prob.setRange(0, 100)
        self._spin_trait_prob.setValue(50)
        self._spin_trait_prob.setSuffix(" %")
        self._spin_trait_prob.setDecimals(1)
        self._spin_trait_prob.setMinimumWidth(72)
        self._spin_trait_prob.setToolTip("Probability that a trait is perturbed when creating a mutant offspring.")
        mut_std_row.addWidget(lbl_mut_std)
        mut_std_row.addWidget(self._spin_mut_std)
        mut_std_row.addSpacing(16)
        mut_std_row.addWidget(lbl_trait_prob)
        mut_std_row.addWidget(self._spin_trait_prob)
        mut_std_row.addStretch()
        mut_std_w = QtWidgets.QWidget()
        mut_std_w.setLayout(mut_std_row)
        mut_std_row_wrapper = _rep_compact_row(mut_std_w)
        mut_std_row_wrapper.setStyleSheet("")  # ensure no groupbox/frame highlight
        mut_layout.addWidget(mut_std_row_wrapper, 0)
        # 4. Mutation distribution graph (same min height as Fitness graph)
        self._mut_dist_widget = MutationDistWidget()
        self._mut_dist_widget.setMinimumHeight(FLOW_GRAPH_MIN_HEIGHT)
        self._mut_dist_widget.setMaximumHeight(400)
        mut_layout.addWidget(self._mut_dist_widget, 0)
        mut_layout.addStretch(1)
        for spin in (self._spin_kept, self._spin_mutate, self._spin_clone, self._spin_mut_std, self._spin_trait_prob):
            spin.valueChanged.connect(self._update_ga_visual)
        flow_row2.addWidget(mut_group, 1)

        # Hidden Export & Hall of Fame controls (no visible box on League Parameters tab).
        # These widgets are kept for state persistence and tests, and may be wired into
        # Dashboard export/HOF flows, but are not added to this tab's layout.
        self._combo_export_when = QtWidgets.QComboBox()
        self._combo_export_when.addItems(["On demand only", "Every generation", "Every N generations"])
        self._spin_export_every_n = QtWidgets.QSpinBox()
        self._spin_export_every_n.setRange(1, 999)
        self._spin_export_every_n.setValue(5)
        self._combo_export_what = QtWidgets.QComboBox()
        self._combo_export_what.addItems(["Full population", "Top N by ELO", "GA-eligible only"])
        self._combo_hof_when = QtWidgets.QComboBox()
        self._combo_hof_when.addItems(["On demand only", "Every generation", "Every N generations"])
        self._spin_hof_every_n = QtWidgets.QSpinBox()
        self._spin_hof_every_n.setRange(1, 999)
        self._spin_hof_every_n.setValue(5)
        self._combo_hof_what = QtWidgets.QComboBox()
        self._combo_hof_what.addItems(["Top N by ELO", "Top N by fitness", "Best agent only"])
        self._spin_hof_top_n = QtWidgets.QSpinBox()
        self._spin_hof_top_n.setRange(1, 999)
        self._spin_hof_top_n.setValue(5)

        # Hidden label for next-generation insights (used by helper but not shown on this tab)
        self._next_gen_insights = QtWidgets.QLabel("")

        flow_container = QtWidgets.QWidget()
        flow_vertical.addLayout(flow_row1)
        flow_vertical.addLayout(flow_row2)
        flow_container.setLayout(flow_vertical)
        content_layout.addWidget(flow_container, 1)

        self._content_container.setMinimumHeight(CONTENT_HEIGHT_1080P)
        self._project_view_layout.addWidget(self._content_container, 0)
        self._project_view_layout.addStretch(1)

        self._stack = QtWidgets.QStackedWidget()
        self._stack.addWidget(self._no_project_view)
        self._stack.addWidget(self._project_view)
        layout.addWidget(self._stack)

        scroll.setWidget(content)
        # Option C: content width always = viewport width (no horizontal scroll).
        # When no project: content height = viewport height exactly so no vertical scroll.
        # When project loaded: content height fills viewport (min 1000px for full layout).
        _min_content_h = CONTENT_HEIGHT_1080P
        def _update_content_size():
            vp = scroll.viewport()
            vh, vw = vp.height(), max(vp.width(), 400)
            if self._state.project_path is None:
                content.setMinimumHeight(vh)
                content.setMaximumHeight(vh)
            else:
                content.setMinimumHeight(max(_min_content_h, vh))
                content.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
            content.setMinimumWidth(vw)
            content.setMaximumWidth(vw)
            if self._state.project_path is None and self._stack.currentWidget() is self._no_project_view:
                one_third = max(200, vh // 3)
                self._no_project_center.setMinimumHeight(one_third)
                self._project_group.setFixedHeight(one_third)
        _update_content_size()
        self._update_league_content_size = _update_content_size
        scroll.viewport().installEventFilter(
            _ResizeFilter(scroll, _update_content_size)
        )

        self._update_ga_visual()
        self._update_pie_chart()
        self._update_tournament_insights()
        self._update_sideline_warning()
        self._update_fitness_formula()
        self._sync_repro_counts_to_slots(self._get_ga_slots())
        self._update_next_gen_insights()
        self._update_tour_elo_enabled()
        self._update_tour_ppo_enabled()

        self._connect_league_params_dirty()

        self._update_content_visibility()

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _refresh_table(self) -> None:
        self._table.setRowCount(0)
        for row, group in enumerate(self._state.groups):
            self._table.insertRow(row)
            self._fill_group_row(row, group)
        # Auto-size columns to fit header and content (Group name stretches to fill)
        self._table.resizeColumnsToContents()
        self._update_player_count_options()
        self._update_pie_chart()
        if hasattr(self, "_sync_repro_counts_to_slots") and hasattr(self, "_get_ga_slots"):
            self._sync_repro_counts_to_slots(self._get_ga_slots())
        if hasattr(self, "_update_next_gen_insights"):
            self._update_next_gen_insights()
        if hasattr(self, "_update_tournament_insights"):
            self._update_tournament_insights()
        if hasattr(self, "_update_sideline_warning"):
            self._update_sideline_warning()

    def _update_pie_chart(self) -> None:
        total = self._state.total_agents()
        player_count = int(self._combo_player_count.currentData() or 4)
        ga_agents = sum(1 for g in self._state.groups for a in g.agents if a.can_use_as_ga_parent)
        fixed_elo = sum(1 for g in self._state.groups for a in g.agents if a.fixed_elo)
        clone_only = sum(1 for g in self._state.groups for a in g.agents if a.clone_only)
        play_in_league = sum(1 for g in self._state.groups for a in g.agents if a.play_in_league)
        group_slices = []
        for g in self._state.groups:
            ga_eligible = sum(1 for a in g.agents if a.can_use_as_ga_parent)
            play_count = sum(1 for a in g.agents if a.play_in_league)
            reference = sum(1 for a in g.agents if not a.can_use_as_ga_parent)
            group_slices.append(GroupSliceData(
                name=g.name,
                total=len(g.agents),
                ga_eligible=ga_eligible,
                play_in_league=play_count,
                reference=reference,
                color=g.color,
            ))
        group_by = "group"
        if hasattr(self, "_combo_pie_group_by"):
            text = self._combo_pie_group_by.currentText()
            group_by = "group" if text == "Group name" else ("ga_status" if text == "GA status" else "play_status")
        self._pie_widget.set_data(group_slices, total, player_count, group_by)
        # Update insights table under pie
        if hasattr(self, "_pie_insights_table"):
            counts = [ga_agents, fixed_elo, clone_only, play_in_league, total]
            for row, val in enumerate(counts):
                item = self._pie_insights_table.item(row, 1)
                if item:
                    item.setText(str(val))

    def _get_ga_slots(self) -> int:
        """GA-eligible agent count (slots for reproduction)."""
        total = self._state.total_agents()
        ga_eligible = sum(1 for g in self._state.groups for a in g.agents if a.can_use_as_ga_parent)
        ref = total - ga_eligible
        return max(0, total - ref)

    def _on_repro_count_changed(self, source: str) -> None:
        """Allow free adjustment; only clamp so each >= 0, each <= slots, and total <= slots (clamp the changed field if over)."""
        slots = self._get_ga_slots()
        if slots <= 0:
            return
        kept = max(0, min(slots, self._spin_kept.value()))
        mutate = max(0, min(slots, self._spin_mutate.value()))
        clone = max(0, min(slots, self._spin_clone.value()))
        total = kept + mutate + clone
        if total > slots:
            # Clamp the field that was just changed so total <= slots
            excess = total - slots
            if source == "kept":
                kept = max(0, kept - excess)
            elif source == "mutate":
                mutate = max(0, mutate - excess)
            else:
                clone = max(0, clone - excess)
        for spin in (self._spin_kept, self._spin_clone, self._spin_mutate):
            spin.blockSignals(True)
        try:
            self._spin_kept.setValue(kept)
            self._spin_mutate.setValue(mutate)
            self._spin_clone.setValue(clone)
        finally:
            for spin in (self._spin_kept, self._spin_clone, self._spin_mutate):
                spin.blockSignals(False)
        self._update_ga_visual()

    def _sync_repro_counts_to_slots(self, slots: int) -> None:
        """Set spin max to slots; clamp values so each in [0, slots] and total <= slots."""
        for spin in (self._spin_kept, self._spin_mutate, self._spin_clone):
            spin.setMaximum(max(0, slots))
        if slots <= 0:
            return
        kept = max(0, min(slots, self._spin_kept.value()))
        mutate = max(0, min(slots, self._spin_mutate.value()))
        clone = max(0, min(slots, self._spin_clone.value()))
        if kept + mutate + clone > slots:
            mutate = max(0, slots - kept - clone)
        for spin in (self._spin_kept, self._spin_clone, self._spin_mutate):
            spin.blockSignals(True)
        try:
            self._spin_kept.setValue(kept)
            self._spin_mutate.setValue(mutate)
            self._spin_clone.setValue(clone)
        finally:
            for spin in (self._spin_kept, self._spin_clone, self._spin_mutate):
                spin.blockSignals(False)
        self._update_ga_visual()

    def _on_sexual_reproduction_settings(self) -> None:
        """Open gearbox dialog to configure parent selection and trait combination for sexual reproduction."""
        dlg = SexualReproductionSettingsDialog(
            self,
            with_replacement=self._sexual_parent_with_replacement,
            fitness_weighted=self._sexual_parent_fitness_weighted,
            trait_combination=self._sexual_trait_combination,
        )
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._sexual_parent_with_replacement = dlg.get_with_replacement()
            self._sexual_parent_fitness_weighted = dlg.get_fitness_weighted()
            self._sexual_trait_combination = dlg.get_trait_combination()
            self._mark_league_params_dirty()

    def _mark_league_params_dirty(self) -> None:
        """Mark league parameters as changed; show unsaved warning when a project is loaded."""
        self._league_params_dirty = True
        if self._state.project_path:
            self._btn_unsaved_warning.setVisible(True)

    def _clear_league_params_dirty(self) -> None:
        """Clear dirty state and hide unsaved warning."""
        self._league_params_dirty = False
        self._btn_unsaved_warning.setVisible(False)

    def _connect_league_params_dirty(self) -> None:
        """Connect all league-parameter widgets to mark dirty when changed."""
        league_config_widgets = [
            self._combo_league_style,
            self._combo_player_count,
            self._spin_deals,
            self._spin_matches,
            self._spin_elo_k,
            self._spin_elo_margin,
            self._spin_ppo_top_k,
            self._spin_ppo_updates,
            self._spin_fitness_a,
            self._spin_fitness_b,
            self._spin_fitness_c,
            self._spin_fitness_d,
            self._spin_kept,
            self._spin_mutate,
            self._spin_clone,
            self._spin_trait_prob,
            self._spin_mut_std,
            self._spin_generations,
            self._spin_export_every_n,
        ]
        for w in league_config_widgets:
            if hasattr(w, "valueChanged"):
                w.valueChanged.connect(self._mark_league_params_dirty)
            elif hasattr(w, "currentIndexChanged"):
                w.currentIndexChanged.connect(self._mark_league_params_dirty)
            elif hasattr(w, "currentTextChanged"):
                w.currentTextChanged.connect(self._mark_league_params_dirty)
        self._cb_elo_tuning.toggled.connect(self._mark_league_params_dirty)
        self._cb_elo_normalize.toggled.connect(self._mark_league_params_dirty)
        self._cb_ppo.toggled.connect(self._mark_league_params_dirty)
        self._combo_export_when.currentIndexChanged.connect(self._mark_league_params_dirty)
        self._combo_export_what.currentIndexChanged.connect(self._mark_league_params_dirty)
        self._combo_hof_when.currentIndexChanged.connect(self._mark_league_params_dirty)
        self._combo_hof_what.currentIndexChanged.connect(self._mark_league_params_dirty)
        self._spin_hof_every_n.valueChanged.connect(self._mark_league_params_dirty)
        self._spin_hof_top_n.valueChanged.connect(self._mark_league_params_dirty)

    def _update_ga_visual(self) -> None:
        slots, kept_count, clone_slots, mutate_slots = self._get_repro_counts()
        # Bar: sexual offspring (red), mutated, cloned; sum = slots
        self._reproduction_bar_widget.set_params(
            0, 0, 0,
            total_agents=slots,
            counts=(kept_count, mutate_slots, clone_slots),
        )
        self._mut_dist_widget.set_mutation_std(self._spin_mut_std.value())
        # Visualize trait-level mutation probability (0–1) in the distribution widget
        self._mut_dist_widget.set_mutation_prob(self._spin_trait_prob.value() / 100.0)

    def _update_tour_elo_enabled(self) -> None:
        enabled = self._cb_elo_tuning.isChecked()
        self._tour_elo_frame.setEnabled(enabled)
        for w in self._tour_elo_frame.findChildren(QtWidgets.QWidget):
            w.setEnabled(enabled)

    def _update_tour_ppo_enabled(self) -> None:
        enabled = self._cb_ppo.isChecked()
        self._tour_ppo_frame.setEnabled(enabled)
        for w in self._tour_ppo_frame.findChildren(QtWidgets.QWidget):
            w.setEnabled(enabled)

    def _update_sideline_warning(self) -> None:
        """Show warning icon (⚠) next to Rules when total agent count is not a multiple of player count; full message in tooltip."""
        total = self._state.total_agents()
        pc = int(self._combo_player_count.currentData() or 4)
        if pc <= 0 or total == 0:
            self._sideline_warning_icon.setText("")
            self._sideline_warning_frame.setToolTip("")
            return
        remainder = total % pc
        if remainder == 0:
            self._sideline_warning_icon.setText("")
            self._sideline_warning_frame.setToolTip("")
            return
        on_sidelines = remainder
        line1 = f"With {total} agents and {pc}-player tables, {on_sidelines} agent(s) will be randomly selected to sit out each matchmaking phase."
        line2 = f"Consider adjusting the population to a multiple of {pc} so every agent can play every round."
        self._sideline_warning_icon.setText("\u26a0")
        self._sideline_warning_frame.setToolTip(line1 + "\n" + line2)

    def _update_tournament_insights(self) -> None:
        total = self._state.total_agents()
        play_count = sum(1 for g in self._state.groups for a in g.agents if a.play_in_league)
        pc = int(self._combo_player_count.currentData() or 4)
        rounds = self._spin_matches.value()
        deals = self._spin_deals.value()
        tables = play_count // pc if pc > 0 else 0
        matches_gen = rounds * tables
        deals_agent = (matches_gen * deals * pc / play_count) if play_count > 0 else 0
        self._tour_insights.setText(
            f"Tables/round: {tables}  Matches/gen: {matches_gen}  "
            f"Deals/agent (approx): {deals_agent:.1f}"
        )

    def _update_fitness_formula(self) -> None:
        a, b = self._spin_fitness_a.value(), self._spin_fitness_b.value()
        c, d = self._spin_fitness_c.value(), self._spin_fitness_d.value()
        self._fitness_formula.setText(
            f"Fitness = {a:.1f}×ELO^{b:.2f} + {c:.1f}×avg_score^{d:.2f}"
        )
        self._fitness_visual.set_params(a, b, c, d)

    def _get_repro_counts(self) -> tuple[int, int, int, int]:
        """Return (slots, sexual_offspring, clone_slots, mutate_slots) from spinboxes; each in [0, slots], sum may be <= slots."""
        slots = self._get_ga_slots()
        if slots <= 0:
            return 0, 0, 0, 0
        kept = max(0, min(slots, self._spin_kept.value()))
        clone = max(0, min(slots, self._spin_clone.value()))
        mutate = max(0, min(slots, self._spin_mutate.value()))
        return slots, kept, clone, mutate

    def _update_next_gen_insights(self) -> None:
        total = self._state.total_agents()
        ga_eligible = sum(1 for g in self._state.groups for a in g.agents if a.can_use_as_ga_parent)
        ref = total - ga_eligible
        self._next_gen_insights.setText(
            f"Population size: {total}  Ref: {ref}  GA-eligible: {ga_eligible}"
        )

    def _on_export_now(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export population", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            pop = self._state.build_population()
            with open(path, "w", encoding="utf-8") as f:
                f.write(population_to_json(pop))
            QtWidgets.QMessageBox.information(self, "Export", f"Exported to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))

    def add_to_hof_from_population(
        self, pop: Population, generation_index: int
    ) -> None:
        """
        If HOF when is every gen or every N, select agents from pop by HOF what
        (top N by ELO, top N by fitness, or best only) and append clones to state.hof_agents.
        """
        when_idx = self._combo_hof_when.currentIndex()
        if when_idx == 0:
            return
        if when_idx == 2:
            n = self._spin_hof_every_n.value()
            if n <= 0 or generation_index % n != 0:
                return
        # Select agents to add
        what_idx = self._combo_hof_what.currentIndex()
        top_n = self._spin_hof_top_n.value() if what_idx != 2 else 1
        agents_to_add: List[Agent] = []
        if what_idx == 0:
            # Top N by ELO
            sorted_agents = sorted(
                pop.agents.values(),
                key=lambda a: a.elo_global,
                reverse=True,
            )
            agents_to_add = sorted_agents[:top_n]
        elif what_idx == 1:
            # Top N by fitness
            cfg = self.get_league_config()
            def fitness_fn(a: Agent) -> float:
                return compute_fitness(
                    a,
                    fitness_elo_a=cfg.fitness_elo_a,
                    fitness_elo_b=cfg.fitness_elo_b,
                    fitness_avg_c=cfg.fitness_avg_c,
                    fitness_avg_d=cfg.fitness_avg_d,
                )
            sorted_agents = sorted(
                pop.agents.values(),
                key=fitness_fn,
                reverse=True,
            )
            agents_to_add = sorted_agents[:top_n]
        else:
            # Best agent only (by ELO)
            if pop.agents:
                best = max(pop.agents.values(), key=lambda a: a.elo_global)
                agents_to_add = [best]
        # Clone with unique ids and append
        for i, a in enumerate(agents_to_add):
            new_id = f"{a.id}_gen{generation_index}_{i}"
            clone = replace(a, id=new_id)
            self._state.hof_agents.append(clone)

    def _on_export_hof_now(self) -> None:
        """Export current Hall of Fame agents to a JSON file."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Hall of Fame", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            pop = Population()
            for a in self._state.hof_agents:
                pop.add(a)
            with open(path, "w", encoding="utf-8") as f:
                f.write(population_to_json(pop))
            QtWidgets.QMessageBox.information(
                self, "Export HOF", f"Exported {len(self._state.hof_agents)} agent(s) to {path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export HOF failed", str(e))

    def _update_player_count_options(self) -> None:
        """Gray out player count options that are not possible given total agents."""
        total = self._state.total_agents()
        model = self._combo_player_count.model()
        current_idx = self._combo_player_count.currentIndex()
        first_enabled = -1
        for row in range(model.rowCount()):
            item = model.item(row)
            n = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if n is None:
                n = 4
            enabled = total >= n
            item.setEnabled(enabled)
            if enabled and first_enabled < 0:
                first_enabled = row
        if 0 <= current_idx < model.rowCount() and not model.item(current_idx).isEnabled() and first_enabled >= 0:
            self._combo_player_count.setCurrentIndex(first_enabled)

    def _fill_group_row(self, row: int, group: Group) -> None:
        # Selection checkbox for Augment / Clear selected
        sel_cb = QtWidgets.QCheckBox()
        sel_cb.setToolTip("Select for Augment or Clear selected")
        sel_cell = QtWidgets.QWidget()
        sel_layout = QtWidgets.QHBoxLayout(sel_cell)
        sel_layout.setContentsMargins(2, 1, 2, 1)
        sel_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        sel_layout.addWidget(sel_cb)
        self._table.setCellWidget(row, GRP_COL_SELECT, sel_cell)

        btn_expand = QtWidgets.QPushButton("Expand")
        btn_expand.setFixedSize(52, 18)
        btn_expand.setStyleSheet("padding: 0 4px; min-height: 0;")
        btn_expand.clicked.connect(lambda checked=False, g=group: self._on_expand_group(g))
        cell = QtWidgets.QWidget()
        ll = QtWidgets.QHBoxLayout(cell)
        ll.setContentsMargins(2, 0, 2, 0)
        ll.addWidget(btn_expand)
        self._table.setCellWidget(row, GRP_COL_EXPAND, cell)

        # Color swatch (click to change)
        color_btn = QtWidgets.QPushButton()
        color_btn.setFixedSize(22, 22)
        color_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        color_btn.setToolTip(f"Click to change color for {group.name}")
        color_btn.clicked.connect(lambda checked=False, g=group: self._on_pick_group_color(g))

        r = (group.color >> 16) & 0xFF
        grn = (group.color >> 8) & 0xFF
        blu = group.color & 0xFF
        color_btn.setStyleSheet(
            f"QPushButton {{ background-color: rgb({r},{grn},{blu}); border: 1px solid #555; }}"
            f"QPushButton:hover {{ border: 1px solid #888; }}"
        )

        color_container = QtWidgets.QWidget()
        color_layout = QtWidgets.QHBoxLayout(color_container)
        color_layout.setContentsMargins(2, 2, 2, 2)
        color_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        color_layout.addWidget(color_btn)
        self._table.setCellWidget(row, GRP_COL_COLOR, color_container)

        def make_flag_cell(checked: bool, setter) -> QtWidgets.QWidget:
            cb = QtWidgets.QCheckBox()
            cb.setChecked(checked)
            def _on_flag_toggle(c: bool) -> None:
                setter(c)
                self._update_pie_chart()
            cb.toggled.connect(_on_flag_toggle)
            w = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(w)
            lay.setContentsMargins(4, 2, 4, 2)
            lay.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(cb)
            return w

        self._table.setCellWidget(
            row, GRP_COL_GA_PARENT,
            make_flag_cell(group.all_can_use_as_ga_parent(), group.set_all_can_use_as_ga_parent),
        )
        self._table.setCellWidget(
            row, GRP_COL_FIXED_ELO,
            make_flag_cell(group.all_fixed_elo(), group.set_all_fixed_elo),
        )
        self._table.setCellWidget(
            row, GRP_COL_CLONE_ONLY,
            make_flag_cell(group.all_clone_only(), group.set_all_clone_only),
        )
        self._table.setCellWidget(
            row, GRP_COL_PLAY_IN_LEAGUE,
            make_flag_cell(group.all_play_in_league(), group.set_all_play_in_league),
        )

        self._table.setItem(row, GRP_COL_NAME, QtWidgets.QTableWidgetItem(group.name))
        self._table.setItem(row, GRP_COL_AGENTS, QtWidgets.QTableWidgetItem(str(len(group.agents))))
        src = group.source_group_name or (group.source_group_id or "—")
        self._table.setItem(row, GRP_COL_SOURCE, QtWidgets.QTableWidgetItem(src))
        elo_str = f"{group.elo_min():.0f} / {group.elo_mean():.0f} / {group.elo_max():.0f}"
        self._table.setItem(row, GRP_COL_ELO, QtWidgets.QTableWidgetItem(elo_str))

        btn_del = QtWidgets.QPushButton("Delete")
        btn_del.setFixedSize(52, 18)
        btn_del.setStyleSheet("padding: 0 4px; min-height: 0;")
        btn_del.clicked.connect(lambda checked=False, g=group: self._on_delete_group(g))
        cell_actions = QtWidgets.QWidget()
        al = QtWidgets.QHBoxLayout(cell_actions)
        al.setContentsMargins(2, 0, 2, 0)
        al.addWidget(btn_del)
        self._table.setCellWidget(row, GRP_COL_ACTIONS, cell_actions)

    def _on_expand_group(self, group: Group) -> None:
        dlg = GroupDetailDialog(group, self)
        dlg.exec()
        self._update_pie_chart()

    def _on_delete_group(self, group: Group) -> None:
        if group in self._state.groups:
            self._state.groups.remove(group)
            self._refresh_table()

    def _on_pick_group_color(self, group: Group) -> None:
        from PySide6 import QtGui
        r = (group.color >> 16) & 0xFF
        g = (group.color >> 8) & 0xFF
        b = group.color & 0xFF
        initial = QtGui.QColor(r, g, b)
        color = QtWidgets.QColorDialog.getColor(initial, self, f"Color for {group.name}")
        if color.isValid():
            group.color = (color.red() << 16) | (color.green() << 8) | color.blue()
            self._refresh_table()

    def _existing_agent_ids(self) -> set[str]:
        ids: set[str] = set()
        for g in self._state.groups:
            for a in g.agents:
                ids.add(a.id)
        return ids

    def _on_add_random(self) -> None:
        n = self._spin_add_random.value()
        gid = _next_group_id("rand")
        agents = generate_random_agents(n, [4], self._rng, id_prefix=gid)
        existing = self._existing_agent_ids()
        for i, a in enumerate(agents):
            a.id = f"{gid}_{i}"
            existing.add(a.id)
        used_names = {g.name for g in self._state.groups}
        used_colors = {g.color for g in self._state.groups}
        group = Group(
            id=gid,
            name=_pick_random_group_name(used_names, self._rng),
            agents=agents,
            source_group_name="Random generation",
            color=_pick_group_color(used_colors, self._rng),
        )
        self._state.groups.append(group)
        self._refresh_table()

    def _on_import(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import population", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            imported = population_from_dict(d)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import failed", f"Could not load population: {e}")
            return
        self._apply_imported_population(imported, "file")

    def _on_import_from_agents_folder(self) -> None:
        """Import all agents from the project's agents/ folder."""
        if not self._state.project_path:
            QtWidgets.QMessageBox.warning(
                self,
                "Import",
                "No project loaded. Open a project first, or use Import → From file... to load from a JSON file.",
            )
            return
        dir_path = Path(self._state.project_path) / "agents"
        if not dir_path.is_dir():
            QtWidgets.QMessageBox.information(
                self,
                "Import from agents folder",
                f"No agents folder found at:\n{dir_path}\n\nExport run output from the Dashboard to create it.",
            )
            return
        imported = load_population_from_directory(dir_path)
        self._apply_imported_population(imported, "agents folder")

    def _on_import_from_hof(self) -> None:
        """Import all agents from the project's agents/Hall of Fame/ folder."""
        if not self._state.project_path:
            QtWidgets.QMessageBox.warning(
                self,
                "Import",
                "No project loaded. Open a project first, or use Import → From file... to load from a JSON file.",
            )
            return
        dir_path = Path(self._state.project_path) / "agents" / "Hall of Fame"
        if not dir_path.is_dir():
            QtWidgets.QMessageBox.information(
                self,
                "Import from Hall of Fame",
                f"No Hall of Fame folder found at:\n{dir_path}\n\nExport run output to Hall of Fame from the Dashboard to create it.",
            )
            return
        imported = load_population_from_directory(dir_path)
        self._apply_imported_population(imported, "Hall of Fame")

    def _apply_imported_population(self, imported: Population, source_label: str) -> None:
        """Show replace/merge dialog and add the imported population as a group."""
        if not imported.agents:
            QtWidgets.QMessageBox.information(self, "Import", f"{source_label} contains no agents.")
            return
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("Import")
        msg.setText("Replace current population or merge with existing?")
        btn_replace = msg.addButton("Replace", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        btn_merge = msg.addButton("Merge", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        btn_cancel = msg.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        msg.exec()
        if msg.clickedButton() == btn_cancel:
            return
        gid = _next_group_id("imp")
        agents_list = list(imported.agents.values())
        renamed = _assign_group_agent_ids(agents_list, gid)
        used_colors = {g.color for g in self._state.groups}
        new_group = Group(
            id=gid,
            name=f"Imported from {source_label} ({len(renamed)})",
            agents=renamed,
            source_group_name=source_label,
            color=_pick_group_color(used_colors, self._rng),
        )
        if msg.clickedButton() == btn_replace:
            self._state.groups = [new_group]
        else:
            self._state.groups.append(new_group)
        self._refresh_table()

    def load_population_from_file(self) -> None:
        """Open a JSON population file and load it into the current population (same behavior as Import button)."""
        self._on_import()

    def _get_checked_group_rows(self) -> List[int]:
        """Return row indices where the selection checkbox is checked."""
        indices: List[int] = []
        for row in range(self._table.rowCount()):
            w = self._table.cellWidget(row, GRP_COL_SELECT)
            if w:
                for child in w.findChildren(QtWidgets.QCheckBox):
                    if child.isChecked():
                        indices.append(row)
                        break
        return indices

    def _on_augment_from_selection(self) -> None:
        rows = self._get_checked_group_rows()
        if not rows:
            QtWidgets.QMessageBox.information(self, "Augment", "Check one or more group rows to use as base.")
            return
        base_agents: List[Agent] = []
        source_names: List[str] = []
        for row in rows:
            if 0 <= row < len(self._state.groups):
                g = self._state.groups[row]
                base_agents.extend(g.agents)
                source_names.append(g.name)
        if not base_agents:
            return

        dialog = AugmentDialog(self, len(base_agents))
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        gid = _next_group_id("mut")
        source_label = ", ".join(source_names) if source_names else "selection"
        group_name = f"Mutated from {source_label}"

        n_mutate = dialog.mutate_count()
        n_clone = dialog.clone_count()
        existing = self._existing_agent_ids()
        new_agents: List[Agent] = []

        if n_mutate > 0:
            children = mutate_from_base(
                base_agents, n_mutate, dialog.mutation_prob(), dialog.mutation_std(), self._rng,
                existing_ids=existing,
            )
            for i, a in enumerate(children):
                a.id = f"{gid}_{len(new_agents)}"
                existing.add(a.id)
                new_agents.append(a)

        if n_clone > 0:
            clones = clone_agents(base_agents, n_clone, self._rng, existing_ids=existing)
            for a in clones:
                a.id = f"{gid}_{len(new_agents)}"
                existing.add(a.id)
                new_agents.append(a)

        if new_agents:
            src_id = self._state.groups[rows[0]].id if rows else None
            used_colors = {g.color for g in self._state.groups}
            new_group = Group(
                id=gid,
                name=group_name,
                agents=new_agents,
                source_group_id=src_id,
                source_group_name=source_label,
                color=_pick_group_color(used_colors, self._rng),
            )
            self._state.groups.append(new_group)

        self._refresh_table()

    def _on_clear_selected(self) -> None:
        rows = self._get_checked_group_rows()
        if not rows:
            QtWidgets.QMessageBox.information(self, "Clear selected", "Check one or more group rows to remove.")
            return
        groups_to_remove = [self._state.groups[r] for r in sorted(rows, reverse=True) if 0 <= r < len(self._state.groups)]
        if not groups_to_remove:
            return
        for g in groups_to_remove:
            if g in self._state.groups:
                self._state.groups.remove(g)
        self._refresh_table()

    def _on_clear(self) -> None:
        if not self._state.groups:
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Clear population",
            "Remove all groups?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self._state.groups = []
            self._refresh_table()

    def state(self) -> LeagueTabState:
        return self._state

    def get_num_generations(self) -> int:
        """Number of generations to run (from Next Generation spinbox)."""
        return self._spin_generations.value()

    def apply_population_from_run(
        self, pop: Population, generation_index: int, summary: Dict[str, float]
    ) -> None:
        """
        Replace state with the result of a league run.

        Default behaviour collapses the evolved Population into a single
        "League (gen N)" group. However, groups marked clone-only are treated
        as anchor populations: their agents keep their own group when possible
        so that reference / Hall-of-Fame cohorts remain distinct.
        """
        previous_groups = list(self._state.groups)
        clone_only_groups = [g for g in previous_groups if g.all_clone_only()]

        if not clone_only_groups:
            # No anchors: keep legacy behaviour (single combined group).
            self._state.groups = [_population_to_single_group(pop, generation_index)]
        else:
            # Partition agents: those that can be mapped back to a clone-only
            # group (by id prefix) stay in that group; the rest go to the main
            # League group.
            anchor_members: Dict[str, List[Agent]] = {g.id: [] for g in clone_only_groups}
            other_agents: List[Agent] = []

            for agent in pop.agents.values():
                assigned = False
                for g in clone_only_groups:
                    if _agent_id_belongs_to_group(agent.id, g.id):
                        anchor_members[g.id].append(agent)
                        assigned = True
                        break
                if not assigned:
                    other_agents.append(agent)

            new_groups: List[Group] = []
            # Preserve existing group order for anchors; keep original metadata.
            for g in previous_groups:
                members = anchor_members.get(g.id)
                if members:
                    new_groups.append(
                        Group(
                            id=g.id,
                            name=g.name,
                            agents=members,
                            source_group_id=g.source_group_id,
                            source_group_name=g.source_group_name,
                            color=g.color,
                        )
                    )

            # Remaining agents (if any) become the combined League group.
            if other_agents:
                other_pop = Population()
                for a in other_agents:
                    other_pop.add(a)
                new_groups.append(_population_to_single_group(other_pop, generation_index))

            # Fallback: if, for some reason, no agents were assignable (e.g. all
            # clone-only groups disappeared), keep legacy behaviour.
            if not new_groups:
                self._state.groups = [_population_to_single_group(pop, generation_index)]
            else:
                self._state.groups = new_groups

        self._state.generation_index = generation_index
        self._state.last_summary = summary

    def get_league_ui(self) -> Dict[str, object]:
        """Return UI state (checkboxes, next-gen, export, HOF) for persistence."""
        return {
            "num_generations": self._spin_generations.value(),
            "export_when_index": self._combo_export_when.currentIndex(),
            "export_every_n": self._spin_export_every_n.value(),
            "export_what_index": self._combo_export_what.currentIndex(),
            "hof_when_index": self._combo_hof_when.currentIndex(),
            "hof_every_n": self._spin_hof_every_n.value(),
            "hof_what_index": self._combo_hof_what.currentIndex(),
            "hof_top_n": self._spin_hof_top_n.value(),
            "elo_tuning_checked": self._cb_elo_tuning.isChecked(),
            "elo_normalize_checked": self._cb_elo_normalize.isChecked(),
            "ppo_checked": self._cb_ppo.isChecked(),
        }

    def set_league_ui(self, ui: Dict[str, object]) -> None:
        """Restore UI state from saved league_ui."""
        if not ui:
            return
        if "num_generations" in ui:
            self._spin_generations.setValue(int(ui["num_generations"]))
        if "export_when_index" in ui:
            idx = int(ui["export_when_index"])
            if 0 <= idx < self._combo_export_when.count():
                self._combo_export_when.setCurrentIndex(idx)
        if "export_every_n" in ui:
            self._spin_export_every_n.setValue(int(ui["export_every_n"]))
        if "export_what_index" in ui:
            idx = int(ui["export_what_index"])
            if 0 <= idx < self._combo_export_what.count():
                self._combo_export_what.setCurrentIndex(idx)
        if "hof_when_index" in ui:
            idx = int(ui["hof_when_index"])
            if 0 <= idx < self._combo_hof_when.count():
                self._combo_hof_when.setCurrentIndex(idx)
        if "hof_every_n" in ui:
            self._spin_hof_every_n.setValue(int(ui["hof_every_n"]))
        if "hof_what_index" in ui:
            idx = int(ui["hof_what_index"])
            if 0 <= idx < self._combo_hof_what.count():
                self._combo_hof_what.setCurrentIndex(idx)
        if "hof_top_n" in ui:
            self._spin_hof_top_n.setValue(int(ui["hof_top_n"]))
        if "elo_tuning_checked" in ui:
            self._cb_elo_tuning.setChecked(bool(ui["elo_tuning_checked"]))
        if "elo_normalize_checked" in ui:
            self._cb_elo_normalize.setChecked(bool(ui["elo_normalize_checked"]))
        if "ppo_checked" in ui:
            self._cb_ppo.setChecked(bool(ui["ppo_checked"]))
        self._update_tour_elo_enabled()
        self._update_tour_ppo_enabled()

    def get_league_config(self) -> LeagueConfig:
        """Build LeagueConfig from current widget values."""
        style = "elo" if self._combo_league_style.currentText() == "ELO-based" else "random"
        player_count = int(self._combo_player_count.currentData() or 4)
        slots, sexual_n, clone, mutate = self._get_repro_counts()
        # Enforce that reproduction counts completely fill GA slots when GA is enabled.
        if slots > 0:
            total = sexual_n + clone + mutate
            if total != slots:
                raise ValueError(
                    f"Reproduction counts must fill all GA slots: clone({clone}) + mutate({mutate}) + sexual({sexual_n}) = {total}, "
                    f"but GA-eligible slots = {slots}."
                )
        trait_prob = self._spin_trait_prob.value() / 100.0
        elo_k = self._spin_elo_k.value() if self._cb_elo_tuning.isChecked() else 32.0
        elo_margin = self._spin_elo_margin.value() if self._cb_elo_tuning.isChecked() else 50.0
        ppo_top = self._spin_ppo_top_k.value() if self._cb_ppo.isChecked() else 0
        ppo_updates = self._spin_ppo_updates.value() if self._cb_ppo.isChecked() else 0
        pop_size = max(1, self._state.total_agents())
        return LeagueConfig(
            player_count=player_count,
            deals_per_match=self._spin_deals.value(),
            rounds_per_generation=self._spin_matches.value(),
            matchmaking_style=style,
            elo_k_factor=elo_k,
            elo_margin_scale=elo_margin,
            normalize_elo_mean=self._cb_elo_normalize.isChecked(),
            ppo_top_k=ppo_top,
            ppo_updates_per_agent=ppo_updates,
            fitness_elo_a=self._spin_fitness_a.value(),
            fitness_elo_b=self._spin_fitness_b.value(),
            fitness_avg_c=self._spin_fitness_c.value(),
            fitness_avg_d=self._spin_fitness_d.value(),
            ga_config=GAConfig(
                population_size=pop_size,
                sexual_offspring_count=sexual_n,
                mutate_count=mutate,
                clone_count=clone,
                sexual_parent_with_replacement=self._sexual_parent_with_replacement,
                sexual_parent_fitness_weighted=self._sexual_parent_fitness_weighted,
                sexual_trait_combination=self._sexual_trait_combination,
                mutation_prob=trait_prob,
                mutation_std=self._spin_mut_std.value(),
            ),
        )

    def set_league_config(self, cfg: LeagueConfig) -> None:
        """Set form values from LeagueConfig."""
        self._combo_league_style.setCurrentText("ELO-based" if cfg.matchmaking_style == "elo" else "Random")
        for i in range(self._combo_player_count.count()):
            if self._combo_player_count.itemData(i) == cfg.player_count:
                self._combo_player_count.setCurrentIndex(i)
                break
        self._spin_deals.setValue(cfg.deals_per_match)
        self._spin_matches.setValue(cfg.rounds_per_generation)
        self._spin_elo_k.setValue(cfg.elo_k_factor)
        self._spin_elo_margin.setValue(cfg.elo_margin_scale)
        # Default to True if field missing (backward compatibility with old configs)
        self._cb_elo_normalize.setChecked(getattr(cfg, "normalize_elo_mean", True))
        self._spin_ppo_top_k.setValue(cfg.ppo_top_k)
        self._spin_ppo_updates.setValue(cfg.ppo_updates_per_agent)
        self._spin_fitness_a.setValue(cfg.fitness_elo_a)
        self._spin_fitness_b.setValue(cfg.fitness_elo_b)
        self._spin_fitness_c.setValue(cfg.fitness_avg_c)
        self._spin_fitness_d.setValue(cfg.fitness_avg_d)
        if cfg.ga_config:
            ga = cfg.ga_config
            slots = self._get_ga_slots()
            if getattr(ga, "sexual_offspring_count", None) is not None and getattr(ga, "mutate_count", None) is not None and getattr(ga, "clone_count", None) is not None:
                sexual_n = max(0, min(slots, ga.sexual_offspring_count))
                clone = max(0, min(slots - sexual_n, ga.clone_count))
                mutate = slots - sexual_n - clone
                self._sexual_parent_with_replacement = getattr(ga, "sexual_parent_with_replacement", True)
                self._sexual_parent_fitness_weighted = getattr(ga, "sexual_parent_fitness_weighted", True)
                self._sexual_trait_combination = getattr(ga, "sexual_trait_combination", "average") or "average"
            else:
                ef = ga.elite_fraction
                ecf = getattr(ga, "elite_clone_fraction", 0)
                kept_elite = max(0, min(slots, int(round(slots * ef))))
                sexual_n = slots - kept_elite
                offspring = kept_elite
                clone = max(0, min(offspring, int(round(offspring * ecf)))) if offspring > 0 else 0
                mutate = offspring - clone
                self._sexual_parent_with_replacement = True
                self._sexual_parent_fitness_weighted = True
                self._sexual_trait_combination = "average"
            for spin in (self._spin_kept, self._spin_mutate, self._spin_clone):
                spin.blockSignals(True)
            try:
                self._spin_kept.setMaximum(max(0, slots))
                self._spin_mutate.setMaximum(max(0, slots))
                self._spin_clone.setMaximum(max(0, slots))
                self._spin_kept.setValue(sexual_n)
                self._spin_clone.setValue(clone)
                self._spin_mutate.setValue(mutate)
            finally:
                for spin in (self._spin_kept, self._spin_mutate, self._spin_clone):
                    spin.blockSignals(False)
            self._spin_trait_prob.setValue(ga.mutation_prob * 100)
            self._spin_mut_std.setValue(ga.mutation_std)
            self._update_ga_visual()

    def _update_project_label(self) -> None:
        p = self._state.project_path
        if p:
            self._label_project.setText(Path(p).name)
            self._label_project.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #e0e0e0; padding: 10px 8px;"
            )
            self._label_project.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
        else:
            self._label_project.setText("No Project Loaded")
            self._label_project.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #c0c0c0; padding: 12px 8px;"
            )
            self._label_project.setAlignment(
                QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignVCenter
            )
        self._update_content_visibility()

    def _update_content_visibility(self) -> None:
        """Show/hide league content, File vs New/Open buttons, and enable/disable menu actions."""
        has_project = self._state.project_path is not None
        self._content_container.setVisible(has_project)
        self._btns_no_project.setVisible(not has_project)
        self._no_project_row.setVisible(not has_project)
        self._btn_file.setVisible(has_project)
        self._btn_save.setVisible(has_project)
        self._btn_unsaved_warning.setVisible(has_project and self._league_params_dirty)
        self._act_save.setEnabled(has_project)
        self._act_save_as.setEnabled(has_project)
        self._act_export.setEnabled(has_project)

        vh = max(self._scroll.viewport().height(), 400) if hasattr(self, "_scroll") and self._scroll else 400
        one_third = max(200, vh // 3)

        if has_project:
            # Project loaded: project bar at top, then full content
            if self._no_project_center.layout().count() > 0:
                self._no_project_center.layout().removeWidget(self._project_group)
                self._project_group.setParent(None)
                self._project_view_layout.insertWidget(0, self._project_group)
            self._project_group.setFixedHeight(self._project_height_when_loaded)
            self._stack.setCurrentWidget(self._project_view)
        else:
            # No project: centered project box, ~1/3 viewport height
            if self._project_view_layout.indexOf(self._project_group) >= 0:
                self._project_view_layout.removeWidget(self._project_group)
                self._project_group.setParent(None)
                self._no_project_center.layout().addWidget(self._project_group)
            self._no_project_center.setMinimumHeight(one_third)
            self._project_group.setFixedHeight(one_third)
            self._stack.setCurrentWidget(self._no_project_view)

        # Option C: content width always follows viewport (handled by _update_league_content_size).
        if hasattr(self, "_update_league_content_size") and self._update_league_content_size is not None:
            self._update_league_content_size()
        if hasattr(self, "_scroll") and self._scroll is not None and not has_project:
            self._scroll.horizontalScrollBar().setValue(0)
            self._scroll.verticalScrollBar().setValue(0)

    def _groups_tuples(self) -> List[tuple]:
        return [
            (g.id, g.name, g.agents, g.source_group_id, g.source_group_name, g.color)
            for g in self._state.groups
        ]

    def _on_new_project(self) -> None:
        base = get_projects_folder()
        if not base:
            QtWidgets.QMessageBox.warning(
                self,
                "Projects folder not set",
                "No projects folder is configured.\n\n"
                "Go to the Settings tab, set a Projects folder, then try again.",
            )
            return
        base_path = Path(base)
        if not base_path.exists():
            try:
                base_path.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Projects folder unavailable",
                    "The projects folder could not be used:\n"
                    f"{base}\n\n"
                    f"{e}\n\n"
                    "Please go to the Settings tab, choose a valid Projects folder, "
                    "and try again.",
                )
                return
        dlg = NewProjectDialog(str(base_path), self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        path = dlg.result_path()
        if not path:
            return
        # Create new project: clear state, set path, save initial empty state
        self._state.groups = []
        self._state.project_path = path
        self._state.generation_index = 0
        self._state.last_summary = None
        self._state.hof_agents = []
        try:
            project_save(
                path,
                groups=[],
                league_config=self.get_league_config(),
                generation_index=0,
                last_summary=None,
                league_ui=self.get_league_ui(),
                hof_agents=[],
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Create failed", str(e))
            return
        self._clear_league_params_dirty()
        self._refresh_table()
        self._update_project_label()
        self._update_content_visibility()
        self.project_path_changed.emit(path)
        QtWidgets.QMessageBox.information(self, "New Project", f"Created and opened project: {Path(path).name}")

    def open_project(self, path: str, *, show_message: bool = False) -> bool:
        """Open a project by path. Returns True if successful."""
        if not Path(path).exists():
            return False
        try:
            data = project_load(path)
        except Exception as e:
            if show_message:
                QtWidgets.QMessageBox.critical(self, "Open failed", str(e))
            return False
        self._load_project_data(data)
        self._state.project_path = path
        self._update_project_label()
        self._refresh_table()
        self.project_path_changed.emit(path)
        if show_message:
            QtWidgets.QMessageBox.information(self, "Open", f"Loaded project from {path}")
        return True

    def _on_open_project(self) -> None:
        base = get_projects_folder()
        if not base:
            QtWidgets.QMessageBox.warning(
                self,
                "Projects folder not set",
                "No projects folder is configured.\n\n"
                "Go to the Settings tab, set a Projects folder, then try again.",
            )
            return
        base_path = Path(base)
        if not base_path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Projects folder unavailable",
                "The projects folder does not exist:\n"
                f"{base}\n\n"
                "Please go to the Settings tab, choose a valid Projects folder, "
                "and try again.",
            )
            return
        dlg = NewProjectDialog(str(base_path), self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        path = dlg.result_path()
        if path:
            self.open_project(path, show_message=True)

    def _load_project_data(self, data: dict) -> None:
        groups_data = data["groups_data"]
        project_dir = data.get("project_dir")
        self._state.groups = []
        for t in groups_data:
            gid, gname, agents, src_id, src_name, color = t
            if project_dir:
                for a in agents:
                    if a.checkpoint_path and not Path(a.checkpoint_path).is_absolute():
                        a.checkpoint_path = str((Path(project_dir) / a.checkpoint_path).resolve())
            self._state.groups.append(
                Group(
                    id=gid,
                    name=gname,
                    agents=agents,
                    source_group_id=src_id,
                    source_group_name=src_name,
                    color=color,
                )
            )
        self.set_league_config(data["league_config"])
        self.set_league_ui(data.get("league_ui") or {})
        self._state.generation_index = data["generation_index"]
        self._state.last_summary = data.get("last_summary")
        self._state.hof_agents = data.get("hof_agents", [])
        self._clear_league_params_dirty()

    def _on_save_project(self) -> None:
        path = self._state.project_path
        if not path:
            QtWidgets.QMessageBox.warning(self, "Save", "No project. Use New or Save As first.")
            return
        try:
            project_save(
                path,
                groups=self._groups_tuples(),
                league_config=self.get_league_config(),
                generation_index=self._state.generation_index,
                last_summary=self._state.last_summary,
                league_ui=self.get_league_ui(),
                hof_agents=self._state.hof_agents,
            )
            self._update_project_label()
            self._clear_league_params_dirty()
            QtWidgets.QMessageBox.information(self, "Save", f"Saved to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def _on_save_project_as(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Save As - choose directory")
        if not path:
            return
        self._state.project_path = path
        self._on_save_project()

    def _on_export_project_json(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export to JSON", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            logs = load_league_log(self._state.project_path) if self._state.project_path else None
            project_export_json(
                path,
                groups=self._groups_tuples(),
                league_config=self.get_league_config(),
                generation_index=self._state.generation_index,
                last_summary=self._state.last_summary,
                logs=logs,
                project_dir=self._state.project_path,
                league_ui=self.get_league_ui(),
                hof_agents=self._state.hof_agents,
            )
            QtWidgets.QMessageBox.information(self, "Export", f"Exported to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))

    def _on_import_project_json(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import from JSON", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            data = project_import_json(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import failed", str(e))
            return
        self._load_project_data(data)
        self._state.project_path = str(Path(path).parent)  # Use JSON dir as project context
        self._update_project_label()
        self._refresh_table()
        QtWidgets.QMessageBox.information(self, "Import", f"Imported from {path}")


def make_league_tab() -> QtWidgets.QWidget:
    """Create the League tab widget."""
    return LeagueTabWidget()
