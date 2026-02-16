"""
League tab: population management (groups of agents), league structure, GA parameters, run controls.

Groups contain agents; the main table shows one row per group. Expand opens a detail dialog.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
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
from tarot_gui.project_dialog import NewProjectDialog
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
from tarot.league import LeagueConfig
from tarot.persistence import population_from_dict
from tarot.population_helpers import clone_agents, generate_random_agents, mutate_from_base
from tarot.tournament import Agent, Population


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


@dataclass
class LeagueTabState:
    """State for the League tab. Groups hold agents; population is built from groups."""

    groups: List[Group] = field(default_factory=list)
    run_status: RunStatus = RunStatus.IDLE
    last_summary: Optional[Dict[str, float]] = None
    project_path: Optional[str] = None
    generation_index: int = 0

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


class RunSectionWidget(QtWidgets.QWidget):
    """Run controls (Start, Pause, Cancel) and ELO metrics. Placed in Dashboard tab."""

    def __init__(self, state: LeagueTabState, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._state = state
        layout = QtWidgets.QVBoxLayout(self)
        run_group = QtWidgets.QGroupBox("Run")
        run_layout = QtWidgets.QVBoxLayout(run_group)
        buttons_row = QtWidgets.QHBoxLayout()
        self._btn_start = QtWidgets.QPushButton("Start")
        self._btn_pause = QtWidgets.QPushButton("Pause at next generation")
        self._btn_cancel = QtWidgets.QPushButton("Cancel")
        self._btn_pause.setEnabled(False)
        self._btn_cancel.setEnabled(False)
        buttons_row.addWidget(self._btn_start)
        buttons_row.addWidget(self._btn_pause)
        buttons_row.addWidget(self._btn_cancel)
        buttons_row.addStretch(1)
        run_layout.addLayout(buttons_row)
        metrics_row = QtWidgets.QHBoxLayout()
        self._label_elo = QtWidgets.QLabel("ELO — min: — mean: — max: —")
        metrics_row.addWidget(self._label_elo)
        metrics_row.addStretch(1)
        run_layout.addLayout(metrics_row)
        layout.addWidget(run_group)

        # Charts placeholder for Phase 4
        charts_placeholder = QtWidgets.QLabel("Charts area (Phase 4)")
        charts_placeholder.setMinimumHeight(180)
        charts_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        charts_placeholder.setObjectName("dashboardChartsPlaceholder")
        charts_placeholder.setStyleSheet("QLabel#dashboardChartsPlaceholder { border: 1px solid #606060; }")
        layout.addWidget(charts_placeholder, stretch=1)

    def showEvent(self, event: QtCore.QEvent) -> None:
        super().showEvent(event)
        self.update_metrics()

    def update_metrics(self) -> None:
        """Refresh ELO label from state."""
        summary = self._state.last_summary
        if summary:
            mn = summary.get("elo_min", 0)
            mean = summary.get("elo_mean", 0)
            mx = summary.get("elo_max", 0)
            self._label_elo.setText(f"ELO — min: {mn:.0f}  mean: {mean:.0f}  max: {mx:.0f}")
        else:
            pop = self._state.build_population()
            elos = [a.elo_global for a in pop.agents.values()]
            if elos:
                mn, mx = min(elos), max(elos)
                mean = sum(elos) / len(elos)
                self._label_elo.setText(f"ELO — min: {mn:.0f}  mean: {mean:.0f}  max: {mx:.0f}")
            else:
                self._label_elo.setText("ELO — min: — mean: — max: —")


class LeagueTabWidget(QtWidgets.QWidget):
    """League Parameters tab: groups table, config sections, export."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._state = LeagueTabState()
        self._rng = random.Random()
        self._setup_ui()
        self._refresh_table()

    def _setup_ui(self) -> None:
        # Option 4: fixed layout for 1080p (1920×1080); scales to WQHD and above
        CONTENT_HEIGHT_1080P = 1000  # fits in 1080 minus window chrome
        _m = 8  # layout vertical margins
        PROJECT_HEIGHT = 84
        POPULATION_HEIGHT = 400
        ARROW_HEIGHT = 14
        FLOW_BOX_HEIGHT = CONTENT_HEIGHT_1080P - _m - PROJECT_HEIGHT - POPULATION_HEIGHT - ARROW_HEIGHT  # 526

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        content = QtWidgets.QWidget()
        content.setMinimumWidth(1920)
        content.setMinimumHeight(CONTENT_HEIGHT_1080P)
        layout = QtWidgets.QVBoxLayout(content)
        layout.setSpacing(0)
        layout.setContentsMargins(4, 4, 4, 4)

        proj_group = QtWidgets.QGroupBox("Project")
        proj_layout = QtWidgets.QHBoxLayout(proj_group)
        self._label_project = QtWidgets.QLabel("No Project Loaded")
        self._label_project.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._label_project.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #aaa; padding: 16px 0;"
        )
        proj_layout.addWidget(self._label_project, stretch=1)
        # No-project buttons (visible when no project loaded)
        self._btns_no_project = QtWidgets.QWidget()
        no_proj_layout = QtWidgets.QHBoxLayout(self._btns_no_project)
        no_proj_layout.setContentsMargins(0, 0, 0, 0)
        btn_new = QtWidgets.QPushButton("New Project")
        btn_new.clicked.connect(self._on_new_project)
        btn_open = QtWidgets.QPushButton("Open Project")
        btn_open.clicked.connect(self._on_open_project)
        no_proj_layout.addWidget(btn_new)
        no_proj_layout.addWidget(btn_open)
        proj_layout.addWidget(self._btns_no_project)
        # File button (visible only when project loaded)
        self._btn_file = QtWidgets.QPushButton("File")
        self._btn_file.setFixedWidth(80)
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
        proj_layout.addWidget(self._btn_file)
        proj_group.setFixedHeight(PROJECT_HEIGHT)
        layout.addWidget(proj_group)

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
        filter_row.addWidget(QtWidgets.QLabel("Group by:"))
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
        tools_row.addWidget(QtWidgets.QLabel("Count:"))
        self._spin_add_random = QtWidgets.QSpinBox()
        self._spin_add_random.setRange(1, 999)
        self._spin_add_random.setValue(4)
        tools_row.addWidget(self._spin_add_random)
        btn_add_random = QtWidgets.QPushButton("Add random")
        btn_add_random.clicked.connect(self._on_add_random)
        tools_row.addWidget(btn_add_random)
        btn_import = QtWidgets.QPushButton("Import")
        btn_import.clicked.connect(self._on_import)
        tools_row.addWidget(btn_import)
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
        self._table.horizontalHeader().setMinimumSectionSize(80)
        self._table.verticalHeader().setDefaultSectionSize(34)
        self._table.horizontalHeader().setSectionResizeMode(
            GRP_COL_ELO, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        right_layout.addWidget(self._table, stretch=1)
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

        # Flow blocks: Tournament | Fitness | Elites+Parents | Mutation/Clone | Next Gen (same row)
        flow_row = QtWidgets.QHBoxLayout()
        flow_row.setSpacing(12)

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
        self._combo_player_count.currentIndexChanged.connect(self._update_tournament_insights)
        self._combo_league_style = QtWidgets.QComboBox()
        self._combo_league_style.addItems(["ELO-based", "Random"])
        self._combo_league_style.setItemData(0, "Match agents of similar ELO strength.", QtCore.Qt.ItemDataRole.ToolTipRole)
        self._combo_league_style.setItemData(1, "Shuffle agents randomly.", QtCore.Qt.ItemDataRole.ToolTipRole)
        core_row1 = QtWidgets.QHBoxLayout()
        core_row1.addWidget(QtWidgets.QLabel("Rules:"))
        core_row1.addWidget(self._combo_player_count)
        core_row1.addSpacing(12)
        core_row1.addWidget(QtWidgets.QLabel("Matchmaking:"))
        core_row1.addWidget(self._combo_league_style)
        core_row1.addStretch()
        core_layout.addRow(core_row1)
        self._spin_deals = QtWidgets.QSpinBox()
        self._spin_deals.setRange(1, 99)
        self._spin_deals.setValue(5)
        self._spin_deals.valueChanged.connect(self._update_tournament_insights)
        self._spin_matches = QtWidgets.QSpinBox()
        self._spin_matches.setRange(1, 999)
        self._spin_matches.setValue(3)
        self._spin_matches.valueChanged.connect(self._update_tournament_insights)
        core_row2 = QtWidgets.QHBoxLayout()
        core_row2.addWidget(QtWidgets.QLabel("Deals/match:"))
        core_row2.addWidget(self._spin_deals)
        core_row2.addSpacing(12)
        core_row2.addWidget(QtWidgets.QLabel("Matches/gen:"))
        core_row2.addWidget(self._spin_matches)
        core_row2.addStretch()
        core_layout.addRow(core_row2)
        self._combo_player_count.setToolTip("Game rules: 3, 4, or 5 players per table (FFT rules).")
        self._combo_league_style.setToolTip("ELO-based: pair similar-strength agents. Random: random table assignment.")
        self._spin_deals.setToolTip("Number of deals played in each match. More deals give more stable ELO updates.")
        self._spin_matches.setToolTip("Number of tournament rounds per generation. More rounds refine ELO rankings.")
        tour_layout.addWidget(tour_core)

        # ELO tuning: checkbox (unchecked by default), params always visible but greyed when unchecked
        self._cb_elo_tuning = QtWidgets.QCheckBox("ELO tuning")
        self._cb_elo_tuning.setChecked(False)
        self._cb_elo_tuning.toggled.connect(self._update_tour_elo_enabled)
        tour_layout.addWidget(self._cb_elo_tuning)
        self._tour_elo_frame = QtWidgets.QFrame()
        self._tour_elo_frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._tour_elo_frame.setEnabled(False)
        elo_layout = QtWidgets.QHBoxLayout(self._tour_elo_frame)
        self._spin_elo_k = QtWidgets.QDoubleSpinBox()
        self._spin_elo_k.setRange(1, 100)
        self._spin_elo_k.setValue(32)
        self._spin_elo_k.setToolTip("Max ELO change per pairwise comparison. Higher = faster rating changes.")
        self._spin_elo_margin = QtWidgets.QDoubleSpinBox()
        self._spin_elo_margin.setRange(1, 200)
        self._spin_elo_margin.setValue(50)
        self._spin_elo_margin.setToolTip("Scale for score-diff to result mapping. Affects how big wins influence ELO.")
        elo_layout.addWidget(QtWidgets.QLabel("ELO K-factor:"))
        elo_layout.addWidget(self._spin_elo_k)
        elo_layout.addSpacing(12)
        elo_layout.addWidget(QtWidgets.QLabel("ELO margin scale:"))
        elo_layout.addWidget(self._spin_elo_margin)
        elo_layout.addStretch()
        tour_layout.addWidget(self._tour_elo_frame)

        # PPO fine-tuning: checkbox (unchecked by default), params always visible but greyed when unchecked
        self._cb_ppo = QtWidgets.QCheckBox("PPO fine-tuning")
        self._cb_ppo.setChecked(False)
        self._cb_ppo.toggled.connect(self._update_tour_ppo_enabled)
        tour_layout.addWidget(self._cb_ppo)
        self._tour_ppo_frame = QtWidgets.QFrame()
        self._tour_ppo_frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._tour_ppo_frame.setEnabled(False)
        ppo_layout = QtWidgets.QHBoxLayout(self._tour_ppo_frame)
        self._spin_ppo_top_k = QtWidgets.QSpinBox()
        self._spin_ppo_top_k.setRange(0, 999)
        self._spin_ppo_top_k.setValue(0)
        self._spin_ppo_top_k.setToolTip("0 = disabled. Top-K agents by fitness get PPO fine-tuning each generation.")
        self._spin_ppo_updates = QtWidgets.QSpinBox()
        self._spin_ppo_updates.setRange(0, 9999)
        self._spin_ppo_updates.setValue(0)
        self._spin_ppo_updates.setToolTip("Number of PPO update steps per agent when top-K > 0.")
        ppo_layout.addWidget(QtWidgets.QLabel("PPO top-K:"))
        ppo_layout.addWidget(self._spin_ppo_top_k)
        ppo_layout.addSpacing(12)
        ppo_layout.addWidget(QtWidgets.QLabel("PPO updates/agent:"))
        ppo_layout.addWidget(self._spin_ppo_updates)
        ppo_layout.addStretch()
        tour_layout.addWidget(self._tour_ppo_frame)

        self._tour_insights = QtWidgets.QLabel("Tables/round: —  Matches/gen: —  Deals/agent: —")
        self._tour_insights.setWordWrap(True)
        self._tour_insights.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        tour_layout.addWidget(self._tour_insights)
        tour_group.setFixedHeight(FLOW_BOX_HEIGHT)
        flow_row.addWidget(tour_group, 1)

        # Fitness block: Weights (core, no checkbox) | Selection (checkbox)
        fit_group = QtWidgets.QGroupBox("Fitness")
        fit_layout = QtWidgets.QVBoxLayout(fit_group)
        # Weights: paired on same line (core, always visible)
        self._spin_weight_elo = QtWidgets.QDoubleSpinBox()
        self._spin_weight_elo.setRange(0, 10)
        self._spin_weight_elo.setValue(1.0)
        self._spin_weight_elo.setDecimals(2)
        self._spin_weight_elo.setSingleStep(0.1)
        self._spin_weight_elo.setToolTip("Multiplier for global ELO in fitness. Higher = ELO dominates fitness.")
        self._spin_weight_avg_score = QtWidgets.QDoubleSpinBox()
        self._spin_weight_avg_score.setRange(0, 10)
        self._spin_weight_avg_score.setValue(0.0)
        self._spin_weight_avg_score.setDecimals(2)
        self._spin_weight_avg_score.setSingleStep(0.1)
        self._spin_weight_avg_score.setToolTip("Multiplier for average match score in fitness. Use to balance ELO vs raw score.")
        fit_row_weights = QtWidgets.QHBoxLayout()
        fit_row_weights.addWidget(QtWidgets.QLabel("Weight (global ELO):"))
        fit_row_weights.addWidget(self._spin_weight_elo)
        fit_row_weights.addSpacing(12)
        fit_row_weights.addWidget(QtWidgets.QLabel("Weight (avg score):"))
        fit_row_weights.addWidget(self._spin_weight_avg_score)
        fit_row_weights.addStretch()
        fit_layout.addLayout(fit_row_weights)
        self._fitness_formula = QtWidgets.QLabel("fitness = 1.0 × ELO + 0.0 × avg_score")
        self._fitness_formula.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        self._fitness_formula.setWordWrap(True)
        self._fitness_formula.setMaximumHeight(22)
        for spin in (self._spin_weight_elo, self._spin_weight_avg_score):
            spin.valueChanged.connect(self._update_fitness_formula)
        fit_layout.addWidget(self._fitness_formula)
        self._fitness_visual = FitnessVisualWidget()
        fit_layout.addWidget(self._fitness_visual)
        fit_group.setFixedHeight(FLOW_BOX_HEIGHT)
        flow_row.addWidget(fit_group, 1)

        # Reproduction block: Elite %, Clone %, Mutation %, Mut std, distribution graph
        MUT_TOP_ROW_HEIGHT = 42
        mut_group = QtWidgets.QGroupBox("Reproduction")
        mut_group.setFixedHeight(FLOW_BOX_HEIGHT)
        mut_layout = QtWidgets.QVBoxLayout(mut_group)
        mut_layout.setSpacing(14)
        mut_layout.setContentsMargins(10, 8, 10, 10)

        def _rep_compact_row(w: QtWidgets.QWidget) -> QtWidgets.QFrame:
            f = QtWidgets.QFrame()
            f.setFixedHeight(MUT_TOP_ROW_HEIGHT)
            f.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            lo = QtWidgets.QVBoxLayout(f)
            lo.setContentsMargins(0, 0, 0, 0)
            lo.addWidget(w)
            return f

        # 1. Elite, Clone, Mutation on one line (must sum to 100%)
        params_row = QtWidgets.QHBoxLayout()
        lbl_elite = QtWidgets.QLabel("Elite:")
        lbl_elite.setToolTip("Fraction of population filled by offspring (mutated + cloned).")
        self._spin_elite = QtWidgets.QDoubleSpinBox()
        self._spin_elite.setRange(0, 100)
        self._spin_elite.setValue(90)
        self._spin_elite.setSuffix(" %")
        self._spin_elite.setDecimals(1)
        self._spin_elite.setMinimumWidth(72)
        self._spin_elite.valueChanged.connect(self._on_repro_params_changed)
        lbl_clone = QtWidgets.QLabel("Cloned:")
        lbl_clone.setToolTip("Fraction of population filled by cloning elites.")
        self._spin_clone = QtWidgets.QDoubleSpinBox()
        self._spin_clone.setRange(0, 100)
        self._spin_clone.setValue(10)
        self._spin_clone.setSuffix(" %")
        self._spin_clone.setDecimals(1)
        self._spin_clone.setMinimumWidth(72)
        self._spin_clone.valueChanged.connect(self._on_repro_params_changed)
        lbl_mut_pct = QtWidgets.QLabel("Mutated:")
        lbl_mut_pct.setToolTip("Fraction of population filled by mutated offspring.")
        self._spin_mut_prob = QtWidgets.QDoubleSpinBox()
        self._spin_mut_prob.setRange(0, 100)
        self._spin_mut_prob.setValue(80)
        self._spin_mut_prob.setSuffix(" %")
        self._spin_mut_prob.setDecimals(1)
        self._spin_mut_prob.setMinimumWidth(72)
        self._spin_mut_prob.valueChanged.connect(self._on_repro_params_changed)
        params_row.addWidget(lbl_elite)
        params_row.addWidget(self._spin_elite)
        params_row.addSpacing(16)
        params_row.addWidget(lbl_mut_pct)
        params_row.addWidget(self._spin_mut_prob)
        params_row.addSpacing(16)
        params_row.addWidget(lbl_clone)
        params_row.addWidget(self._spin_clone)
        params_row.addStretch()
        params_w = QtWidgets.QWidget()
        params_w.setLayout(params_row)
        mut_layout.addWidget(_rep_compact_row(params_w), 0)
        # 2. Selection bar with color legend
        self._reproduction_bar_widget = ReproductionBarWidget()
        mut_layout.addWidget(self._reproduction_bar_widget, 0)
        self._elites_insights = QtWidgets.QLabel("Elite count: —  Clone slots: —  Mutate slots: —")
        self._elites_insights.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        self._elites_insights.setWordWrap(True)
        mut_layout.addWidget(self._elites_insights, 0)
        # 3. Mut. std and Trait prob (mutation probability when mutating)
        mut_std_row = QtWidgets.QHBoxLayout()
        lbl_mut_std = QtWidgets.QLabel("Mut. std:")
        lbl_mut_std.setToolTip("Standard deviation of the Gaussian used to perturb traits.")
        self._spin_mut_std = QtWidgets.QDoubleSpinBox()
        self._spin_mut_std.setRange(0.01, 1.0)
        self._spin_mut_std.setValue(0.1)
        self._spin_mut_std.setSingleStep(0.05)
        self._spin_mut_std.setMinimumWidth(72)
        lbl_trait_prob = QtWidgets.QLabel("Trait prob:")
        lbl_trait_prob.setToolTip("Probability that a trait is perturbed when creating a mutant offspring.")
        self._spin_trait_prob = QtWidgets.QDoubleSpinBox()
        self._spin_trait_prob.setRange(0, 100)
        self._spin_trait_prob.setValue(50)
        self._spin_trait_prob.setSuffix(" %")
        self._spin_trait_prob.setDecimals(1)
        self._spin_trait_prob.setMinimumWidth(72)
        mut_std_row.addWidget(lbl_mut_std)
        mut_std_row.addWidget(self._spin_mut_std)
        mut_std_row.addSpacing(16)
        mut_std_row.addWidget(lbl_trait_prob)
        mut_std_row.addWidget(self._spin_trait_prob)
        mut_std_row.addStretch()
        mut_std_w = QtWidgets.QWidget()
        mut_std_w.setLayout(mut_std_row)
        mut_layout.addWidget(_rep_compact_row(mut_std_w), 0)
        # 4. Mutation distribution graph
        self._mut_dist_widget = MutationDistWidget()
        self._mut_dist_widget.setMinimumHeight(180)
        self._mut_dist_widget.setMaximumHeight(280)
        mut_layout.addWidget(self._mut_dist_widget, 0)
        mut_layout.addStretch(1)
        for spin in (self._spin_elite, self._spin_clone, self._spin_mut_prob, self._spin_mut_std, self._spin_trait_prob):
            spin.valueChanged.connect(self._update_ga_visual)
        flow_row.addWidget(mut_group, 1)

        # Next Generation block: Run (core, no checkbox) | Export (checkbox, params greyed when unchecked)
        next_group = QtWidgets.QGroupBox("Next Generation")
        next_layout = QtWidgets.QVBoxLayout(next_group)
        # Run: core, always visible
        self._spin_generations = QtWidgets.QSpinBox()
        self._spin_generations.setRange(1, 9999)
        self._spin_generations.setValue(10)
        self._spin_generations.setToolTip("Total number of generations to run.")
        run_form = QtWidgets.QFormLayout()
        run_form.addRow(QtWidgets.QLabel("Generations:"), self._spin_generations)
        next_layout.addLayout(run_form)
        # Export: checkbox (checked by default), params always visible but greyed when unchecked
        self._cb_export = QtWidgets.QCheckBox("Export")
        self._cb_export.setChecked(True)
        self._cb_export.toggled.connect(self._update_next_gen_export_enabled)
        next_layout.addWidget(self._cb_export)
        self._next_export_frame = QtWidgets.QFrame()
        self._next_export_frame.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        export_layout = QtWidgets.QVBoxLayout(self._next_export_frame)
        self._combo_export_when = QtWidgets.QComboBox()
        self._combo_export_when.addItems(["On demand only", "Every generation", "Every N generations"])
        self._combo_export_when.setItemData(0, "Export manually.", QtCore.Qt.ItemDataRole.ToolTipRole)
        self._combo_export_when.setItemData(1, "Auto-export after each generation.", QtCore.Qt.ItemDataRole.ToolTipRole)
        self._combo_export_when.setItemData(2, "Auto-export every N generations.", QtCore.Qt.ItemDataRole.ToolTipRole)
        self._spin_export_every_n = QtWidgets.QSpinBox()
        self._spin_export_every_n.setRange(1, 999)
        self._spin_export_every_n.setValue(5)
        self._spin_export_every_n.setEnabled(False)
        def _on_export_when_changed() -> None:
            if self._cb_export.isChecked():
                self._spin_export_every_n.setEnabled(
                    self._combo_export_when.currentIndex() == 2
                )
        self._combo_export_when.currentIndexChanged.connect(_on_export_when_changed)
        export_row = QtWidgets.QHBoxLayout()
        export_row.addWidget(QtWidgets.QLabel("Export when:"))
        export_row.addWidget(self._combo_export_when)
        export_row.addSpacing(12)
        export_row.addWidget(QtWidgets.QLabel("Every N gens:"))
        export_row.addWidget(self._spin_export_every_n)
        export_row.addStretch()
        export_layout.addLayout(export_row)
        self._combo_export_what = QtWidgets.QComboBox()
        self._combo_export_what.addItems(["Full population", "Top N by ELO", "GA-eligible only"])
        export_form_row = QtWidgets.QFormLayout()
        export_form_row.addRow(QtWidgets.QLabel("Export what:"), self._combo_export_what)
        export_layout.addLayout(export_form_row)
        btn_export = QtWidgets.QPushButton("Export now")
        btn_export.clicked.connect(self._on_export_now)
        export_layout.addWidget(btn_export)
        next_layout.addWidget(self._next_export_frame)
        self._next_gen_insights = QtWidgets.QLabel("Population size: —  Composition: —")
        self._next_gen_insights.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")
        self._next_gen_insights.setWordWrap(True)
        next_layout.addWidget(self._next_gen_insights)
        next_group.setFixedHeight(FLOW_BOX_HEIGHT)
        flow_row.addWidget(next_group, 1)

        flow_container = QtWidgets.QWidget()
        flow_container.setLayout(flow_row)
        content_layout.addWidget(flow_container, 1)

        self._content_container.setMinimumHeight(CONTENT_HEIGHT_1080P)
        layout.addWidget(self._content_container, 0)
        layout.addStretch(1)

        scroll.setWidget(content)
        # Fill viewport when taller than content so no grey strip at bottom
        _min_content_h = CONTENT_HEIGHT_1080P
        def _update_content_min_height():
            vh = scroll.viewport().height()
            content.setMinimumHeight(max(_min_content_h, vh))
        _update_content_min_height()
        scroll.viewport().installEventFilter(
            _ResizeFilter(scroll, lambda: _update_content_min_height())
        )

        self._update_ga_visual()
        self._update_pie_chart()
        self._update_tournament_insights()
        self._update_fitness_formula()
        self._update_elites_insights()
        self._update_next_gen_insights()
        self._update_tour_elo_enabled()
        self._update_tour_ppo_enabled()
        self._update_next_gen_export_enabled()

        self._update_content_visibility()

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _refresh_table(self) -> None:
        self._table.setRowCount(0)
        for row, group in enumerate(self._state.groups):
            self._table.insertRow(row)
            self._fill_group_row(row, group)
        # Ensure columns are wide enough for content
        self._table.setColumnWidth(GRP_COL_SELECT, max(36, self._table.columnWidth(GRP_COL_SELECT)))
        self._table.setColumnWidth(GRP_COL_COLOR, max(60, self._table.columnWidth(GRP_COL_COLOR)))
        self._table.setColumnWidth(GRP_COL_EXPAND, max(72, self._table.columnWidth(GRP_COL_EXPAND)))
        self._table.setColumnWidth(GRP_COL_ACTIONS, max(75, self._table.columnWidth(GRP_COL_ACTIONS)))
        self._table.setColumnWidth(GRP_COL_ELO, max(160, self._table.columnWidth(GRP_COL_ELO)))
        self._update_player_count_options()
        self._update_pie_chart()
        if hasattr(self, "_update_elites_insights"):
            self._update_elites_insights()
        if hasattr(self, "_update_next_gen_insights"):
            self._update_next_gen_insights()
        if hasattr(self, "_update_tournament_insights"):
            self._update_tournament_insights()

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

    def _on_repro_params_changed(self) -> None:
        """Enforce Elite = Mutated + Cloned. When Clone changes, adjust Mutated to keep Elite. Same for Mutated."""
        e = self._spin_elite.value()
        c = self._spin_clone.value()
        m = self._spin_mut_prob.value()
        for spin in (self._spin_elite, self._spin_clone, self._spin_mut_prob):
            spin.blockSignals(True)
        try:
            if abs((m + c) - e) < 0.01:
                self._update_elites_insights()
                self._update_ga_visual()
                return
            # Mutated + Cloned must equal Elite
            if m + c > e:
                # Need to reduce: take from Mutated first
                self._spin_mut_prob.setValue(max(0, e - c))
            else:
                # Need to increase: add to Mutated
                self._spin_mut_prob.setValue(min(e, e - c))
        finally:
            for spin in (self._spin_elite, self._spin_clone, self._spin_mut_prob):
                spin.blockSignals(False)
        self._update_elites_insights()
        self._update_ga_visual()

    def _update_ga_visual(self) -> None:
        ga_eligible = sum(1 for g in self._state.groups for a in g.agents if a.can_use_as_ga_parent)
        self._reproduction_bar_widget.set_params(
            self._spin_elite.value(),
            self._spin_clone.value(),
            self._spin_mut_prob.value(),
            total_agents=ga_eligible,
        )
        self._mut_dist_widget.set_mutation_std(self._spin_mut_std.value())

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

    def _update_next_gen_export_enabled(self) -> None:
        enabled = self._cb_export.isChecked()
        self._next_export_frame.setEnabled(enabled)
        for w in self._next_export_frame.findChildren(QtWidgets.QWidget):
            w.setEnabled(enabled)
        if enabled:
            self._spin_export_every_n.setEnabled(
                self._combo_export_when.currentIndex() == 2
            )

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
        w_elo = self._spin_weight_elo.value()
        w_score = self._spin_weight_avg_score.value()
        d_elo = self._spin_weight_elo.decimals()
        d_score = self._spin_weight_avg_score.decimals()
        self._fitness_formula.setText(
            f"fitness = {w_elo:.{d_elo}f} × ELO + {w_score:.{d_score}f} × avg_score"
        )
        self._fitness_visual.set_params(w_elo, w_score)

    def _update_elites_insights(self) -> None:
        total = self._state.total_agents()
        ga_eligible = sum(1 for g in self._state.groups for a in g.agents if a.can_use_as_ga_parent)
        ref = total - ga_eligible
        slots = max(0, total - ref)
        elite_pct = self._spin_elite.value() / 100
        clone_pct = self._spin_clone.value() / 100
        mut_pct = self._spin_mut_prob.value() / 100
        kept_count = max(1, int(slots * (1 - elite_pct))) if slots > 0 else 0
        offspring_slots = slots - kept_count
        clone_slots = int(offspring_slots * clone_pct / elite_pct) if elite_pct > 0 else 0
        mutate_slots = max(0, offspring_slots - clone_slots)
        self._elites_insights.setText(
            f"Elite count: {kept_count}  Clone slots: {clone_slots}  Mutate slots: {mutate_slots}"
        )

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
            from tarot.persistence import population_to_json
            with open(path, "w", encoding="utf-8") as f:
                f.write(population_to_json(pop))
            QtWidgets.QMessageBox.information(self, "Export", f"Exported to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))

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
        if not imported.agents:
            QtWidgets.QMessageBox.information(self, "Import", "File contains no agents.")
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
            name=f"Imported ({len(renamed)})",
            agents=renamed,
            source_group_name="Imported file",
            color=_pick_group_color(used_colors, self._rng),
        )

        if msg.clickedButton() == btn_replace:
            self._state.groups = [new_group]
        else:
            self._state.groups.append(new_group)
        self._refresh_table()

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

    def get_league_config(self) -> LeagueConfig:
        """Build LeagueConfig from current widget values."""
        style = "elo" if self._combo_league_style.currentText() == "ELO-based" else "random"
        player_count = int(self._combo_player_count.currentData() or 4)
        elite_pct = self._spin_elite.value()
        clone_pct = self._spin_clone.value()
        mut_pct = self._spin_mut_prob.value()
        elite_frac = 1.0 - elite_pct / 100.0
        elite_clone_frac = (
            clone_pct / elite_pct if elite_pct > 0 else 0.0
        )
        trait_prob = self._spin_trait_prob.value() / 100.0
        elo_k = self._spin_elo_k.value() if self._cb_elo_tuning.isChecked() else 32.0
        elo_margin = self._spin_elo_margin.value() if self._cb_elo_tuning.isChecked() else 50.0
        ppo_top = self._spin_ppo_top_k.value() if self._cb_ppo.isChecked() else 0
        ppo_updates = self._spin_ppo_updates.value() if self._cb_ppo.isChecked() else 0
        return LeagueConfig(
            player_count=player_count,
            deals_per_match=self._spin_deals.value(),
            rounds_per_generation=self._spin_matches.value(),
            matchmaking_style=style,
            elo_k_factor=elo_k,
            elo_margin_scale=elo_margin,
            ppo_top_k=ppo_top,
            ppo_updates_per_agent=ppo_updates,
            fitness_weight_global_elo=self._spin_weight_elo.value(),
            fitness_weight_avg_score=self._spin_weight_avg_score.value(),
            ga_config=GAConfig(
                population_size=max(1, self._state.total_agents()),
                elite_fraction=elite_frac,
                elite_clone_fraction=elite_clone_frac,
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
        self._spin_ppo_top_k.setValue(cfg.ppo_top_k)
        self._spin_ppo_updates.setValue(cfg.ppo_updates_per_agent)
        self._spin_weight_elo.setValue(cfg.fitness_weight_global_elo)
        self._spin_weight_avg_score.setValue(cfg.fitness_weight_avg_score)
        if cfg.ga_config:
            for spin in (self._spin_elite, self._spin_clone, self._spin_mut_prob):
                spin.blockSignals(True)
            try:
                ef = cfg.ga_config.elite_fraction
                ecf = getattr(cfg.ga_config, "elite_clone_fraction", 0)
                offspring_pct = (1.0 - ef) * 100
                self._spin_elite.setValue(round(offspring_pct, 1))
                if offspring_pct > 0:
                    self._spin_clone.setValue(round(offspring_pct * ecf, 1))
                    self._spin_mut_prob.setValue(round(offspring_pct * (1 - ecf), 1))
                else:
                    self._spin_clone.setValue(0)
                    self._spin_mut_prob.setValue(0)
            finally:
                for spin in (self._spin_elite, self._spin_clone, self._spin_mut_prob):
                    spin.blockSignals(False)
            self._spin_trait_prob.setValue(cfg.ga_config.mutation_prob * 100)
            self._spin_mut_std.setValue(cfg.ga_config.mutation_std)
            self._on_repro_params_changed()

    def _update_project_label(self) -> None:
        p = self._state.project_path
        if p:
            self._label_project.setText(Path(p).name)
            self._label_project.setStyleSheet(
                "font-size: 13px; color: #888; padding: 8px 0 8px 0;"
            )
        else:
            self._label_project.setText("No Project Loaded")
            self._label_project.setStyleSheet(
                "font-size: 16px; font-weight: bold; color: #aaa; padding: 24px 0 16px 0;"
            )
        self._update_content_visibility()

    def _update_content_visibility(self) -> None:
        """Show/hide league content, File vs New/Open buttons, and enable/disable menu actions."""
        has_project = self._state.project_path is not None
        self._content_container.setVisible(has_project)
        self._btns_no_project.setVisible(not has_project)
        self._btn_file.setVisible(has_project)
        self._act_save.setEnabled(has_project)
        self._act_save_as.setEnabled(has_project)
        self._act_export.setEnabled(has_project)

    def _groups_tuples(self) -> List[tuple]:
        return [
            (g.id, g.name, g.agents, g.source_group_id, g.source_group_name, g.color)
            for g in self._state.groups
        ]

    def _on_new_project(self) -> None:
        base = get_projects_folder()
        if not base or not Path(base).exists():
            Path(base).mkdir(parents=True, exist_ok=True)
        dlg = NewProjectDialog(base, self)
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
        try:
            project_save(
                path,
                groups=[],
                league_config=self.get_league_config(),
                generation_index=0,
                last_summary=None,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Create failed", str(e))
            return
        self._refresh_table()
        self._update_project_label()
        self._update_content_visibility()
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
        if show_message:
            QtWidgets.QMessageBox.information(self, "Open", f"Loaded project from {path}")
        return True

    def _on_open_project(self) -> None:
        base = get_projects_folder()
        dlg = NewProjectDialog(base, self)
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
        self._state.generation_index = data["generation_index"]
        self._state.last_summary = data.get("last_summary")

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
            )
            self._update_project_label()
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
