"""
League tab: population management (groups of agents), league structure, GA parameters, run controls.

Groups contain agents; the main table shows one row per group. Expand opens a detail dialog.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from tarot.ga import GAConfig
from tarot_gui.charts import GAVisualWidget, GenerationFlowWidget, PIE_COLORS, PopulationPieWidget
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
GRP_COL_EXPAND = 0
GRP_COL_COLOR = 1
GRP_COL_GA_PARENT = 2
GRP_COL_FIXED_ELO = 3
GRP_COL_CLONE_ONLY = 4
GRP_COL_PLAY_IN_LEAGUE = 5
GRP_COL_NAME = 6
GRP_COL_AGENTS = 7
GRP_COL_SOURCE = 8
GRP_COL_ELO = 9
GRP_COL_ACTIONS = 10
GRP_NUM_COLUMNS = 11

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
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)

        pop_group = QtWidgets.QGroupBox("Population")
        pop_layout = QtWidgets.QHBoxLayout(pop_group)
        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(GRP_NUM_COLUMNS)
        self._table.setHorizontalHeaderLabels(
            [
                "Expand",
                "",
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
        # Set tooltips on column headers for the flag checkboxes
        from PySide6.QtCore import Qt
        model = self._table.model()
        model.setHeaderData(GRP_COL_COLOR, Qt.Orientation.Horizontal, "Color used in pie chart", Qt.ItemDataRole.ToolTipRole)
        model.setHeaderData(GRP_COL_GA_PARENT, Qt.Orientation.Horizontal, GRP_TOOLTIP_GA_PARENT, Qt.ItemDataRole.ToolTipRole)
        model.setHeaderData(GRP_COL_FIXED_ELO, Qt.Orientation.Horizontal, GRP_TOOLTIP_FIXED_ELO, Qt.ItemDataRole.ToolTipRole)
        model.setHeaderData(GRP_COL_CLONE_ONLY, Qt.Orientation.Horizontal, GRP_TOOLTIP_CLONE_ONLY, Qt.ItemDataRole.ToolTipRole)
        model.setHeaderData(GRP_COL_PLAY_IN_LEAGUE, Qt.Orientation.Horizontal, GRP_TOOLTIP_PLAY_IN_LEAGUE, Qt.ItemDataRole.ToolTipRole)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._table.horizontalHeader().setMinimumSectionSize(80)
        self._table.verticalHeader().setDefaultSectionSize(38)
        self._table.horizontalHeader().setSectionResizeMode(
            GRP_COL_ELO, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        pop_layout.addWidget(self._table, stretch=1)

        self._pie_widget = PopulationPieWidget()
        pop_layout.addWidget(self._pie_widget)

        tools_widget = QtWidgets.QWidget()
        tools_layout = QtWidgets.QVBoxLayout(tools_widget)
        tools_layout.addWidget(QtWidgets.QLabel("Tools"))
        self._spin_add_random = QtWidgets.QSpinBox()
        self._spin_add_random.setRange(1, 999)
        self._spin_add_random.setValue(4)
        btn_add_random = QtWidgets.QPushButton("Add random")
        btn_add_random.clicked.connect(self._on_add_random)
        tools_layout.addWidget(QtWidgets.QLabel("Count:"))
        tools_layout.addWidget(self._spin_add_random)
        tools_layout.addWidget(btn_add_random)
        btn_import = QtWidgets.QPushButton("Import")
        btn_import.clicked.connect(self._on_import)
        tools_layout.addWidget(btn_import)
        btn_augment = QtWidgets.QPushButton("Augment from selection")
        btn_augment.clicked.connect(self._on_augment_from_selection)
        tools_layout.addWidget(btn_augment)
        btn_clear = QtWidgets.QPushButton("Clear")
        btn_clear.clicked.connect(self._on_clear)
        tools_layout.addWidget(btn_clear)
        tools_layout.addStretch(1)
        pop_layout.addWidget(tools_widget)
        pop_group.setMinimumHeight(300)
        layout.addWidget(pop_group)

        # Bottom row: config column (left, narrow) | charts (large, stretches to bottom)
        bottom_row = QtWidgets.QHBoxLayout()

        config_column = QtWidgets.QWidget()
        config_column.setMaximumWidth(220)
        config_layout = QtWidgets.QVBoxLayout(config_column)
        config_layout.setContentsMargins(0, 0, 8, 0)

        league_group = QtWidgets.QGroupBox("League structure")
        league_layout = QtWidgets.QFormLayout(league_group)
        self._combo_player_count = QtWidgets.QComboBox()
        player_model = QtGui.QStandardItemModel()
        for n in [3, 4, 5]:
            item = QtGui.QStandardItem(str(n))
            item.setData(n, QtCore.Qt.ItemDataRole.UserRole)
            player_model.appendRow(item)
        self._combo_player_count.setModel(player_model)
        self._combo_player_count.setCurrentIndex(1)  # default 4
        league_layout.addRow("Players:", self._combo_player_count)
        self._spin_deals = QtWidgets.QSpinBox()
        self._spin_deals.setRange(1, 99)
        self._spin_deals.setValue(5)
        league_layout.addRow("Deals/match:", self._spin_deals)
        self._spin_matches = QtWidgets.QSpinBox()
        self._spin_matches.setRange(1, 999)
        self._spin_matches.setValue(3)
        league_layout.addRow("Matches/gen:", self._spin_matches)
        self._combo_league_style = QtWidgets.QComboBox()
        self._combo_league_style.addItems(["ELO-based", "Random"])
        from PySide6.QtCore import Qt
        self._combo_league_style.setItemData(0, "Match agents of similar ELO strength together.", Qt.ItemDataRole.ToolTipRole)
        self._combo_league_style.setItemData(1, "Shuffle agents randomly into tables.", Qt.ItemDataRole.ToolTipRole)
        league_layout.addRow("Style:", self._combo_league_style)
        config_layout.addWidget(league_group)

        ga_group = QtWidgets.QGroupBox("GA parameters")
        ga_layout = QtWidgets.QFormLayout(ga_group)
        self._spin_elite = QtWidgets.QDoubleSpinBox()
        self._spin_elite.setRange(0, 100)
        self._spin_elite.setValue(10)
        self._spin_elite.setSuffix(" %")
        self._spin_elite.setDecimals(1)
        ga_layout.addRow("Elite %:", self._spin_elite)
        self._spin_mut_prob = QtWidgets.QDoubleSpinBox()
        self._spin_mut_prob.setRange(0, 100)
        self._spin_mut_prob.setValue(50)
        self._spin_mut_prob.setSuffix(" %")
        self._spin_mut_prob.setDecimals(1)
        ga_layout.addRow("Mutation %:", self._spin_mut_prob)
        self._spin_mut_std = QtWidgets.QDoubleSpinBox()
        self._spin_mut_std.setRange(0.01, 1.0)
        self._spin_mut_std.setValue(0.1)
        self._spin_mut_std.setSingleStep(0.05)
        self._spin_mut_std.setToolTip("Std deviation for trait perturbation when mutating. E.g. 0.1 = small random changes.")
        ga_layout.addRow("Mut. std:", self._spin_mut_std)
        self._spin_generations = QtWidgets.QSpinBox()
        self._spin_generations.setRange(1, 9999)
        self._spin_generations.setValue(10)
        ga_layout.addRow("Generations:", self._spin_generations)
        config_layout.addWidget(ga_group)
        # Connect GA params to GA visual
        for spin in (self._spin_elite, self._spin_mut_prob, self._spin_mut_std):
            spin.valueChanged.connect(self._update_ga_visual)

        export_group = QtWidgets.QGroupBox("Export")
        export_layout = QtWidgets.QVBoxLayout(export_group)
        export_layout.addWidget(QtWidgets.QLabel("Export options (Phase 5)"))
        config_layout.addWidget(export_group)

        config_layout.addStretch(1)
        bottom_row.addWidget(config_column)

        charts_container = QtWidgets.QWidget()
        charts_container.setObjectName("chartsPlaceholder")
        charts_container.setStyleSheet("QWidget#chartsPlaceholder { border: 1px solid #606060; }")
        charts_layout = QtWidgets.QVBoxLayout(charts_container)
        flow_label = QtWidgets.QLabel("Generation flow")
        flow_label.setStyleSheet("font-weight: bold;")
        charts_layout.addWidget(flow_label)
        self._flow_widget = GenerationFlowWidget()
        charts_layout.addWidget(self._flow_widget)
        ga_label = QtWidgets.QLabel("GA visual")
        ga_label.setStyleSheet("font-weight: bold; margin-top: 8px;")
        charts_layout.addWidget(ga_label)
        self._ga_visual_widget = GAVisualWidget()
        charts_layout.addWidget(self._ga_visual_widget)
        charts_layout.addStretch(1)
        bottom_row.addWidget(charts_container, stretch=1)

        bottom_container = QtWidgets.QWidget()
        bottom_container.setLayout(bottom_row)
        layout.addWidget(bottom_container, stretch=1)
        scroll.setWidget(content)

        self._update_ga_visual()

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _refresh_table(self) -> None:
        self._table.setRowCount(0)
        for row, group in enumerate(self._state.groups):
            self._table.insertRow(row)
            self._fill_group_row(row, group)
        # Ensure columns are wide enough for content
        self._table.setColumnWidth(GRP_COL_EXPAND, max(72, self._table.columnWidth(GRP_COL_EXPAND)))
        self._table.setColumnWidth(GRP_COL_ACTIONS, max(75, self._table.columnWidth(GRP_COL_ACTIONS)))
        self._table.setColumnWidth(GRP_COL_ELO, max(160, self._table.columnWidth(GRP_COL_ELO)))
        self._update_player_count_options()
        self._update_pie_chart()

    def _update_pie_chart(self) -> None:
        slices = [(g.name, len(g.agents)) for g in self._state.groups]
        total = self._state.total_agents()
        colors = [g.color for g in self._state.groups]
        self._pie_widget.set_data(slices, total, colors)

    def _update_ga_visual(self) -> None:
        self._ga_visual_widget.set_params(
            self._spin_elite.value(),
            self._spin_mut_prob.value(),
            self._spin_mut_std.value(),
        )

    def _update_player_count_options(self) -> None:
        """Gray out player count options that are not possible given total agents."""
        total = self._state.total_agents()
        model = self._combo_player_count.model()
        current_idx = self._combo_player_count.currentIndex()
        first_enabled = -1
        for row in range(model.rowCount()):
            item = model.item(row)
            n = int(item.text())
            enabled = total >= n
            item.setEnabled(enabled)
            if enabled and first_enabled < 0:
                first_enabled = row
        if 0 <= current_idx < model.rowCount() and not model.item(current_idx).isEnabled() and first_enabled >= 0:
            self._combo_player_count.setCurrentIndex(first_enabled)

    def _fill_group_row(self, row: int, group: Group) -> None:
        btn_expand = QtWidgets.QPushButton("Expand")
        btn_expand.setMinimumWidth(60)
        btn_expand.setMinimumHeight(22)
        btn_expand.clicked.connect(lambda checked=False, g=group: self._on_expand_group(g))
        cell = QtWidgets.QWidget()
        ll = QtWidgets.QHBoxLayout(cell)
        ll.setContentsMargins(4, 2, 4, 2)
        ll.addWidget(btn_expand)
        self._table.setCellWidget(row, GRP_COL_EXPAND, cell)

        # Color swatch
        color_cell = QtWidgets.QFrame()
        color_cell.setFixedSize(22, 22)
        color_cell.setFrameStyle(QtWidgets.QFrame.Shape.Box | QtWidgets.QFrame.Shadow.Plain)
        color_cell.setLineWidth(1)
        red = (group.color >> 16) & 0xFF
        grn = (group.color >> 8) & 0xFF
        blu = group.color & 0xFF
        color_cell.setStyleSheet(f"background-color: rgb({red},{grn},{blu});")
        color_cell.setToolTip(group.name)
        self._table.setCellWidget(row, GRP_COL_COLOR, color_cell)

        def make_flag_cell(checked: bool, setter) -> QtWidgets.QWidget:
            cb = QtWidgets.QCheckBox()
            cb.setChecked(checked)
            cb.toggled.connect(lambda checked, setter=setter: setter(checked))
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
        btn_del.setMinimumWidth(60)
        btn_del.setMinimumHeight(22)
        btn_del.clicked.connect(lambda checked=False, g=group: self._on_delete_group(g))
        cell_actions = QtWidgets.QWidget()
        al = QtWidgets.QHBoxLayout(cell_actions)
        al.setContentsMargins(4, 2, 4, 2)
        al.addWidget(btn_del)
        self._table.setCellWidget(row, GRP_COL_ACTIONS, cell_actions)

    def _on_expand_group(self, group: Group) -> None:
        dlg = GroupDetailDialog(group, self)
        dlg.exec()

    def _on_delete_group(self, group: Group) -> None:
        if group in self._state.groups:
            self._state.groups.remove(group)
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

    def _on_augment_from_selection(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            QtWidgets.QMessageBox.information(self, "Augment", "Select one or more group rows to use as base.")
            return
        base_agents: List[Agent] = []
        source_names: List[str] = []
        for idx in rows:
            row = idx.row()
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
            src_id = self._state.groups[rows[0].row()].id if rows else None
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
        player_count = int(self._combo_player_count.currentText())
        elite_frac = self._spin_elite.value() / 100.0
        mut_prob = self._spin_mut_prob.value() / 100.0
        return LeagueConfig(
            player_count=player_count,
            deals_per_match=self._spin_deals.value(),
            rounds_per_generation=self._spin_matches.value(),
            matchmaking_style=style,
            ga_config=GAConfig(
                population_size=max(1, self._state.total_agents()),
                elite_fraction=elite_frac,
                mutation_prob=mut_prob,
                mutation_std=self._spin_mut_std.value(),
            ),
        )


def make_league_tab() -> QtWidgets.QWidget:
    """Create the League tab widget."""
    return LeagueTabWidget()
