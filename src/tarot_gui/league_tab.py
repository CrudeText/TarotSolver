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

from PySide6 import QtCore, QtWidgets

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

    def all_can_use_as_ga_parent(self) -> bool:
        return all(a.can_use_as_ga_parent for a in self.agents)

    def set_all_can_use_as_ga_parent(self, value: bool) -> None:
        for a in self.agents:
            a.can_use_as_ga_parent = value

    def elo_min(self) -> float:
        return min(a.elo_global for a in self.agents) if self.agents else 0.0

    def elo_mean(self) -> float:
        if not self.agents:
            return 0.0
        return sum(a.elo_global for a in self.agents) / len(self.agents)

    def elo_max(self) -> float:
        return max(a.elo_global for a in self.agents) if self.agents else 0.0


# Group ID counters
_group_counters: Dict[str, int] = {"rand": 0, "mut": 0, "imp": 0}


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
GRP_COL_GA_PARENT = 1
GRP_COL_NAME = 2
GRP_COL_AGENTS = 3
GRP_COL_SOURCE = 4
GRP_COL_ELO = 5
GRP_COL_ACTIONS = 6
GRP_NUM_COLUMNS = 7


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
        btn.setMinimumHeight(28)
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


class LeagueTabWidget(QtWidgets.QWidget):
    """League tab: groups table, config sections, run controls."""

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
            ["Expand", "GA parent", "Group name", "# agents", "Source", "ELO (min/mean/max)", "Actions"]
        )
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self._table.horizontalHeader().setMinimumSectionSize(80)
        self._table.horizontalHeader().setSectionResizeMode(
            GRP_COL_ELO, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        pop_layout.addWidget(self._table, stretch=1)

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
        layout.addWidget(pop_group)

        config_row = QtWidgets.QHBoxLayout()
        league_group = QtWidgets.QGroupBox("League structure")
        league_layout = QtWidgets.QFormLayout(league_group)
        league_layout.addRow("Player count:", QtWidgets.QComboBox())
        league_layout.addRow("Deals / match:", QtWidgets.QSpinBox())
        league_layout.addRow("Matches / generation:", QtWidgets.QSpinBox())
        league_layout.addRow("League style:", QtWidgets.QComboBox())
        config_row.addWidget(league_group)

        ga_group = QtWidgets.QGroupBox("GA parameters")
        ga_layout = QtWidgets.QFormLayout(ga_group)
        ga_layout.addRow("Elite fraction:", QtWidgets.QDoubleSpinBox())
        ga_layout.addRow("Mutation prob:", QtWidgets.QDoubleSpinBox())
        ga_layout.addRow("Mutation std:", QtWidgets.QDoubleSpinBox())
        ga_layout.addRow("Generations:", QtWidgets.QSpinBox())
        config_row.addWidget(ga_group)
        layout.addLayout(config_row)

        run_export_row = QtWidgets.QHBoxLayout()
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
        run_export_row.addWidget(run_group)

        export_group = QtWidgets.QGroupBox("Export")
        export_layout = QtWidgets.QVBoxLayout(export_group)
        export_layout.addWidget(QtWidgets.QLabel("Export options (Phase 5)"))
        run_export_row.addWidget(export_group)
        layout.addLayout(run_export_row)

        graphs_placeholder = QtWidgets.QLabel("Charts area (Phase 4)")
        graphs_placeholder.setMinimumHeight(200)
        graphs_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        graphs_placeholder.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        layout.addWidget(graphs_placeholder, stretch=1)
        scroll.setWidget(content)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll)

    def _refresh_table(self) -> None:
        self._table.setRowCount(0)
        for row, group in enumerate(self._state.groups):
            self._table.insertRow(row)
            self._fill_group_row(row, group)
        # Ensure ELO column is wide enough for "1500 / 1500 / 1500" and header "ELO (min/mean/max)"
        self._table.setColumnWidth(GRP_COL_ELO, max(160, self._table.columnWidth(GRP_COL_ELO)))
        self._update_metrics_label()

    def _fill_group_row(self, row: int, group: Group) -> None:
        btn_expand = QtWidgets.QPushButton("Expand")
        btn_expand.clicked.connect(lambda checked=False, g=group: self._on_expand_group(g))
        cell = QtWidgets.QWidget()
        ll = QtWidgets.QHBoxLayout(cell)
        ll.setContentsMargins(4, 2, 4, 2)
        ll.addWidget(btn_expand)
        self._table.setCellWidget(row, GRP_COL_EXPAND, cell)

        cb = QtWidgets.QCheckBox()
        cb.setChecked(group.all_can_use_as_ga_parent())
        cb.stateChanged.connect(
            lambda s, g=group: g.set_all_can_use_as_ga_parent(s == QtCore.Qt.CheckState.Checked)
        )
        cell_cb = QtWidgets.QWidget()
        ll2 = QtWidgets.QHBoxLayout(cell_cb)
        ll2.setContentsMargins(4, 2, 4, 2)
        ll2.addWidget(cb)
        self._table.setCellWidget(row, GRP_COL_GA_PARENT, cell_cb)

        self._table.setItem(row, GRP_COL_NAME, QtWidgets.QTableWidgetItem(group.name))
        self._table.setItem(row, GRP_COL_AGENTS, QtWidgets.QTableWidgetItem(str(len(group.agents))))
        src = group.source_group_name or (group.source_group_id or "—")
        self._table.setItem(row, GRP_COL_SOURCE, QtWidgets.QTableWidgetItem(src))
        elo_str = f"{group.elo_min():.0f} / {group.elo_mean():.0f} / {group.elo_max():.0f}"
        self._table.setItem(row, GRP_COL_ELO, QtWidgets.QTableWidgetItem(elo_str))

        btn_del = QtWidgets.QPushButton("Delete")
        btn_del.setMinimumHeight(28)
        btn_del.clicked.connect(lambda checked=False, g=group: self._on_delete_group(g))
        cell_actions = QtWidgets.QWidget()
        al = QtWidgets.QHBoxLayout(cell_actions)
        al.addWidget(btn_del)
        self._table.setCellWidget(row, GRP_COL_ACTIONS, cell_actions)

    def _on_expand_group(self, group: Group) -> None:
        dlg = GroupDetailDialog(group, self)
        dlg.exec()

    def _on_delete_group(self, group: Group) -> None:
        if group in self._state.groups:
            self._state.groups.remove(group)
            self._refresh_table()

    def _update_metrics_label(self) -> None:
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
        group = Group(id=gid, name=f"Random {n}")
        group.agents = agents
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
        new_group = Group(id=gid, name=f"Imported ({len(renamed)})", agents=renamed)

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
            new_group = Group(
                id=gid,
                name=group_name,
                agents=new_agents,
                source_group_id=src_id,
                source_group_name=source_label,
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


def make_league_tab() -> QtWidgets.QWidget:
    """Create the League tab widget."""
    return LeagueTabWidget()
