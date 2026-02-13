"""Tests for League tab GUI (requires PySide6 and QApplication)."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Skip if PySide6 not available
pyside6 = pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets

from tarot.persistence import population_to_dict
from tarot.tournament import Agent, Population
from tarot_gui.league_tab import (
    Group,
    GRP_COL_AGENTS,
    GRP_COL_CLONE_ONLY,
    GRP_COL_FIXED_ELO,
    GRP_COL_GA_PARENT,
    GRP_COL_NAME,
    GRP_COL_PLAY_IN_LEAGUE,
    LeagueTabState,
    LeagueTabWidget,
    make_league_tab,
)


def test_league_tab_state():
    g = Group(id="grp_test_0", name="Test", agents=[Agent(id="a", name="A", player_counts=[4])])
    state = LeagueTabState(groups=[g])
    assert len(state.groups) == 1
    assert state.total_agents() == 1
    pop = state.build_population()
    assert len(pop.agents) == 1
    assert "a" in pop.agents


def test_league_tab_widget_creates():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    tab = make_league_tab()
    assert isinstance(tab, LeagueTabWidget)
    assert len(tab.state().groups) == 0
    assert tab.state().total_agents() == 0


def test_league_tab_table_sync():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    g = Group(id="grp_x", name="Test group", agents=[Agent(id="x", name="X", player_counts=[4])])
    state = LeagueTabState(groups=[g])
    tab = LeagueTabWidget()
    tab._state = state
    tab._refresh_table()
    assert tab._table.rowCount() == 1
    assert tab._table.item(0, GRP_COL_NAME).text() == "Test group"
    assert tab._table.item(0, GRP_COL_AGENTS).text() == "1"  # # agents


def test_add_random():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    tab = make_league_tab()
    assert tab.state().total_agents() == 0
    tab._spin_add_random.setValue(3)
    tab._on_add_random()
    assert len(tab.state().groups) == 1
    assert tab.state().total_agents() == 3
    g = tab.state().groups[0]
    assert g.name == "Random 3"
    assert g.id.startswith("grp_rand_")


def test_import_replace():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    pop = Population()
    pop.add(Agent(id="imp1", name="Imported", player_counts=[4]))
    d = population_to_dict(pop)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(d, f)
        path = f.name
    try:
        tab = LeagueTabWidget()
        tab._on_add_random()  # Add one group
        assert tab.state().total_agents() >= 1
        from tarot.persistence import population_from_dict
        imported = population_from_dict(d)
        gid = "grp_imp_0"
        from tarot_gui.league_tab import _assign_group_agent_ids
        agents = _assign_group_agent_ids(list(imported.agents.values()), gid)
        new_group = Group(id=gid, name="Imported (1)", agents=agents)
        tab._state.groups = [new_group]
        tab._refresh_table()
        assert tab.state().total_agents() == 1
        assert tab.state().groups[0].agents[0].id == "grp_imp_0_0"
    finally:
        Path(path).unlink(missing_ok=True)


def _get_checkbox(tab: LeagueTabWidget, row: int, col: int) -> QtWidgets.QCheckBox:
    """Get the checkbox in a table cell."""
    cell = tab._table.cellWidget(row, col)
    assert cell is not None
    cb = cell.findChild(QtWidgets.QCheckBox)
    assert cb is not None
    return cb


def test_checkbox_ga_parent():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    agents = [Agent(id="a", name="A", player_counts=[4], can_use_as_ga_parent=True)]
    g = Group(id="grp_x", name="Test", agents=agents)
    tab = LeagueTabWidget()
    tab._state = LeagueTabState(groups=[g])
    tab._refresh_table()
    cb = _get_checkbox(tab, 0, GRP_COL_GA_PARENT)
    assert cb.isChecked()
    cb.setChecked(False)
    assert not g.all_can_use_as_ga_parent()
    assert not agents[0].can_use_as_ga_parent
    cb.setChecked(True)
    assert g.all_can_use_as_ga_parent()
    assert agents[0].can_use_as_ga_parent


def test_checkbox_fixed_elo():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    agents = [Agent(id="a", name="A", player_counts=[4], fixed_elo=False)]
    g = Group(id="grp_x", name="Test", agents=agents)
    tab = LeagueTabWidget()
    tab._state = LeagueTabState(groups=[g])
    tab._refresh_table()
    cb = _get_checkbox(tab, 0, GRP_COL_FIXED_ELO)
    assert not cb.isChecked()
    cb.setChecked(True)
    assert g.all_fixed_elo()
    assert agents[0].fixed_elo
    cb.setChecked(False)
    assert not g.all_fixed_elo()
    assert not agents[0].fixed_elo


def test_checkbox_clone_only():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    agents = [Agent(id="a", name="A", player_counts=[4], clone_only=False)]
    g = Group(id="grp_x", name="Test", agents=agents)
    tab = LeagueTabWidget()
    tab._state = LeagueTabState(groups=[g])
    tab._refresh_table()
    cb = _get_checkbox(tab, 0, GRP_COL_CLONE_ONLY)
    assert not cb.isChecked()
    cb.setChecked(True)
    assert g.all_clone_only()
    assert agents[0].clone_only
    cb.setChecked(False)
    assert not g.all_clone_only()
    assert not agents[0].clone_only


def test_checkbox_play_in_league():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    agents = [Agent(id="a", name="A", player_counts=[4], play_in_league=True)]
    g = Group(id="grp_x", name="Test", agents=agents)
    tab = LeagueTabWidget()
    tab._state = LeagueTabState(groups=[g])
    tab._refresh_table()
    cb = _get_checkbox(tab, 0, GRP_COL_PLAY_IN_LEAGUE)
    assert cb.isChecked()
    cb.setChecked(False)
    assert not g.all_play_in_league()
    assert not agents[0].play_in_league
    cb.setChecked(True)
    assert g.all_play_in_league()
    assert agents[0].play_in_league


def test_checkbox_persists_after_refresh():
    """Checkbox state persists when table is refreshed."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    agents = [
        Agent(id="a", name="A", player_counts=[4], fixed_elo=True, clone_only=True),
    ]
    g = Group(id="grp_x", name="Test", agents=agents)
    tab = LeagueTabWidget()
    tab._state = LeagueTabState(groups=[g])
    tab._refresh_table()
    cb_fixed = _get_checkbox(tab, 0, GRP_COL_FIXED_ELO)
    cb_clone = _get_checkbox(tab, 0, GRP_COL_CLONE_ONLY)
    assert cb_fixed.isChecked()
    assert cb_clone.isChecked()
    tab._refresh_table()  # Rebuild table
    cb_fixed2 = _get_checkbox(tab, 0, GRP_COL_FIXED_ELO)
    cb_clone2 = _get_checkbox(tab, 0, GRP_COL_CLONE_ONLY)
    assert cb_fixed2.isChecked()
    assert cb_clone2.isChecked()
