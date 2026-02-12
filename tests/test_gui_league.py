"""Tests for League tab GUI (requires PySide6 and QApplication)."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Skip if PySide6 not available
pyside6 = pytest.importorskip("PySide6")

from PySide6 import QtWidgets

from tarot.persistence import population_to_dict
from tarot.tournament import Agent, Population
from tarot_gui.league_tab import Group, GRP_COL_NAME, LeagueTabState, LeagueTabWidget, make_league_tab


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
    assert tab._table.item(0, 3).text() == "1"  # # agents


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
