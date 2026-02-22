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
    GROUP_NAMES,
    Group,
    GRP_COL_AGENTS,
    GRP_COL_CLONE_ONLY,
    GRP_COL_FIXED_ELO,
    GRP_COL_GA_PARENT,
    GRP_COL_NAME,
    GRP_COL_PLAY_IN_LEAGUE,
    LeagueTabState,
    LeagueTabWidget,
    RunSectionWidget,
    make_league_tab,
    _format_duration,
)
from tarot_gui.run_log import RunLogManager


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
    # Group name is picked from GROUP_NAMES, with a fallback "Cohort <digits>" when exhausted.
    in_group_names = g.name in GROUP_NAMES
    is_cohort = g.name.startswith("Cohort ") and g.name[7:].isdigit()
    assert in_group_names or is_cohort
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


def test_add_to_hof_from_population():
    """HOF when=Every generation, what=Best agent only: one agent added with unique id."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    tab = make_league_tab()
    tab._combo_hof_when.setCurrentIndex(1)  # Every generation
    tab._combo_hof_what.setCurrentIndex(2)   # Best agent only
    pop = Population()
    pop.add(Agent(id="a1", name="A1", player_counts=[4], elo_global=1400.0))
    pop.add(Agent(id="a2", name="A2", player_counts=[4], elo_global=1600.0))
    pop.add(Agent(id="a3", name="A3", player_counts=[4], elo_global=1500.0))
    assert len(tab.state().hof_agents) == 0
    tab.add_to_hof_from_population(pop, 0)
    assert len(tab.state().hof_agents) == 1
    hof_agent = tab.state().hof_agents[0]
    assert hof_agent.elo_global == 1600.0
    assert hof_agent.id == "a2_gen0_0"


def test_add_to_hof_on_demand_only():
    """HOF when=On demand only: no agents added during add_to_hof_from_population."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    tab = make_league_tab()
    tab._combo_hof_when.setCurrentIndex(0)  # On demand only
    pop = Population()
    pop.add(Agent(id="a1", name="A1", player_counts=[4], elo_global=1600.0))
    tab.add_to_hof_from_population(pop, 0)
    assert len(tab.state().hof_agents) == 0


def test_add_to_hof_top_n_by_elo():
    """HOF what=Top N by ELO: N agents added with unique ids."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    tab = make_league_tab()
    tab._combo_hof_when.setCurrentIndex(1)
    tab._combo_hof_what.setCurrentIndex(0)  # Top N by ELO
    tab._spin_hof_top_n.setValue(2)
    pop = Population()
    pop.add(Agent(id="x", name="X", player_counts=[4], elo_global=1000.0))
    pop.add(Agent(id="y", name="Y", player_counts=[4], elo_global=1700.0))
    pop.add(Agent(id="z", name="Z", player_counts=[4], elo_global=1500.0))
    tab.add_to_hof_from_population(pop, 1)
    assert len(tab.state().hof_agents) == 2
    elos = [a.elo_global for a in tab.state().hof_agents]
    assert elos == [1700.0, 1500.0]
    assert tab.state().hof_agents[0].id == "y_gen1_0"
    assert tab.state().hof_agents[1].id == "z_gen1_1"


def test_run_section_widget_without_run_log_manager():
    """RunSectionWidget works without run_log_manager; Save is disabled."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    state = LeagueTabState()
    widget = RunSectionWidget(state, run_log_manager=None)
    assert not widget._btn_save_run_log.isEnabled()
    assert widget._run_log_manager is None


def test_run_section_widget_with_run_log_manager_save_disabled_when_no_data():
    """RunSectionWidget with RunLogManager and no data: Save disabled."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    state = LeagueTabState()
    mgr = RunLogManager()
    widget = RunSectionWidget(state, run_log_manager=mgr)
    widget.update_run_log_buttons()
    assert not widget._btn_save_run_log.isEnabled()


def test_run_section_widget_with_run_log_manager_save_enabled_after_append():
    """RunSectionWidget with RunLogManager: Save enabled after append_generation."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    state = LeagueTabState()
    mgr = RunLogManager()
    pop = Population()
    pop.add(Agent(id="a", name="A", player_counts=[4], elo_global=1500.0))
    summary = {"elo_min": 1500.0, "elo_mean": 1500.0, "elo_max": 1500.0, "num_agents": 1.0}
    mgr.append_generation(0, pop, summary)
    widget = RunSectionWidget(state, run_log_manager=mgr)
    widget.update_run_log_buttons()
    assert widget._btn_save_run_log.isEnabled()


# --- Step 2: Run box status line ---


def test_format_duration():
    """Duration formatting for status line (M:SS and H:MM:SS)."""
    assert _format_duration(0) == "0:00"
    assert _format_duration(45) == "0:45"
    assert _format_duration(90) == "1:30"
    assert _format_duration(3661) == "1:01:01"
    assert _format_duration(-1) == "0:00"


def test_run_section_status_idle():
    """Status line shows 'Status: —' when not running (gen_index < 0)."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    state = LeagueTabState()
    widget = RunSectionWidget(state, run_log_manager=None)
    widget.update_run_status(-1, 0, 0.0, None)
    assert widget._label_status.text() == "Status: —"


def test_run_section_status_first_gen():
    """Status line shows Generation 1 of Y, Elapsed, ETA 'calculating…' for first gen."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    state = LeagueTabState()
    widget = RunSectionWidget(state, run_log_manager=None)
    widget.update_run_status(0, 5, 10.5, None)
    text = widget._label_status.text()
    assert "Generation 1 of 5" in text
    assert "Elapsed:" in text
    assert "0:10" in text or "0:11" in text
    assert "ETA:" in text
    assert "calculating" in text


def test_run_section_status_second_gen_with_eta():
    """Status line shows real ETA when gen_index >= 1."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    state = LeagueTabState()
    widget = RunSectionWidget(state, run_log_manager=None)
    widget.update_run_status(1, 5, 60.0, 90.0)
    text = widget._label_status.text()
    assert "Generation 2 of 5" in text
    assert "Elapsed:" in text
    assert "1:00" in text
    assert "ETA:" in text
    assert "1:30" in text
    assert "calculating" not in text
