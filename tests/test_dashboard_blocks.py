"""Tests for Dashboard blocks: Compute and ELO (requires PySide6)."""

import sys

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets

from tarot_gui.dashboard_blocks import (
    ComputeBlockWidget,
    ELOBlockWidget,
    ExportBlockWidget,
    GameMetricsBlockWidget,
    RLPerformanceBlockWidget,
    _delta_elo_from_entries,
    _elo_stats_from_entries,
    _format_duration,
)


def test_format_duration():
    assert _format_duration(0) == "0:00"
    assert _format_duration(45) == "0:45"
    assert _format_duration(90) == "1:30"
    assert _format_duration(3661) == "1:01:01"
    assert _format_duration(-1) == "0:00"


def test_compute_block_update_metrics():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = ComputeBlockWidget()
    w.update_metrics(121.0, 60.0, 30.25)
    assert "2:01" in w._label_time_used.text()
    assert "1:00" in w._label_eta.text()
    assert "0:30" in w._label_avg.text()
    w.update_metrics(0, None, 10.0)
    assert "0:00" in w._label_time_used.text()
    assert "—" in w._label_eta.text()
    assert "0:10" in w._label_avg.text()


def test_compute_block_clear_metrics():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = ComputeBlockWidget()
    w.update_metrics(100.0, 50.0, 25.0)
    w.clear_metrics()
    assert "—" in w._label_time_used.text()
    assert "—" in w._label_eta.text()
    assert "—" in w._label_avg.text()


def test_elo_stats_from_entries_empty():
    assert _elo_stats_from_entries([]) is None


def test_elo_stats_from_entries_no_agents_uses_elo_mean():
    entries = [{"generation_index": 0, "elo_mean": 1520.5, "agents": []}]
    stats = _elo_stats_from_entries(entries)
    assert stats is not None
    assert stats["min"] == 1520.5
    assert stats["mean"] == 1520.5
    assert stats["max"] == 1520.5
    assert stats["std"] == 0.0


def test_elo_stats_from_entries_with_agents():
    entries = [
        {
            "generation_index": 0,
            "elo_mean": 1500.0,
            "agents": [
                {"elo_global": 1400.0},
                {"elo_global": 1500.0},
                {"elo_global": 1600.0},
            ],
        }
    ]
    stats = _elo_stats_from_entries(entries)
    assert stats is not None
    assert stats["min"] == 1400.0
    assert stats["mean"] == 1500.0
    assert stats["max"] == 1600.0
    assert abs(stats["std"] - 81.65) < 1.0


def test_elo_block_empty_shows_dash():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = ELOBlockWidget()
    w.set_entries([])
    assert "—" in w._summary_label.text()


def test_elo_block_summary_from_entries():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = ELOBlockWidget()
    entries = [
        {
            "generation_index": 0,
            "elo_min": 1000.0,
            "elo_mean": 1200.0,
            "elo_max": 1400.0,
            "agents": [
                {"elo_global": 1000.0},
                {"elo_global": 1200.0},
                {"elo_global": 1400.0},
            ],
        }
    ]
    w.set_entries(entries)
    assert "1000" in w._summary_label.text()
    assert "1200" in w._summary_label.text()
    assert "1400" in w._summary_label.text()


def test_elo_block_chart_receives_entries():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = ELOBlockWidget()
    entries = [
        {"generation_index": i, "elo_min": 1000 + i * 10, "elo_mean": 1200 + i * 10, "elo_max": 1400 + i * 10}
        for i in range(5)
    ]
    w.set_entries(entries)
    w._chart.set_entries(entries)
    w._chart.update()
    assert len(w._chart._entries) == 5


def test_delta_elo_from_entries():
    assert _delta_elo_from_entries([]) == {}
    entries_one = [{"generation_index": 0, "agents": [{"id": "a", "elo_global": 1500}]}]
    assert _delta_elo_from_entries(entries_one) == {}
    entries_two = entries_one + [
        {"generation_index": 1, "agents": [{"id": "a", "elo_global": 1520}, {"id": "b", "elo_global": 1480}]}
    ]
    d = _delta_elo_from_entries(entries_two)
    assert d["a"] == 20.0
    assert d["b"] == 1480.0


def test_rl_block_set_entries():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = RLPerformanceBlockWidget()
    entries = [
        {
            "generation_index": 0,
            "agents": [
                {"id": "x", "name": "X", "elo_global": 1600, "deals_played": 10, "deals_won": 6, "matches_played": 2, "matches_won": 1, "total_match_score": 50},
                {"id": "y", "name": "Y", "elo_global": 1400, "deals_played": 10, "deals_won": 4, "matches_played": 2, "matches_won": 0, "total_match_score": -50},
            ],
        }
    ]
    w.set_entries(entries)
    assert w._table.rowCount() == 2
    assert "X" in w._table.item(0, 0).text()
    assert "1600" in w._table.item(0, 1).text()
    assert "6/10" in w._table.item(0, 3).text()


def test_game_metrics_block_scope():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = GameMetricsBlockWidget()
    w.set_entries([])
    assert "—" in w._label_metrics.text()
    entries = [
        {"generation_index": i, "game_metrics": {"deals": 10 + i, "petit_au_bout": i, "grand_slem": 0}}
        for i in range(3)
    ]
    w.set_entries(entries)
    w._scope_combo.setCurrentIndex(0)
    w._refresh_metrics()
    assert "33" in w._label_metrics.text()
    w._scope_combo.setCurrentIndex(1)
    w._refresh_metrics()
    assert "12" in w._label_metrics.text()


def test_export_block_placeholder():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = ExportBlockWidget()
    child = w.findChild(QtWidgets.QLabel)
    assert child is not None
    assert "League Parameters" in child.toolTip()
