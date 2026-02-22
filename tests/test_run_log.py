"""Tests for Dashboard run log (RunLogManager, JSONL schema)."""

import json
import tempfile
from pathlib import Path

import pytest

from tarot.tournament import Agent, Population
from tarot_gui.run_log import (
    RUN_LOG_SCHEMA_VERSION,
    LoadedRunLog,
    RunLogManager,
    build_run_log_entry,
    parse_run_log_entry,
)


def _make_population(n: int = 2) -> Population:
    pop = Population()
    for i in range(n):
        pop.add(
            Agent(
                id=f"agent_{i}",
                name=f"Agent {i}",
                player_counts=[4],
                elo_global=1500.0 + i * 10,
                generation=i,
                matches_played=5 + i,
                matches_won=i,
                total_match_score=100.0 + i * 20,
                deals_played=20,
                deals_won=10 + i,
            )
        )
    return pop


def test_build_run_log_entry():
    pop = _make_population(2)
    summary = {"elo_min": 1490.0, "elo_mean": 1510.0, "elo_max": 1530.0, "num_agents": 2.0}
    entry = build_run_log_entry(0, pop, summary)
    assert entry["schema_version"] == RUN_LOG_SCHEMA_VERSION
    assert entry["generation_index"] == 0
    assert "timestamp_utc" in entry
    assert entry["elo_min"] == 1490.0
    assert entry["elo_mean"] == 1510.0
    assert entry["elo_max"] == 1530.0
    assert entry["num_agents"] == 2
    assert len(entry["agents"]) == 2
    agent_ids = {a["id"] for a in entry["agents"]}
    assert agent_ids == {"agent_0", "agent_1"}
    for a in entry["agents"]:
        assert "elo_global" in a
        assert "generation" in a
        assert "matches_played" in a
        assert "total_match_score" in a


def test_build_run_log_entry_includes_game_metrics():
    pop = _make_population(1)
    summary = {
        "elo_min": 1500.0,
        "elo_mean": 1500.0,
        "elo_max": 1500.0,
        "num_agents": 1.0,
        "deals": 100,
        "petit_au_bout": 3,
        "grand_slem": 1,
    }
    entry = build_run_log_entry(0, pop, summary)
    assert "game_metrics" in entry
    assert entry["game_metrics"]["deals"] == 100
    assert entry["game_metrics"]["petit_au_bout"] == 3
    assert entry["game_metrics"]["grand_slem"] == 1


def test_parse_run_log_entry():
    pop = _make_population(1)
    summary = {"elo_min": 1500.0, "elo_mean": 1500.0, "elo_max": 1500.0, "num_agents": 1.0}
    entry = build_run_log_entry(0, pop, summary)
    line = json.dumps(entry, ensure_ascii=False)
    parsed = parse_run_log_entry(line)
    assert parsed["generation_index"] == entry["generation_index"]
    assert parsed["timestamp_utc"] == entry["timestamp_utc"]
    assert len(parsed["agents"]) == 1
    assert parsed["agents"][0]["id"] == "agent_0"


def test_run_log_manager_append_and_current_entries():
    mgr = RunLogManager()
    assert not mgr.has_current_data()
    assert mgr.get_current_entries() == []

    pop = _make_population(1)
    summary = {"elo_min": 1500.0, "elo_mean": 1500.0, "elo_max": 1500.0, "num_agents": 1.0}
    mgr.append_generation(0, pop, summary)
    assert mgr.has_current_data()
    entries = mgr.get_current_entries()
    assert len(entries) == 1
    assert entries[0]["generation_index"] == 0

    mgr.append_generation(1, pop, summary)
    entries = mgr.get_current_entries()
    assert len(entries) == 2
    assert entries[1]["generation_index"] == 1


def test_run_log_manager_clear_current():
    mgr = RunLogManager()
    pop = _make_population(1)
    summary = {"elo_min": 1500.0, "elo_mean": 1500.0, "elo_max": 1500.0, "num_agents": 1.0}
    mgr.append_generation(0, pop, summary)
    assert mgr.has_current_data()
    mgr.clear_current()
    assert not mgr.has_current_data()
    assert mgr.get_current_entries() == []


def test_run_log_manager_save_to_path():
    mgr = RunLogManager()
    pop = _make_population(2)
    summary = {"elo_min": 1490.0, "elo_mean": 1510.0, "elo_max": 1530.0, "num_agents": 2.0}
    mgr.append_generation(0, pop, summary)
    mgr.append_generation(1, pop, summary)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "run.jsonl"
        mgr.save_to_path(str(path))
        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        for i, line in enumerate(lines):
            entry = parse_run_log_entry(line)
            assert entry["generation_index"] == i
            assert len(entry["agents"]) == 2


def test_run_log_manager_load_from_path():
    mgr = RunLogManager()
    pop = _make_population(1)
    summary = {"elo_min": 1500.0, "elo_mean": 1500.0, "elo_max": 1500.0, "num_agents": 1.0}
    entry = build_run_log_entry(0, pop, summary)
    line = json.dumps(entry, ensure_ascii=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "saved_run.jsonl"
        path.write_text(line + "\n", encoding="utf-8")

        log_id = mgr.load_from_path(str(path))
        assert isinstance(log_id, str)
        assert len(log_id) > 0
        loaded = mgr.get_loaded_logs()
        assert len(loaded) == 1
        assert loaded[0].id == log_id
        assert loaded[0].path == str(path.resolve())
        assert len(loaded[0].entries) == 1
        assert loaded[0].entries[0]["generation_index"] == 0


def test_run_log_manager_auto_save():
    mgr = RunLogManager()
    pop = _make_population(1)
    summary = {"elo_min": 1500.0, "elo_mean": 1500.0, "elo_max": 1500.0, "num_agents": 1.0}

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr.set_auto_save(tmpdir, "auto.jsonl")
        assert mgr.get_auto_save_dir() == tmpdir
        assert mgr.get_auto_save_filename() == "auto.jsonl"

        mgr.append_generation(0, pop, summary)
        path = Path(tmpdir) / "auto.jsonl"
        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        entry = parse_run_log_entry(lines[0])
        assert entry["generation_index"] == 0

        mgr.append_generation(1, pop, summary)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        mgr.set_auto_save(None, None)
        assert mgr.get_auto_save_dir() is None
        assert mgr.get_auto_save_filename() is None
        mgr.append_generation(2, pop, summary)
        assert len(path.read_text(encoding="utf-8").strip().split("\n")) == 2


def test_run_log_manager_multiple_loads():
    mgr = RunLogManager()
    pop = _make_population(1)
    summary = {"elo_min": 1500.0, "elo_mean": 1500.0, "elo_max": 1500.0, "num_agents": 1.0}
    entry = build_run_log_entry(0, pop, summary)
    line = json.dumps(entry, ensure_ascii=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = Path(tmpdir) / "run1.jsonl"
        path2 = Path(tmpdir) / "run2.jsonl"
        path1.write_text(line + "\n", encoding="utf-8")
        path2.write_text(line + "\n", encoding="utf-8")

        id1 = mgr.load_from_path(str(path1))
        id2 = mgr.load_from_path(str(path2))
        assert id1 != id2
        loaded = mgr.get_loaded_logs()
        assert len(loaded) == 2
        ids = {log.id for log in loaded}
        assert ids == {id1, id2}


def test_run_log_manager_get_loaded_log():
    mgr = RunLogManager()
    pop = _make_population(1)
    entry = build_run_log_entry(0, pop, {"elo_min": 1500.0, "elo_mean": 1500.0, "elo_max": 1500.0, "num_agents": 1.0})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        path = f.name
    try:
        log_id = mgr.load_from_path(path)
        log = mgr.get_loaded_log(log_id)
        assert log is not None
        assert log.id == log_id
        assert len(log.entries) == 1
        assert mgr.get_loaded_log("nonexistent") is None
    finally:
        Path(path).unlink(missing_ok=True)


def test_run_log_manager_remove_loaded_log():
    mgr = RunLogManager()
    pop = _make_population(1)
    entry = build_run_log_entry(0, pop, {"elo_min": 1500.0, "elo_mean": 1500.0, "elo_max": 1500.0, "num_agents": 1.0})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        path = f.name
    try:
        log_id = mgr.load_from_path(path)
        assert len(mgr.get_loaded_logs()) == 1
        assert mgr.remove_loaded_log(log_id) is True
        assert len(mgr.get_loaded_logs()) == 0
        assert mgr.remove_loaded_log(log_id) is False
    finally:
        Path(path).unlink(missing_ok=True)
