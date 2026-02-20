"""Tests for project save/load, export/import JSON, and log persistence."""

import json
import tempfile
from pathlib import Path

import pytest

from tarot.ga import GAConfig
from tarot.league import LeagueConfig, run_league_generations
from tarot.project import (
    append_league_log,
    load_league_log,
    project_export_json,
    project_import_json,
    project_load,
    project_save,
)
from tarot.tournament import Agent, Population


def _make_population(size: int = 4) -> Population:
    pop = Population()
    for i in range(size):
        pop.add(
            Agent(
                id=f"A{i}",
                name=f"A{i}",
                player_counts=[4],
            )
        )
    return pop


def _make_groups_tuples() -> list:
    pop = _make_population(4)
    agents = list(pop.agents.values())
    return [
        ("grp_0", "Test Group", agents, None, "Test source", 0x4A90D9),
    ]


def test_project_save_and_load(tmp_path: Path) -> None:
    groups = _make_groups_tuples()
    cfg = LeagueConfig(
        player_count=4,
        deals_per_match=2,
        rounds_per_generation=1,
        ga_config=GAConfig(population_size=4, elite_fraction=0.25, mutation_prob=0.5, mutation_std=0.1),
    )
    project_save(
        tmp_path,
        groups=groups,
        league_config=cfg,
        generation_index=2,
        last_summary={"elo_min": 1400, "elo_mean": 1500, "elo_max": 1600, "num_agents": 4.0},
    )

    assert (tmp_path / "project.json").exists()
    assert (tmp_path / "checkpoints").exists()
    assert (tmp_path / "logs").exists()

    data = project_load(tmp_path)
    assert data["generation_index"] == 2
    assert data["last_summary"]["elo_mean"] == 1500
    assert len(data["groups_data"]) == 1
    g = data["groups_data"][0]
    assert g[1] == "Test Group"
    assert len(g[2]) == 4
    assert data["league_config"].player_count == 4
    assert data["league_config"].deals_per_match == 2
    assert data["league_config"].ga_config is not None
    assert data["league_config"].ga_config.population_size == 4


def test_project_save_and_load_with_sexual_reproduction_params(tmp_path: Path) -> None:
    """Save and load a project with count-based GA and sexual reproduction params; they round-trip."""
    groups = _make_groups_tuples()
    cfg = LeagueConfig(
        player_count=4,
        deals_per_match=2,
        rounds_per_generation=1,
        ga_config=GAConfig(
            population_size=4,
            sexual_offspring_count=1,
            mutate_count=2,
            clone_count=1,
            sexual_parent_with_replacement=False,
            sexual_parent_fitness_weighted=False,
            sexual_trait_combination="crossover",
            mutation_prob=0.5,
            mutation_std=0.1,
        ),
    )
    project_save(
        tmp_path,
        groups=groups,
        league_config=cfg,
        generation_index=0,
        last_summary={"elo_min": 1500, "elo_mean": 1500, "elo_max": 1500, "num_agents": 4.0},
    )

    data = project_load(tmp_path)
    ga = data["league_config"].ga_config
    assert ga is not None
    assert ga.sexual_offspring_count == 1
    assert ga.mutate_count == 2
    assert ga.clone_count == 1
    assert ga.sexual_parent_with_replacement is False
    assert ga.sexual_parent_fitness_weighted is False
    assert ga.sexual_trait_combination == "crossover"


def test_project_export_and_import_json(tmp_path: Path) -> None:
    groups = _make_groups_tuples()
    cfg = LeagueConfig(player_count=4, deals_per_match=1, ga_config=None)
    json_file = tmp_path / "export.json"

    project_export_json(
        json_file,
        groups=groups,
        league_config=cfg,
        generation_index=0,
        logs=[{"generation_index": 0, "elo_mean": 1500}],
    )

    assert json_file.exists()
    with json_file.open() as f:
        payload = json.load(f)
    assert payload["type"] == "tarot_project_export"
    assert payload["generation_index"] == 0
    assert len(payload["logs"]) == 1

    data = project_import_json(json_file)
    assert data["generation_index"] == 0
    assert data["logs"][0]["elo_mean"] == 1500
    assert data["league_config"].player_count == 4


def test_append_and_load_league_log(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    log_file = log_dir / "league_run.jsonl"

    append_league_log(tmp_path, 0, {"elo_min": 1400, "elo_mean": 1500, "elo_max": 1600, "num_agents": 4.0})
    append_league_log(tmp_path, 1, {"elo_min": 1450, "elo_mean": 1550, "elo_max": 1650, "num_agents": 4.0})

    entries = load_league_log(tmp_path)
    assert len(entries) == 2
    assert entries[0]["generation_index"] == 0
    assert entries[0]["elo_mean"] == 1500
    assert entries[1]["generation_index"] == 1
    assert entries[1]["elo_mean"] == 1550


def test_load_league_log_empty(tmp_path: Path) -> None:
    entries = load_league_log(tmp_path)
    assert entries == []


def test_run_league_generations_writes_log(tmp_path: Path) -> None:
    pop = _make_population()
    cfg = LeagueConfig(
        player_count=4,
        deals_per_match=1,
        rounds_per_generation=1,
        ppo_top_k=0,
        ga_config=GAConfig(population_size=4, elite_fraction=0.25, mutation_prob=1.0, mutation_std=0.1),
    )
    log_path = tmp_path / "logs" / "league_run.jsonl"

    gen = run_league_generations(
        pop, cfg, num_generations=2, rng=__import__("random").Random(42),
        log_path=log_path,
    )
    list(gen)

    assert log_path.exists()
    entries = load_league_log(tmp_path)
    assert len(entries) == 2
