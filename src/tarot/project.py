"""
Project save/load for league runs.

A project is a directory containing:
- project.json: metadata, groups, league config, generation index
- checkpoints/: agent checkpoints (saved during league run)
- logs/: league_run.jsonl (per-generation ELO metrics)

Export to single JSON is supported for easy sharing (config + population + logs).
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ga import GAConfig
from .league import LeagueConfig
from .persistence import _agent_from_dict, _agent_to_dict
from .tournament import Agent

PROJECT_SCHEMA_VERSION = 1
PROJECT_JSON = "project.json"
LOGS_DIR = "logs"
CHECKPOINTS_DIR = "checkpoints"
LEAGUE_LOG_FILE = "league_run.jsonl"


def _league_config_to_dict(cfg: LeagueConfig) -> Dict[str, Any]:
    return asdict(cfg)


def _league_config_from_dict(d: Dict[str, Any]) -> LeagueConfig:
    ga = d.get("ga_config")
    ga_config = GAConfig(**ga) if ga else None
    return LeagueConfig(
        player_count=int(d.get("player_count", 4)),
        deals_per_match=int(d.get("deals_per_match", 5)),
        rounds_per_generation=int(d.get("rounds_per_generation", 3)),
        matchmaking_style=d.get("matchmaking_style", "random"),
        elo_k_factor=float(d.get("elo_k_factor", 32.0)),
        elo_margin_scale=float(d.get("elo_margin_scale", 50.0)),
        ppo_top_k=int(d.get("ppo_top_k", 0)),
        ppo_updates_per_agent=int(d.get("ppo_updates_per_agent", 0)),
        ga_config=ga_config,
        fitness_weight_global_elo=float(d.get("fitness_weight_global_elo", 1.0)),
        fitness_weight_avg_score=float(d.get("fitness_weight_avg_score", 0.0)),
    )


def _group_to_dict(
    group_id: str,
    group_name: str,
    agents: List[Agent],
    *,
    source_group_id: Optional[str] = None,
    source_group_name: Optional[str] = None,
    color: int = 0x4A90D9,
) -> Dict[str, Any]:
    return {
        "id": group_id,
        "name": group_name,
        "agents": [_agent_to_dict(a) for a in agents],
        "source_group_id": source_group_id,
        "source_group_name": source_group_name,
        "color": color,
    }


def _groups_from_dict(
    groups_data: List[Dict[str, Any]],
) -> List[tuple]:
    """Returns list of (group_id, group_name, agents, source_group_id, source_group_name, color)."""
    result = []
    for g in groups_data:
        agents = [_agent_from_dict(a) for a in g.get("agents", [])]
        result.append((
            g.get("id", ""),
            g.get("name", ""),
            agents,
            g.get("source_group_id"),
            g.get("source_group_name"),
            int(g.get("color", 0x4A90D9)),
        ))
    return result


def _copy_checkpoint_into_project(
    project_dir: Path,
    agent: Agent,
    agent_dict: Dict[str, Any],
) -> None:
    """Copy agent checkpoint into project/checkpoints/ if it exists externally. Update agent_dict in place."""
    path = agent.checkpoint_path
    if not path:
        return
    src = Path(path)
    if not src.exists():
        return
    # Normalize: if already under project_dir, just ensure relative path
    try:
        src.resolve().relative_to(project_dir.resolve())
        # Already inside project
        agent_dict["checkpoint_path"] = str(src.relative_to(project_dir))
        return
    except ValueError:
        pass
    # Copy into project
    dest_name = f"league_4p_agent_{agent.id}"
    dest_dir = project_dir / CHECKPOINTS_DIR / dest_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dest_dir / f.name)
    agent_dict["checkpoint_path"] = f"{CHECKPOINTS_DIR}/{dest_name}"


def project_save(
    project_dir: Path | str,
    *,
    groups: List[tuple],
    league_config: LeagueConfig,
    generation_index: int = 0,
    last_summary: Optional[Dict[str, float]] = None,
    league_ui: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save project to directory.

    Args:
        project_dir: Project root directory (created if needed).
        groups: List of (group_id, group_name, agents, source_group_id, source_group_name, color).
        league_config: League configuration.
        generation_index: Last completed generation (0 if not yet run).
        last_summary: Last ELO summary dict.
        league_ui: Optional UI state (checkboxes, export/next-gen params) to persist.
    """
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / LOGS_DIR).mkdir(exist_ok=True)
    (project_dir / CHECKPOINTS_DIR).mkdir(exist_ok=True)

    groups_data: List[Dict[str, Any]] = []
    for t in groups:
        gid, gname, agents, src_id, src_name, color = t
        group_agents = []
        for a in agents:
            ad = _agent_to_dict(a)
            _copy_checkpoint_into_project(project_dir, a, ad)
            group_agents.append(ad)
        groups_data.append({
            "id": gid,
            "name": gname,
            "agents": group_agents,
            "source_group_id": src_id,
            "source_group_name": src_name,
            "color": color,
        })

    payload: Dict[str, Any] = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "groups": groups_data,
        "league_config": _league_config_to_dict(league_config),
        "generation_index": generation_index,
        "last_summary": last_summary,
    }
    if league_ui is not None:
        payload["league_ui"] = league_ui

    with (project_dir / PROJECT_JSON).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def project_load(project_dir: Path | str) -> Dict[str, Any]:
    """
    Load project from directory.

    Returns dict with keys:
        groups_data: List of (group_id, group_name, agents, source_group_id, source_group_name, color).
        league_config: LeagueConfig instance.
        generation_index: int.
        last_summary: dict | None.
        project_dir: Path (resolved, for resolving relative checkpoint paths).
    """
    project_dir = Path(project_dir).resolve()
    with (project_dir / PROJECT_JSON).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    groups_data = _groups_from_dict(payload.get("groups", []))
    # Resolve relative checkpoint paths to absolute
    for _, _, agents, _, _, _ in groups_data:
        for a in agents:
            if a.checkpoint_path and not Path(a.checkpoint_path).is_absolute():
                a.checkpoint_path = str((project_dir / a.checkpoint_path).resolve())

    return {
        "groups_data": groups_data,
        "league_config": _league_config_from_dict(payload.get("league_config", {})),
        "generation_index": int(payload.get("generation_index", 0)),
        "last_summary": payload.get("last_summary"),
        "project_dir": project_dir,
        "league_ui": payload.get("league_ui"),
    }


def project_export_json(
    json_path: Path | str,
    *,
    groups: List[tuple],
    league_config: LeagueConfig,
    generation_index: int = 0,
    last_summary: Optional[Dict[str, float]] = None,
    logs: Optional[List[Dict[str, Any]]] = None,
    project_dir: Optional[Path | str] = None,
    league_ui: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Export project to a single JSON file for easy sharing.

    Checkpoint paths remain as stored in agents (relative if from project).
    Logs are embedded if provided.

    Args:
        json_path: Output JSON file path.
        groups: Same as project_save.
        league_config: League configuration.
        generation_index: Last completed generation.
        last_summary: Last ELO summary.
        logs: Optional list of log entries (from load_league_log).
        project_dir: If provided, checkpoint paths are made relative to project_dir.
    """
    json_path = Path(json_path)
    project_dir = Path(project_dir).resolve() if project_dir else None

    groups_data: List[Dict[str, Any]] = []
    for t in groups:
        gid, gname, agents, src_id, src_name, color = t
        group_agents = []
        for a in agents:
            ad = _agent_to_dict(a)
            if project_dir and a.checkpoint_path:
                p = Path(a.checkpoint_path)
                if p.is_absolute():
                    try:
                        ad["checkpoint_path"] = str(p.relative_to(project_dir))
                    except ValueError:
                        pass
            group_agents.append(ad)
        groups_data.append({
            "id": gid,
            "name": gname,
            "agents": group_agents,
            "source_group_id": src_id,
            "source_group_name": src_name,
            "color": color,
        })

    payload: Dict[str, Any] = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "type": "tarot_project_export",
        "groups": groups_data,
        "league_config": _league_config_to_dict(league_config),
        "generation_index": generation_index,
        "last_summary": last_summary,
        "logs": logs,
    }
    if league_ui is not None:
        payload["league_ui"] = league_ui

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def project_import_json(json_path: Path | str) -> Dict[str, Any]:
    """
    Import project from a single JSON file (from project_export_json).

    Returns same structure as project_load, but project_dir is the JSON file's parent.
    Checkpoint paths in agents are left as-is (may be relative); caller should resolve
    against a project directory when loading into a project.
    """
    json_path = Path(json_path).resolve()
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if payload.get("type") != "tarot_project_export":
        raise ValueError("Not a Tarot project export JSON")

    groups_data = _groups_from_dict(payload.get("groups", []))
    return {
        "groups_data": groups_data,
        "league_config": _league_config_from_dict(payload.get("league_config", {})),
        "generation_index": int(payload.get("generation_index", 0)),
        "last_summary": payload.get("last_summary"),
        "project_dir": json_path.parent,
        "logs": payload.get("logs"),
        "league_ui": payload.get("league_ui"),
    }


def append_league_log(project_dir: Path | str, generation_index: int, summary: Dict[str, float]) -> None:
    """Append one log entry for a completed generation."""
    project_dir = Path(project_dir)
    log_path = project_dir / LOGS_DIR / LEAGUE_LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "generation_index": generation_index,
        "elo_min": summary.get("elo_min", 0),
        "elo_mean": summary.get("elo_mean", 0),
        "elo_max": summary.get("elo_max", 0),
        "num_agents": summary.get("num_agents", 0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_league_log(project_dir: Path | str) -> List[Dict[str, Any]]:
    """Load league run log entries from project directory."""
    project_dir = Path(project_dir)
    log_path = project_dir / LOGS_DIR / LEAGUE_LOG_FILE
    if not log_path.exists():
        return []
    entries = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def get_checkpoint_base_dir(project_dir: Path | str) -> Path:
    """Return the checkpoints directory path for a project (used by league run)."""
    return Path(project_dir) / CHECKPOINTS_DIR


def get_log_path(project_dir: Path | str) -> Path:
    """Return the league log file path for a project."""
    return Path(project_dir) / LOGS_DIR / LEAGUE_LOG_FILE


__all__ = [
    "append_league_log",
    "load_league_log",
    "project_export_json",
    "project_import_json",
    "project_load",
    "project_save",
    "get_checkpoint_base_dir",
    "get_log_path",
]
