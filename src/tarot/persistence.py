"""
Population serialization for import/export.

Exports and imports Population to/from JSON-compatible dicts for saving and
loading league populations.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

from .tournament import Agent, Population

SCHEMA_VERSION = 1


def _agent_to_dict(agent: Agent) -> Dict[str, Any]:
    return {
        "id": agent.id,
        "name": agent.name,
        "player_counts": list(agent.player_counts),
        "elo_3p": agent.elo_3p,
        "elo_4p": agent.elo_4p,
        "elo_5p": agent.elo_5p,
        "elo_global": agent.elo_global,
        "generation": agent.generation,
        "traits": dict(agent.traits),
        "checkpoint_path": agent.checkpoint_path,
        "arch_name": agent.arch_name,
        "parents": list(agent.parents),
        "can_use_as_ga_parent": agent.can_use_as_ga_parent,
        "fixed_elo": agent.fixed_elo,
        "clone_only": agent.clone_only,
        "play_in_league": agent.play_in_league,
        "matches_played": agent.matches_played,
        "total_match_score": agent.total_match_score,
    }


def _agent_from_dict(d: Dict[str, Any]) -> Agent:
    return Agent(
        id=d["id"],
        name=d["name"],
        player_counts=list(d["player_counts"]),
        elo_3p=float(d.get("elo_3p", 1500.0)),
        elo_4p=float(d.get("elo_4p", 1500.0)),
        elo_5p=float(d.get("elo_5p", 1500.0)),
        elo_global=float(d.get("elo_global", 1500.0)),
        generation=int(d.get("generation", 0)),
        traits=dict(d.get("traits", {})),
        checkpoint_path=d.get("checkpoint_path"),
        arch_name=d.get("arch_name"),
        parents=list(d.get("parents", [])),
        can_use_as_ga_parent=bool(d.get("can_use_as_ga_parent", True)),
        fixed_elo=bool(d.get("fixed_elo", False)),
        clone_only=bool(d.get("clone_only", False)),
        play_in_league=bool(d.get("play_in_league", True)),
        matches_played=int(d.get("matches_played", 0)),
        total_match_score=float(d.get("total_match_score", 0.0)),
    )


def population_to_dict(
    pop: Population,
    *,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Serialize a Population to a JSON-compatible dict.

    Args:
        pop: The population to serialize.
        metadata: Optional extra metadata (e.g. league config snapshot).

    Returns:
        Dict with schema_version, exported_at, agents, and optional metadata.
    """
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "agents": [_agent_to_dict(a) for a in pop.agents.values()],
    }
    if metadata:
        result["metadata"] = metadata
    return result


def population_from_dict(d: Dict[str, Any]) -> Population:
    """
    Deserialize a Population from a dict (e.g. from JSON).

    Args:
        d: Dict produced by population_to_dict (or compatible).

    Returns:
        Restored Population.
    """
    pop = Population()
    for agent_d in d.get("agents", []):
        agent = _agent_from_dict(agent_d)
        pop.add(agent)
    return pop


def population_to_json(
    pop: Population,
    *,
    metadata: Dict[str, Any] | None = None,
) -> str:
    """Serialize a Population to a JSON string."""
    return json.dumps(population_to_dict(pop, metadata=metadata), indent=2)


def population_from_json(s: str) -> Population:
    """Deserialize a Population from a JSON string."""
    return population_from_dict(json.loads(s))


__all__ = [
    "population_to_dict",
    "population_from_dict",
    "population_to_json",
    "population_from_json",
    "SCHEMA_VERSION",
]
