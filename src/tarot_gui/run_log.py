"""
Run log for Dashboard: in-memory per-generation entries, JSONL persistence, multiple loaded logs.

Step 1 (Dashboard): RunLogManager holds current run log and loaded logs; user-defined path/name
for auto-save each generation; Save/Load buttons write or read JSONL.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tarot.persistence import _agent_to_dict
from tarot.tournament import Agent, Population

# Schema version for each JSONL line (run log entry)
RUN_LOG_SCHEMA_VERSION = 1


def _agent_snapshot(agent: Agent) -> Dict[str, Any]:
    """Per-agent snapshot for one generation (same shape as persistence for compatibility)."""
    return _agent_to_dict(agent)


def build_run_log_entry(
    generation_index: int,
    population: Population,
    summary: Dict[str, float],
    *,
    timestamp: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Build one run log entry (one JSONL line) for the given generation.

    Args:
        generation_index: 0-based generation index.
        population: Population after this generation (for per-agent snapshot).
        summary: League summary dict (elo_min, elo_mean, elo_max, num_agents).
        timestamp: Optional UTC time; default now.

    Returns:
        Dict suitable for JSON serialization (one line of JSONL).
    """
    ts = timestamp or datetime.now(timezone.utc)
    agents_data = [_agent_snapshot(a) for a in population.agents.values()]
    entry: Dict[str, Any] = {
        "schema_version": RUN_LOG_SCHEMA_VERSION,
        "generation_index": generation_index,
        "timestamp_utc": ts.isoformat(),
        "elo_min": summary.get("elo_min", 0.0),
        "elo_mean": summary.get("elo_mean", 0.0),
        "elo_max": summary.get("elo_max", 0.0),
        "num_agents": summary.get("num_agents", len(population.agents)),
        "agents": agents_data,
    }
    if "deals" in summary or "petit_au_bout" in summary or "grand_slem" in summary:
        entry["game_metrics"] = {
            "deals": int(summary.get("deals", 0)),
            "petit_au_bout": int(summary.get("petit_au_bout", 0)),
            "grand_slem": int(summary.get("grand_slem", 0)),
        }
    return entry


def parse_run_log_entry(line: str) -> Dict[str, Any]:
    """Parse one JSONL line into a run log entry dict. Raises on invalid JSON."""
    return json.loads(line.strip())


@dataclass
class LoadedRunLog:
    """One loaded run log (from file)."""

    id: str
    path: str
    entries: List[Dict[str, Any]] = field(default_factory=list)


class RunLogManager:
    """
    Holds current run log (in-memory) and list of loaded run logs.
    Auto-save appends one JSONL line per generation to a user-defined path.
    """

    def __init__(self) -> None:
        self._current_entries: List[Dict[str, Any]] = []
        self._loaded: List[LoadedRunLog] = []
        self._auto_save_dir: Optional[str] = None
        self._auto_save_filename: Optional[str] = None

    def set_auto_save(self, directory: Optional[str], filename: Optional[str]) -> None:
        """Set where to auto-save each generation. None clears."""
        self._auto_save_dir = directory
        self._auto_save_filename = filename

    def get_auto_save_dir(self) -> Optional[str]:
        return self._auto_save_dir

    def get_auto_save_filename(self) -> Optional[str]:
        return self._auto_save_filename

    def append_generation(
        self,
        gen_idx: int,
        population: Population,
        summary: Dict[str, float],
    ) -> None:
        """
        Append one generation to the current run log and optionally auto-save to disk.
        """
        entry = build_run_log_entry(gen_idx, population, summary)
        self._current_entries.append(entry)
        if self._auto_save_dir and self._auto_save_filename:
            path = Path(self._auto_save_dir) / self._auto_save_filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def clear_current(self) -> None:
        """Clear current run log (e.g. when starting a new run)."""
        self._current_entries.clear()

    def get_current_entries(self) -> List[Dict[str, Any]]:
        """Return a copy of current run log entries."""
        return list(self._current_entries)

    def has_current_data(self) -> bool:
        """True if there is at least one generation in the current run log."""
        return len(self._current_entries) > 0

    def save_to_path(self, file_path: str) -> None:
        """
        Write current run log to the given path (JSONL, one line per generation).
        Overwrites the file if it exists.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for entry in self._current_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def load_from_path(self, file_path: str) -> str:
        """
        Load a run log from a JSONL file and add it to the loaded logs.
        Returns the id of the newly loaded log.
        """
        path = Path(file_path).resolve()
        entries: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(parse_run_log_entry(line))
        log_id = f"{path.stem}_{uuid.uuid4().hex[:8]}"
        self._loaded.append(LoadedRunLog(id=log_id, path=str(path), entries=entries))
        return log_id

    def get_loaded_logs(self) -> List[LoadedRunLog]:
        """Return list of loaded run logs (read-only view)."""
        return list(self._loaded)

    def get_loaded_log(self, log_id: str) -> Optional[LoadedRunLog]:
        """Return the loaded log with the given id, or None."""
        for log in self._loaded:
            if log.id == log_id:
                return log
        return None

    def remove_loaded_log(self, log_id: str) -> bool:
        """Remove a loaded log by id. Returns True if found and removed."""
        for i, log in enumerate(self._loaded):
            if log.id == log_id:
                self._loaded.pop(i)
                return True
        return False
