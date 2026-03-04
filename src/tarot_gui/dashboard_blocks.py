"""
Dashboard tab blocks: Compute (time used/ETA/avg per gen), ELO (summary + time series chart).

Uses same approach as League tab charts: custom QPainter-based widgets, no matplotlib.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets


def _format_duration(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    if seconds < 0 or not isinstance(seconds, (int, float)):
        return "0:00"
    total = int(round(seconds))
    if total >= 3600:
        h, r = divmod(total, 3600)
        m, s = divmod(r, 60)
        return f"{h}:{m:02d}:{s:02d}"
    m, s = divmod(total, 60)
    return f"{m}:{s:02d}"


class ComputeBlockWidget(QtWidgets.QGroupBox):
    """
    Compute block: Time used, Time left (ETA), Average time per generation.
    Data from run start time and per-generation timestamps (passed by MainWindow on generation_done).
    No CPU/GPU yet.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Compute", parent)
        layout = QtWidgets.QHBoxLayout(self)
        self._label_time_used = QtWidgets.QLabel("Time used: —")
        self._label_eta = QtWidgets.QLabel("Time left (ETA): —")
        self._label_avg = QtWidgets.QLabel("Avg time/gen: —")
        layout.addWidget(self._label_time_used)
        layout.addWidget(self._label_eta)
        layout.addWidget(self._label_avg)
        layout.addStretch(1)

    def update_metrics(
        self,
        elapsed_seconds: Optional[float] = None,
        eta_seconds: Optional[float] = None,
        avg_seconds_per_gen: Optional[float] = None,
    ) -> None:
        """Update the three metrics. Pass None for any to show '—'."""
        self._label_time_used.setText(
            f"Time used: {_format_duration(elapsed_seconds) if elapsed_seconds is not None else '—'}"
        )
        self._label_eta.setText(
            f"Time left (ETA): {_format_duration(eta_seconds) if eta_seconds is not None else '—'}"
        )
        self._label_avg.setText(
            f"Avg time/gen: {_format_duration(avg_seconds_per_gen) if avg_seconds_per_gen is not None else '—'}"
        )

    def clear_metrics(self) -> None:
        """Reset all to '—' (e.g. when run finishes)."""
        self.update_metrics(None, None, None)


def _elo_stats_from_entries(entries: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """From run log entries, compute min/mean/max/std of ELO for the latest generation. Returns None if no data."""
    if not entries:
        return None
    last = entries[-1]
    agents = last.get("agents", [])
    if not agents:
        elos = [last.get("elo_mean", 0.0)]
    else:
        elos = [float(a.get("elo_global", 0.0)) for a in agents]
    if not elos:
        return None
    n = len(elos)
    mean = sum(elos) / n
    variance = sum((x - mean) ** 2 for x in elos) / n if n > 0 else 0.0
    std = math.sqrt(variance) if variance >= 0 else 0.0
    return {"min": min(elos), "mean": mean, "max": max(elos), "std": std}


class ELOBlockWidget(QtWidgets.QGroupBox):
    """
    Statistics block: summary (min, mean, max, std) + optional game metrics
    and a time series chart (ELO / Fitness / Origin) over generations.
    Data from RunLogManager current entries or selected loaded log.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Statistics", parent)
        self._entries: List[Dict[str, Any]] = []
        layout = QtWidgets.QVBoxLayout(self)

        # Top row: ELO summary on the left, game metrics on the right
        top_row = QtWidgets.QHBoxLayout()
        self._summary_label = QtWidgets.QLabel("min: —  mean: —  max: —  std: —")
        top_row.addWidget(self._summary_label)
        top_row.addStretch(1)
        self._game_metrics_label = QtWidgets.QLabel("Deals: —  Petit au bout: —  Grand schlem: —")
        top_row.addWidget(self._game_metrics_label)
        layout.addLayout(top_row)

        # Mode selector row (above the chart)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addStretch(1)
        mode_row.addWidget(QtWidgets.QLabel("View:"))
        self._mode_combo = QtWidgets.QComboBox()
        self._mode_combo.addItems(["ELO", "Fitness", "Origin"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self._mode_combo)
        layout.addLayout(mode_row)

        self._chart = _ELOChartWidget(self)
        self._chart.setMinimumHeight(270)
        layout.addWidget(self._chart)

    def set_total_generations(self, total: Optional[int]) -> None:
        """
        Fix the X axis [0, total-1] for the current run.
        Pass None or <=0 to let the chart autoscale to data.
        """
        if total is None or total <= 0:
            self._chart.set_generation_bounds(None, None)
        else:
            self._chart.set_generation_bounds(0, total - 1)

    def set_entries(self, entries: List[Dict[str, Any]]) -> None:
        """Set run log entries (e.g. from RunLogManager.get_current_entries()). Updates summary and chart."""
        self._entries = list(entries) if entries else []
        stats = _elo_stats_from_entries(self._entries)
        if stats is None:
            self._summary_label.setText("min: —  mean: —  max: —  std: —")
        else:
            self._summary_label.setText(
                f"min: {stats['min']:.0f}  mean: {stats['mean']:.0f}  max: {stats['max']:.0f}  std: {stats['std']:.1f}"
            )
        # Aggregate game metrics across the run (League scope)
        if not self._entries:
            self._game_metrics_label.setText("Deals: —  Petit au bout: —  Grand schlem: —")
        else:
            deals = sum(int(e.get("game_metrics", {}).get("deals", 0)) for e in self._entries)
            petit = sum(int(e.get("game_metrics", {}).get("petit_au_bout", 0)) for e in self._entries)
            grand = sum(int(e.get("game_metrics", {}).get("grand_slem", 0)) for e in self._entries)
            self._game_metrics_label.setText(
                f"Deals: {deals}  Petit au bout: {petit}  Grand schlem: {grand}"
            )
        self._chart.set_entries(self._entries)
        self._chart.update()

    def _on_mode_changed(self) -> None:
        text = self._mode_combo.currentText() if hasattr(self, "_mode_combo") else "ELO"
        mode = text.lower()
        if mode not in ("elo", "fitness", "origin"):
            mode = "elo"
        self._chart.set_mode(mode)


class _ELOChartWidget(QtWidgets.QWidget):
    """Time series over generations for ELO / Fitness, or Origin placeholder."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._entries: List[Dict[str, Any]] = []
        self._gen_min_override: Optional[int] = None
        self._gen_max_override: Optional[int] = None
        self._mode: str = "elo"  # "elo" | "fitness" | "origin"
        self.setMinimumWidth(260)

    def set_entries(self, entries: List[Dict[str, Any]]) -> None:
        self._entries = list(entries) if entries else []

    def set_mode(self, mode: str) -> None:
        mode = (mode or "").lower()
        if mode not in ("elo", "fitness", "origin"):
            mode = "elo"
        if self._mode != mode:
            self._mode = mode
            self.update()

    def set_generation_bounds(self, gen_min: Optional[int], gen_max: Optional[int]) -> None:
        """
        Optionally override the X-axis generation range.

        If both are None, the chart auto-scales to the data range.
        """
        self._gen_min_override = gen_min
        self._gen_max_override = gen_max
        self.update()

    def paintEvent(self, event: QtCore.QEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = self.rect()
            w, h = rect.width(), rect.height()
            pen_color = self.palette().color(QtGui.QPalette.ColorRole.WindowText)
            margin_left, margin_right = 36, 24
            margin_top, margin_bot = 16, 24
            gx, gy = margin_left, margin_top
            gw = max(1, w - margin_left - margin_right)
            gh = max(1, h - margin_top - margin_bot)

            # When no data and no explicit bounds, show a placeholder message.
            if not self._entries and self._gen_max_override is None:
                painter.setPen(QtGui.QPen(pen_color, 1))
                painter.setFont(QtGui.QFont(self.font().family(), 9))
                painter.drawText(
                    int(gx), int(gy), int(gw), int(gh),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    "No data",
                )
                return

            gens = [e.get("generation_index", 0) for e in self._entries]

            if self._mode == "elo":
                series_min = [e.get("elo_min", 0.0) for e in self._entries]
                series_mean = [e.get("elo_mean", 0.0) for e in self._entries]
                series_max = [e.get("elo_max", 0.0) for e in self._entries]
                y_label = "ELO"
            elif self._mode == "fitness":
                # Compute per-entry fitness min/mean/max from per-agent snapshots
                series_min = []
                series_mean = []
                series_max = []
                for e in self._entries:
                    agents = e.get("agents", [])
                    if not agents:
                        # Fallback: mirror ELO when no per-agent data
                        series_min.append(float(e.get("elo_min", 0.0)))
                        series_mean.append(float(e.get("elo_mean", 0.0)))
                        series_max.append(float(e.get("elo_max", 0.0)))
                        continue
                    vals: List[float] = []
                    for a in agents:
                        elo = max(0.0, float(a.get("elo_global", 0.0)))
                        matches_played = int(a.get("matches_played", 0))
                        total_score = float(a.get("total_match_score", 0.0))
                        avg_score = (total_score / matches_played) if matches_played > 0 else 0.0
                        vals.append(elo + avg_score)
                    if not vals:
                        vals = [0.0]
                    series_min.append(min(vals))
                    series_mean.append(sum(vals) / len(vals))
                    series_max.append(max(vals))
                y_label = "Fitness"
            else:  # origin mode – placeholder chart
                painter.setPen(QtGui.QPen(pen_color, 1))
                painter.setFont(QtGui.QFont(self.font().family(), 9))
                painter.drawText(
                    int(gx), int(gy), int(gw), int(gh),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    "Origin chart not yet available\n(run log lacks origin metadata).",
                )
                return

            elo_mins = series_min
            elo_means = series_mean
            elo_maxs = series_max
            steps = list(range(len(self._entries)))
            if steps:
                step_min, step_max = steps[0], steps[-1]
            else:
                step_min = step_max = 0
            # Use explicit generation bounds only for drawing generation labels/boundaries,
            # but the X coordinate is driven by match index (step index).
            if gens:
                data_gen_min, data_gen_max = min(gens), max(gens)
            else:
                data_gen_min = data_gen_max = 0
            gen_min = self._gen_min_override if self._gen_min_override is not None else data_gen_min
            gen_max = self._gen_max_override if self._gen_max_override is not None else data_gen_max
            if gen_max < gen_min:
                gen_max = gen_min
            step_range = (step_max - step_min) or 1
            all_vals = elo_mins + elo_means + elo_maxs
            val_lo = min(all_vals) if all_vals else 0.0
            val_hi = max(all_vals) if all_vals else 1500.0
            val_range = (val_hi - val_lo) or 1.0

            # Axes
            painter.setPen(QtGui.QPen(pen_color, 1))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRect(int(gx), int(gy), int(gw), int(gh))
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            painter.drawText(int(gx + gw // 2 - 25), int(h - 4), "Match")
            painter.save()
            painter.translate(8, gy + gh // 2 - 10)
            painter.rotate(-90)
            painter.drawText(-20, 0, y_label)
            painter.restore()

            # X ticks: match indices (0 .. last)
            for i in range(5):
                t = i / 4.0
                s = step_min + t * step_range
                fx = gx + t * (gw - 1)
                painter.drawLine(int(fx), int(gy + gh), int(fx), int(gy + gh + 4))
                painter.drawText(int(fx - 18), int(gy + gh + 12), f"{int(s)}")
            # Y ticks
            for i in range(5):
                t = i / 4.0
                val = val_lo + t * val_range
                fy = gy + gh - t * (gh - 1)
                painter.drawLine(int(gx - 4), int(fy), int(gx), int(fy))
                painter.drawText(int(gx - 30), int(fy + 4), f"{int(val)}")

            def path_for(values: List[float]) -> QtGui.QPainterPath:
                path = QtGui.QPainterPath()
                for i, val in enumerate(values):
                    t = (steps[i] - step_min) / step_range if step_range else 0
                    norm = (val - val_lo) / val_range if val_range else 0
                    norm = max(0.0, min(1.0, norm))
                    px = gx + t * (gw - 1)
                    py = gy + gh - norm * (gh - 1)
                    if i == 0:
                        path.moveTo(px, py)
                    else:
                        path.lineTo(px, py)
                return path

            # Min (blue), mean (green), max (red)
            for path, color in [
                (path_for(elo_mins), 0x4A90D9),
                (path_for(elo_means), 0x50C878),
                (path_for(elo_maxs), 0xD9534F),
            ]:
                painter.setPen(QtGui.QPen(QtGui.QColor((color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF), 1.5))
                painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                painter.drawPath(path)

            # Vertical generation boundaries and labels
            if len(gens) > 1:
                prev_gen = gens[0]
                boundary_steps: List[int] = []
                for i in range(1, len(gens)):
                    if gens[i] != prev_gen:
                        boundary_steps.append(i)
                        prev_gen = gens[i]
                painter.setPen(QtGui.QPen(QtGui.QColor(120, 120, 120), 1, QtCore.Qt.PenStyle.DashLine))
                for b in boundary_steps:
                    t = (b - step_min) / step_range if step_range else 0
                    fx = gx + t * (gw - 1)
                    painter.drawLine(int(fx), int(gy), int(fx), int(gy + gh))
                # Generation labels centered between boundaries
                unique_gens = sorted(set(gens))
                for g in unique_gens:
                    idxs = [i for i, gv in enumerate(gens) if gv == g]
                    if not idxs:
                        continue
                    center_step = (idxs[0] + idxs[-1]) / 2.0
                    t = (center_step - step_min) / step_range if step_range else 0
                    fx = gx + t * (gw - 1)
                    painter.setPen(QtGui.QPen(pen_color, 1))
                    painter.drawText(int(fx - 10), int(gy - 4), f"G{g}")

            # Legend
            leg_x = gx + gw + 4
            leg_y = gy
            for label, color in [("min", 0x4A90D9), ("mean", 0x50C878), ("max", 0xD9534F)]:
                painter.setBrush(QtGui.QBrush(QtGui.QColor((color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF)))
                painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawRect(int(leg_x), int(leg_y), 10, 10)
                painter.setPen(QtGui.QPen(pen_color, 1))
                painter.drawText(int(leg_x + 14), int(leg_y + 10), label)
                leg_y += 14
        finally:
            painter.end()


# --- RL performance block ---


def _delta_elo_from_entries(entries: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Per-agent ELO delta: current elo_global minus the ELO when that agent was
    first seen in the run. This gives a value for all agents (including those
    born in later generations), not only those present in generation 0.
    """
    if not entries:
        return {}
    # First occurrence of each agent id → elo_global
    first_elo: Dict[str, float] = {}
    for entry in entries:
        for a in entry.get("agents", []):
            aid = a.get("id", "")
            if aid not in first_elo:
                first_elo[aid] = float(a.get("elo_global", 0))
    cur_agents = entries[-1].get("agents", [])
    deltas: Dict[str, float] = {}
    for a in cur_agents:
        aid = a.get("id", "")
        cur_elo = float(a.get("elo_global", 0))
        base_elo = first_elo.get(aid, cur_elo)
        deltas[aid] = cur_elo - base_elo
    return deltas


class RLPerformanceBlockWidget(QtWidgets.QGroupBox):
    """
    Top-N table: name, ELO, Δ ELO, W/L deal, W/L match, avg score.
    N configurable (default 10). Data from run log entries (latest generation).
    """
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("RL performance", parent)
        self._entries: List[Dict[str, Any]] = []
        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()

        # Sort [Top/Bottom] [N] by [column]
        row.addWidget(QtWidgets.QLabel("Sort"))
        self._combo_direction = QtWidgets.QComboBox()
        self._combo_direction.addItems(["Top", "Bottom"])
        self._combo_direction.currentIndexChanged.connect(self._refresh_table)
        row.addWidget(self._combo_direction)

        self._spin_n = QtWidgets.QSpinBox()
        self._spin_n.setRange(1, 50)
        self._spin_n.setValue(10)
        self._spin_n.valueChanged.connect(self._refresh_table)
        row.addWidget(self._spin_n)

        row.addWidget(QtWidgets.QLabel("by"))
        self._combo_sort = QtWidgets.QComboBox()
        self._combo_sort.addItems(["ELO", "ELO delta", "W/L deal", "W/L match", "Average Score"])
        self._combo_sort.currentIndexChanged.connect(self._refresh_table)
        row.addWidget(self._combo_sort)

        row.addStretch()
        layout.addLayout(row)
        self._table = QtWidgets.QTableWidget()
        # Columns: Name, ELO, Δ, W/L deal, W/L match, Average Score
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            ["Name", "ELO", "Δ", "W/L deal", "W/L match", "Average Score"]
        )
        header = self._table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        # Make ELO and Δ columns narrower to free space for others
        self._table.setColumnWidth(1, 70)  # ELO
        self._table.setColumnWidth(2, 60)  # Δ
        layout.addWidget(self._table)

    def set_entries(self, entries: List[Dict[str, Any]]) -> None:
        self._entries = list(entries) if entries else []
        self._refresh_table()

    def _refresh_table(self) -> None:
        n = self._spin_n.value()
        self._table.setRowCount(0)
        if not self._entries:
            return
        agents = self._entries[-1].get("agents", [])
        if not agents:
            return
        deltas = _delta_elo_from_entries(self._entries)

        # Determine sort metric and direction (Top/Bottom)
        sort_mode = self._combo_sort.currentText() if hasattr(self, "_combo_sort") else "ELO"

        def sort_key(a: Dict[str, Any]) -> float:
            aid = a.get("id", "")
            elo = float(a.get("elo_global", 0))
            deals_p, deals_w = int(a.get("deals_played", 0)), int(a.get("deals_won", 0))
            match_p, match_w = int(a.get("matches_played", 0)), int(a.get("matches_won", 0))
            total_s = float(a.get("total_match_score", 0))

            if sort_mode == "ELO delta":
                return float(deltas.get(aid, 0.0))
            if sort_mode == "W/L deal":
                return (deals_w / deals_p) if deals_p > 0 else -1.0
            if sort_mode == "W/L match":
                return (match_w / match_p) if match_p > 0 else -1.0
            if sort_mode == "Average Score":
                return (total_s / match_p) if match_p > 0 else -1.0
            # Default: ELO
            return elo

        direction = self._combo_direction.currentText() if hasattr(self, "_combo_direction") else "Top"
        reverse = direction != "Bottom"
        sorted_agents = sorted(agents, key=sort_key, reverse=reverse)[:n]
        for a in sorted_agents:
            aid = a.get("id", "")
            deals_p, deals_w = int(a.get("deals_played", 0)), int(a.get("deals_won", 0))
            match_p, match_w = int(a.get("matches_played", 0)), int(a.get("matches_won", 0))
            total_s = float(a.get("total_match_score", 0))
            wl_d = f"{deals_w}/{deals_p}" if deals_p else "—"
            wl_m = f"{match_w}/{match_p}" if match_p else "—"
            avg = f"{total_s / match_p:.1f}" if match_p else "—"
            delta = deltas.get(aid, 0)
            delta_str = f"+{delta:.0f}" if delta > 0 else (f"{delta:.0f}" if delta < 0 else "0")
            row_idx = self._table.rowCount()
            self._table.insertRow(row_idx)
            self._table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(str(a.get("name", aid))))
            self._table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(f"{float(a.get('elo_global', 0)):.0f}"))
            delta_item = QtWidgets.QTableWidgetItem(delta_str)
            if delta > 0:
                delta_item.setForeground(QtGui.QBrush(QtGui.QColor(0, 160, 0)))
            elif delta < 0:
                delta_item.setForeground(QtGui.QBrush(QtGui.QColor(200, 0, 0)))
            self._table.setItem(row_idx, 2, delta_item)
            self._table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(wl_d))
            self._table.setItem(row_idx, 4, QtWidgets.QTableWidgetItem(wl_m))
            self._table.setItem(row_idx, 5, QtWidgets.QTableWidgetItem(avg))


# --- Game metrics block ---


class GameMetricsBlockWidget(QtWidgets.QGroupBox):
    """
    Deals, petit au bout, grand schlem. Scope: League | Generation | Last N generations.
    """
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Game metrics", parent)
        self._entries: List[Dict[str, Any]] = []
        layout = QtWidgets.QVBoxLayout(self)
        scope_row = QtWidgets.QHBoxLayout()
        scope_row.addWidget(QtWidgets.QLabel("Scope:"))
        self._scope_combo = QtWidgets.QComboBox()
        self._scope_combo.addItems(["League", "Generation", "Last N"])
        self._scope_combo.currentIndexChanged.connect(self._refresh_metrics)
        scope_row.addWidget(self._scope_combo)
        self._spin_last_n = QtWidgets.QSpinBox()
        self._spin_last_n.setRange(1, 999)
        self._spin_last_n.setValue(5)
        self._spin_last_n.valueChanged.connect(self._refresh_metrics)
        scope_row.addWidget(self._spin_last_n)
        scope_row.addStretch()
        layout.addLayout(scope_row)
        self._label_metrics = QtWidgets.QLabel("Deals: —  Petit au bout: —  Grand schlem: —")
        layout.addWidget(self._label_metrics)

    def set_entries(self, entries: List[Dict[str, Any]]) -> None:
        self._entries = list(entries) if entries else []
        self._refresh_metrics()

    def _refresh_metrics(self) -> None:
        if not self._entries:
            self._label_metrics.setText("Deals: —  Petit au bout: —  Grand schlem: —")
            return
        scope = self._scope_combo.currentIndex()
        n = self._spin_last_n.value()
        if scope == 0:
            entries = self._entries
        elif scope == 1:
            entries = self._entries[-1:] if self._entries else []
        else:
            entries = self._entries[-n:] if len(self._entries) >= n else self._entries
        deals = sum(int(e.get("game_metrics", {}).get("deals", 0)) for e in entries)
        petit = sum(int(e.get("game_metrics", {}).get("petit_au_bout", 0)) for e in entries)
        grand = sum(int(e.get("game_metrics", {}).get("grand_slem", 0)) for e in entries)
        self._label_metrics.setText(f"Deals: {deals}  Petit au bout: {petit}  Grand schlem: {grand}")


# --- Export block (run output → agents / Hall of Fame) ---


def _sort_agents_for_export(
    agents: list,
    order_by: str,
    reverse: bool,
    fitness_config: Optional[Any],
) -> list:
    """Sort agent list by ELO, average score, or fitness. Returns new list."""
    if order_by == "ELO":
        return sorted(agents, key=lambda a: a.elo_global, reverse=reverse)
    if order_by == "Average score":
        def avg_score(a):
            return (a.total_match_score / a.matches_played) if a.matches_played else 0.0
        return sorted(agents, key=avg_score, reverse=reverse)
    if order_by == "Fitness" and fitness_config:
        from tarot.ga import compute_fitness
        def fit(a):
            return compute_fitness(
                a,
                fitness_elo_a=getattr(fitness_config, "fitness_elo_a", 1.0),
                fitness_elo_b=getattr(fitness_config, "fitness_elo_b", 1.0),
                fitness_avg_c=getattr(fitness_config, "fitness_avg_c", 0.0),
                fitness_avg_d=getattr(fitness_config, "fitness_avg_d", 1.0),
            )
        return sorted(agents, key=fit, reverse=reverse)
    return sorted(agents, key=lambda a: a.elo_global, reverse=reverse)


class ExportBlockWidget(QtWidgets.QGroupBox):
    """
    Export run output to the project's agents/ or agents/Hall of Fame/ folder.
    Shown when there is run output (after a league run). User chooses order, range, and destination.
    """
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Export run output", parent)
        self._project_path: Optional[str] = None
        self._population: Optional[Any] = None
        self._generation_index: int = 0
        self._fitness_config: Optional[Any] = None

        layout = QtWidgets.QVBoxLayout(self)

        self._placeholder = QtWidgets.QLabel("No run output to export. Run a league to export the final population.")
        self._placeholder.setWordWrap(True)
        layout.addWidget(self._placeholder)

        form = QtWidgets.QWidget()
        form_layout = QtWidgets.QFormLayout(form)
        form_layout.setContentsMargins(0, 8, 0, 0)

        order_row = QtWidgets.QHBoxLayout()
        order_row.addWidget(QtWidgets.QLabel("Order by:"))
        self._combo_order = QtWidgets.QComboBox()
        self._combo_order.addItems(["ELO", "Average score", "Fitness"])
        self._combo_order.setToolTip("Sort agents before applying range (e.g. top N by ELO).")
        order_row.addWidget(self._combo_order)
        order_row.addStretch()
        form_layout.addRow(order_row)

        range_row = QtWidgets.QHBoxLayout()
        range_row.addWidget(QtWidgets.QLabel("Export:"))
        self._combo_range = QtWidgets.QComboBox()
        self._combo_range.addItems(["Whole population", "Top N", "Range (from–to)"])
        self._combo_range.currentIndexChanged.connect(self._on_range_mode_changed)
        range_row.addWidget(self._combo_range)
        self._spin_top_n = QtWidgets.QSpinBox()
        self._spin_top_n.setRange(1, 9999)
        self._spin_top_n.setValue(10)
        range_row.addWidget(self._spin_top_n)
        self._spin_from = QtWidgets.QSpinBox()
        self._spin_from.setRange(0, 9999)
        self._spin_from.setValue(0)
        range_row.addWidget(QtWidgets.QLabel("from"))
        range_row.addWidget(self._spin_from)
        self._label_to = QtWidgets.QLabel("to")
        range_row.addWidget(self._label_to)
        self._spin_to = QtWidgets.QSpinBox()
        self._spin_to.setRange(0, 9999)
        self._spin_to.setValue(9)
        range_row.addWidget(self._spin_to)
        range_row.addStretch()
        form_layout.addRow(range_row)
        self._on_range_mode_changed()

        btn_row = QtWidgets.QHBoxLayout()
        self._btn_export_agents = QtWidgets.QPushButton("Export to agents folder")
        self._btn_export_agents.setToolTip("Write selected agents to project's agents/ folder (one JSON per agent).")
        self._btn_export_agents.clicked.connect(self._on_export_to_agents)
        self._btn_export_hof = QtWidgets.QPushButton("Export to Hall of Fame")
        self._btn_export_hof.setToolTip("Write selected agents to project's agents/Hall of Fame/ folder.")
        self._btn_export_hof.clicked.connect(self._on_export_to_hof)
        btn_row.addWidget(self._btn_export_agents)
        btn_row.addWidget(self._btn_export_hof)
        btn_row.addStretch()
        form_layout.addRow(btn_row)

        layout.addWidget(form)
        self._form_widget = form
        self._form_widget.setEnabled(False)
        self._form_widget.setVisible(False)
        self._placeholder.setVisible(True)
        self._btn_export_agents.setEnabled(False)
        self._btn_export_hof.setEnabled(False)

    def set_run_output(
        self,
        project_path: Optional[str],
        population: Optional[Any],
        generation_index: int,
        fitness_config: Optional[Any] = None,
    ) -> None:
        """Set the last run output (from MainWindow after a generation or at run end)."""
        self._project_path = project_path
        self._population = population
        self._generation_index = generation_index
        self._fitness_config = fitness_config
        has_output = bool(
            population is not None and population.agents and project_path
        )
        self._placeholder.setVisible(not has_output)
        self._form_widget.setVisible(has_output)
        self._form_widget.setEnabled(has_output)
        self._btn_export_agents.setEnabled(has_output)
        self._btn_export_hof.setEnabled(has_output)
        if has_output:
            n = len(population.agents)
            self._spin_top_n.setMaximum(max(n, 1))
            self._spin_top_n.setValue(min(self._spin_top_n.value(), n))
            self._spin_from.setMaximum(max(n - 1, 0))
            self._spin_to.setMaximum(max(n - 1, 0))
            self._spin_to.setValue(min(n - 1, self._spin_to.value()))

    def clear_run_output(self) -> None:
        """Clear run output (e.g. when a new run starts)."""
        self.set_run_output(None, None, 0, None)

    def _on_range_mode_changed(self) -> None:
        idx = self._combo_range.currentIndex()
        self._spin_top_n.setVisible(idx == 1)
        self._spin_from.setVisible(idx == 2)
        self._label_to.setVisible(idx == 2)
        self._spin_to.setVisible(idx == 2)

    def _get_selected_agents(self) -> Optional[list]:
        if not self._population or not self._population.agents:
            return None
        agents = list(self._population.agents.values())
        order_by = self._combo_order.currentText()
        reverse = True
        sorted_agents = _sort_agents_for_export(
            agents, order_by, reverse, self._fitness_config
        )
        idx = self._combo_range.currentIndex()
        if idx == 0:
            return sorted_agents
        if idx == 1:
            n = self._spin_top_n.value()
            return sorted_agents[:n]
        from_idx = self._spin_from.value()
        to_idx = self._spin_to.value()
        if from_idx > to_idx:
            from_idx, to_idx = to_idx, from_idx
        return sorted_agents[from_idx : to_idx + 1]

    def _on_export_to_agents(self) -> None:
        agents = self._get_selected_agents()
        if not agents or not self._project_path:
            return
        from pathlib import Path
        from tarot.persistence import save_agents_to_directory
        dir_path = Path(self._project_path) / "agents"
        try:
            save_agents_to_directory(agents, dir_path)
            QtWidgets.QMessageBox.information(
                self,
                "Export",
                f"Exported {len(agents)} agent(s) to\n{dir_path}",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))

    def _on_export_to_hof(self) -> None:
        agents = self._get_selected_agents()
        if not agents or not self._project_path:
            return
        from pathlib import Path
        from tarot.persistence import save_agents_to_directory
        dir_path = Path(self._project_path) / "agents" / "Hall of Fame"
        try:
            save_agents_to_directory(agents, dir_path)
            QtWidgets.QMessageBox.information(
                self,
                "Export to Hall of Fame",
                f"Exported {len(agents)} agent(s) to\n{dir_path}",
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export to Hall of Fame failed", str(e))


# --- Charts area (ELO evolution from current + loaded logs) ---


class ChartsAreaWidget(QtWidgets.QGroupBox):
    """
    ELO evolution chart from current run and/or loaded logs.
    Banner "Viewing saved run: ..." when showing loaded data; generation slider for loaded log.
    """
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Charts", parent)
        self._current_entries: List[Dict[str, Any]] = []
        self._loaded_logs: List[tuple[str, str, List[Dict[str, Any]]]] = []  # (id, path, entries)
        layout = QtWidgets.QVBoxLayout(self)
        self._banner = QtWidgets.QLabel("")
        self._banner.setStyleSheet("font-weight: bold; color: #888;")
        layout.addWidget(self._banner)
        self._checkboxes_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(self._checkboxes_layout)
        slider_row = QtWidgets.QHBoxLayout()
        slider_row.addWidget(QtWidgets.QLabel("Generation:"))
        self._gen_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._gen_slider.setMinimum(0)
        self._gen_slider.setMaximum(0)
        self._gen_slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self._gen_slider)
        self._gen_label = QtWidgets.QLabel("0")
        slider_row.addWidget(self._gen_label)
        layout.addLayout(slider_row)
        self._chart = _ELOChartWidget(self)
        self._chart.setMinimumHeight(330)
        self._chart.setToolTip("ELO min/mean/max over generations. Hover for tooltips.")
        layout.addWidget(self._chart)

    def set_total_generations(self, total: Optional[int]) -> None:
        """
        Fix the X axis [0, total-1] for the current run.
        Pass None or <=0 to let the chart autoscale to data.
        """
        if total is None or total <= 0:
            self._chart.set_generation_bounds(None, None)
        else:
            self._chart.set_generation_bounds(0, total - 1)

    def set_current_entries(self, entries: List[Dict[str, Any]]) -> None:
        self._current_entries = list(entries) if entries else []
        self._rebuild_chart()

    def set_loaded_logs(self, loaded: List[tuple[str, str, List[Dict[str, Any]]]]) -> None:
        """List of (log_id, path, entries)."""
        self._loaded_logs = list(loaded)
        self._refresh_checkboxes()
        self._rebuild_chart()

    def _refresh_checkboxes(self) -> None:
        while self._checkboxes_layout.count():
            child = self._checkboxes_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        for log_id, path, entries in self._loaded_logs:
            cb = QtWidgets.QCheckBox(Path(path).name)
            cb.setProperty("log_id", log_id)
            cb.setChecked(False)
            cb.toggled.connect(self._rebuild_chart)
            self._checkboxes_layout.addWidget(cb)
        self._checkboxes_layout.addStretch()

    def _on_slider_changed(self, value: int) -> None:
        self._gen_label.setText(str(value))
        self._rebuild_chart()

    def _rebuild_chart(self) -> None:
        combined: List[Dict[str, Any]] = []
        combined.extend(self._current_entries)
        for i, (_log_id, _path, entries) in enumerate(self._loaded_logs):
            if i >= self._checkboxes_layout.count():
                break
            w = self._checkboxes_layout.itemAt(i).widget()
            if isinstance(w, QtWidgets.QCheckBox) and w.isChecked():
                combined.extend(entries)
        combined.sort(key=lambda e: e.get("generation_index", 0))
        gen_max = max((e.get("generation_index", 0) for e in combined), default=0)
        self._gen_slider.setMaximum(max(0, gen_max))
        self._gen_slider.setValue(gen_max)
        cap = self._gen_slider.value()
        filtered = [e for e in combined if e.get("generation_index", 0) <= cap] if cap >= 0 else combined
        any_loaded_checked = any(
            self._checkboxes_layout.itemAt(i).widget().isChecked()
            for i in range(min(len(self._loaded_logs), self._checkboxes_layout.count() - 1))
            if isinstance(self._checkboxes_layout.itemAt(i).widget(), QtWidgets.QCheckBox)
        )
        if self._loaded_logs and any_loaded_checked:
            self._banner.setText("Viewing saved run(s) + current")
        else:
            self._banner.setText("")
        self._chart.set_entries(filtered)
        self._chart.update()
