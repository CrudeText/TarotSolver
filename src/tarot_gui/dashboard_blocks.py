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
        layout = QtWidgets.QVBoxLayout(self)
        self._label_time_used = QtWidgets.QLabel("Time used: —")
        self._label_eta = QtWidgets.QLabel("Time left (ETA): —")
        self._label_avg = QtWidgets.QLabel("Avg time/gen: —")
        layout.addWidget(self._label_time_used)
        layout.addWidget(self._label_eta)
        layout.addWidget(self._label_avg)

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
    ELO block: summary (min, mean, max, std) + time series chart (ELO over generations).
    Data from RunLogManager current entries or selected loaded log.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("ELO", parent)
        self._entries: List[Dict[str, Any]] = []
        layout = QtWidgets.QVBoxLayout(self)
        self._summary_label = QtWidgets.QLabel("min: —  mean: —  max: —  std: —")
        layout.addWidget(self._summary_label)
        self._chart = _ELOChartWidget(self)
        self._chart.setMinimumHeight(180)
        layout.addWidget(self._chart)

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
        self._chart.set_entries(self._entries)
        self._chart.update()


class _ELOChartWidget(QtWidgets.QWidget):
    """Time series: elo_min, elo_mean, elo_max over generation_index (same style as League tab charts)."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._entries: List[Dict[str, Any]] = []
        self.setMinimumWidth(260)

    def set_entries(self, entries: List[Dict[str, Any]]) -> None:
        self._entries = list(entries) if entries else []

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

            if not self._entries:
                painter.setPen(QtGui.QPen(pen_color, 1))
                painter.setFont(QtGui.QFont(self.font().family(), 9))
                painter.drawText(
                    int(gx), int(gy), int(gw), int(gh),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    "No data",
                )
                return

            gens = [e.get("generation_index", 0) for e in self._entries]
            elo_mins = [e.get("elo_min", 0.0) for e in self._entries]
            elo_means = [e.get("elo_mean", 0.0) for e in self._entries]
            elo_maxs = [e.get("elo_max", 0.0) for e in self._entries]
            gen_min, gen_max = min(gens), max(gens) if gens else 0
            gen_range = (gen_max - gen_min) or 1
            all_elos = elo_mins + elo_means + elo_maxs
            elo_lo = min(all_elos) if all_elos else 0.0
            elo_hi = max(all_elos) if all_elos else 1500.0
            elo_range = (elo_hi - elo_lo) or 1.0

            # Axes
            painter.setPen(QtGui.QPen(pen_color, 1))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawRect(int(gx), int(gy), int(gw), int(gh))
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            painter.drawText(int(gx + gw // 2 - 25), int(h - 4), "Generation")
            painter.save()
            painter.translate(8, gy + gh // 2 - 10)
            painter.rotate(-90)
            painter.drawText(-20, 0, "ELO")
            painter.restore()

            # X ticks
            for i in range(5):
                t = i / 4.0
                g = gen_min + t * gen_range
                fx = gx + t * (gw - 1)
                painter.drawLine(int(fx), int(gy + gh), int(fx), int(gy + gh + 4))
                painter.drawText(int(fx - 12), int(gy + gh + 12), f"{int(g)}")
            # Y ticks
            for i in range(5):
                t = i / 4.0
                elo = elo_lo + t * elo_range
                fy = gy + gh - t * (gh - 1)
                painter.drawLine(int(gx - 4), int(fy), int(gx), int(fy))
                painter.drawText(int(gx - 30), int(fy + 4), f"{int(elo)}")

            def path_for(values: List[float]) -> QtGui.QPainterPath:
                path = QtGui.QPainterPath()
                for i, val in enumerate(values):
                    t = (gens[i] - gen_min) / gen_range if gen_range else 0
                    norm = (val - elo_lo) / elo_range if elo_range else 0
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
    """Per-agent ELO delta (current gen vs previous gen). Keys = agent id."""
    if len(entries) < 2:
        return {}
    prev_agents = {a.get("id"): float(a.get("elo_global", 0)) for a in entries[-2].get("agents", [])}
    cur_agents = entries[-1].get("agents", [])
    return {
        a.get("id", ""): float(a.get("elo_global", 0)) - prev_agents.get(a.get("id"), 0)
        for a in cur_agents
    }


class RLPerformanceBlockWidget(QtWidgets.QGroupBox):
    """
    Top-N table: name, ELO, Δ ELO, W/L deal, W/L match, avg score, generation.
    N configurable (default 10). Data from run log entries (latest generation).
    """
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("RL performance", parent)
        self._entries: List[Dict[str, Any]] = []
        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Top"))
        self._spin_n = QtWidgets.QSpinBox()
        self._spin_n.setRange(1, 50)
        self._spin_n.setValue(10)
        self._spin_n.valueChanged.connect(self._refresh_table)
        row.addWidget(self._spin_n)
        row.addWidget(QtWidgets.QLabel("agents"))
        row.addStretch()
        layout.addLayout(row)
        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels(["Name", "ELO", "Δ", "W/L deal", "W/L match", "Avg", "Gen"])
        self._table.horizontalHeader().setStretchLastSection(True)
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
        sorted_agents = sorted(agents, key=lambda a: float(a.get("elo_global", 0)), reverse=True)[:n]
        deltas = _delta_elo_from_entries(self._entries)
        gen = self._entries[-1].get("generation_index", 0)
        for a in sorted_agents:
            aid = a.get("id", "")
            deals_p, deals_w = int(a.get("deals_played", 0)), int(a.get("deals_won", 0))
            match_p, match_w = int(a.get("matches_played", 0)), int(a.get("matches_won", 0))
            total_s = float(a.get("total_match_score", 0))
            wl_d = f"{deals_w}/{deals_p}" if deals_p else "—"
            wl_m = f"{match_w}/{match_p}" if match_p else "—"
            avg = f"{total_s / match_p:.1f}" if match_p else "—"
            delta = deltas.get(aid, 0)
            delta_str = f"+{delta:.0f}" if delta > 0 else (f"{delta:.0f}" if delta < 0 else "—")
            row_idx = self._table.rowCount()
            self._table.insertRow(row_idx)
            self._table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(str(a.get("name", aid))))
            self._table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(f"{float(a.get('elo_global', 0)):.0f}"))
            self._table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(delta_str))
            self._table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(wl_d))
            self._table.setItem(row_idx, 4, QtWidgets.QTableWidgetItem(wl_m))
            self._table.setItem(row_idx, 5, QtWidgets.QTableWidgetItem(avg))
            self._table.setItem(row_idx, 6, QtWidgets.QTableWidgetItem(str(gen)))


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


# --- Export block (placeholder) ---


class ExportBlockWidget(QtWidgets.QGroupBox):
    """Placeholder until League Parameters export/HOF is wired. Tooltip: Configure in League Parameters."""
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Export", parent)
        layout = QtWidgets.QVBoxLayout(self)
        label = QtWidgets.QLabel("—")
        label.setToolTip("Configure in League Parameters")
        layout.addWidget(label)


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
        self._chart.setMinimumHeight(220)
        self._chart.setToolTip("ELO min/mean/max over generations. Hover for tooltips.")
        layout.addWidget(self._chart)

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
