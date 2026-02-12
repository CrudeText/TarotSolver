"""
Tarot Solver GUI entrypoint (placeholder implementation).

This creates a simple window with tabbed sections corresponding to the
planned GUI structure (Dashboard, League, Agents, Play, Settings).

All views are currently placeholders: layouts, buttons, and labels only,
with no real wiring to the backend yet.
"""
from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtWidgets

from .league_tab import make_league_tab
from .themes import DARK, LIGHT, apply_theme, get_saved_theme, save_theme


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Tarot Solver (Placeholder GUI)")
        self.resize(1200, 800)

        tabs = QtWidgets.QTabWidget()

        tabs.addTab(self._make_dashboard_tab(), "Dashboard")
        tabs.addTab(make_league_tab(), "League")
        tabs.addTab(self._make_agents_tab(), "Agents")
        tabs.addTab(self._make_play_tab(), "Play")
        tabs.addTab(self._make_settings_tab(), "Settings")

        self.setCentralWidget(tabs)

    @staticmethod
    def _make_centered_label(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignCenter)
        return label

    def _make_dashboard_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        layout.addWidget(self._make_centered_label("Dashboard overview (placeholder)."))

        buttons_row = QtWidgets.QHBoxLayout()
        buttons_row.addStretch(1)
        buttons_row.addWidget(QtWidgets.QPushButton("Start League Run (placeholder)"))
        buttons_row.addWidget(QtWidgets.QPushButton("Open Latest League Log (placeholder)"))
        buttons_row.addStretch(1)
        layout.addLayout(buttons_row)

        layout.addStretch(1)
        return widget

    def _make_agents_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        layout.addWidget(self._make_centered_label("Agents list and Hall of Fame (placeholder)."))

        list_group = QtWidgets.QGroupBox("Agents (placeholder list)")
        list_layout = QtWidgets.QVBoxLayout(list_group)
        list_widget = QtWidgets.QListWidget()
        list_widget.addItem("Example Agent A (placeholder)")
        list_widget.addItem("Example Agent B (placeholder)")
        list_layout.addWidget(list_widget)
        layout.addWidget(list_group)

        buttons_row = QtWidgets.QHBoxLayout()
        buttons_row.addWidget(QtWidgets.QPushButton("View Details (placeholder)"))
        buttons_row.addWidget(QtWidgets.QPushButton("Open Hall of Fame (placeholder)"))
        buttons_row.addStretch(1)
        layout.addLayout(buttons_row)

        layout.addStretch(1)
        return widget

    def _make_play_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        layout.addWidget(self._make_centered_label("Play tab (custom tables, spectate, play-vs-AI)."))

        buttons_row = QtWidgets.QHBoxLayout()
        buttons_row.addWidget(QtWidgets.QPushButton("New Custom Table (placeholder)"))
        buttons_row.addWidget(QtWidgets.QPushButton("Play vs AI (placeholder)"))
        buttons_row.addWidget(QtWidgets.QPushButton("Open Replay (placeholder)"))
        buttons_row.addStretch(1)
        layout.addLayout(buttons_row)

        layout.addStretch(1)
        return widget

    def _make_settings_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        form_group = QtWidgets.QGroupBox("Appearance")
        form_layout = QtWidgets.QFormLayout(form_group)
        theme_combo = QtWidgets.QComboBox()
        theme_combo.addItems(["Dark", "Light"])
        saved = get_saved_theme()
        theme_combo.setCurrentIndex(0 if saved == "dark" else 1)
        theme_combo.currentTextChanged.connect(self._on_theme_changed)
        form_layout.addRow("Theme:", theme_combo)
        layout.addWidget(form_group)

        other_group = QtWidgets.QGroupBox("Other (placeholder)")
        other_layout = QtWidgets.QFormLayout(other_group)
        other_layout.addRow("Default device:", QtWidgets.QComboBox())
        other_layout.addRow("Max parallel jobs:", QtWidgets.QSpinBox())
        layout.addWidget(other_group)

        layout.addStretch(1)
        return widget

    def _on_theme_changed(self, text: str) -> None:
        theme = DARK if text.lower() == "dark" else LIGHT
        save_theme(theme)
        app = QtWidgets.QApplication.instance()
        if app:
            apply_theme(app, theme)


def main(argv: Optional[list[str]] = None) -> None:
    import sys

    app = QtWidgets.QApplication(argv or sys.argv)
    theme = get_saved_theme()
    apply_theme(app, theme)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

