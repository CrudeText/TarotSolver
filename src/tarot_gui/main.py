"""
Tarot Solver GUI entrypoint (placeholder implementation).

This creates a simple window with tabbed sections corresponding to the
planned GUI structure (Dashboard, League, Agents, Play, Settings).

All views are currently placeholders: layouts, buttons, and labels only,
with no real wiring to the backend yet.
"""
from __future__ import annotations

from typing import Optional

from PySide6 import QtWidgets, QtCore


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Tarot Solver (Placeholder GUI)")
        self.resize(1200, 800)

        tabs = QtWidgets.QTabWidget()

        tabs.addTab(self._make_dashboard_tab(), "Dashboard")
        tabs.addTab(self._make_league_tab(), "League")
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

    def _make_league_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        layout.addWidget(self._make_centered_label("League configuration and status (placeholder)."))

        form_group = QtWidgets.QGroupBox("League config (no-op)")
        form_layout = QtWidgets.QFormLayout(form_group)
        form_layout.addRow("Generations:", QtWidgets.QSpinBox())
        form_layout.addRow("Population size:", QtWidgets.QSpinBox())
        form_layout.addRow("Rounds / generation:", QtWidgets.QSpinBox())
        form_layout.addRow("Deals / match:", QtWidgets.QSpinBox())
        layout.addWidget(form_group)

        buttons_row = QtWidgets.QHBoxLayout()
        buttons_row.addStretch(1)
        buttons_row.addWidget(QtWidgets.QPushButton("Run One Generation (placeholder)"))
        buttons_row.addWidget(QtWidgets.QPushButton("Run Full League (placeholder)"))
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

        layout.addWidget(self._make_centered_label("Global settings (placeholder)."))

        form_group = QtWidgets.QGroupBox("Settings (no-op)")
        form_layout = QtWidgets.QFormLayout(form_group)
        form_layout.addRow("Default device:", QtWidgets.QComboBox())
        form_layout.addRow("Max parallel jobs:", QtWidgets.QSpinBox())
        layout.addWidget(form_group)

        layout.addStretch(1)
        return widget


def main(argv: Optional[list[str]] = None) -> None:
    import sys

    app = QtWidgets.QApplication(argv or sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

