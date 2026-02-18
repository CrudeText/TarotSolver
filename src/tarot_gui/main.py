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

from .league_tab import (
    LeagueRunWorker,
    LeagueTabState,
    LeagueTabWidget,
    RunSectionWidget,
    make_league_tab,
)
from tarot.league import LeagueRunControl
from tarot.project import project_save
from .themes import (
    DARK,
    LIGHT,
    apply_theme,
    get_projects_folder,
    get_saved_theme,
    save_projects_folder,
    save_theme,
)


# Default project to open on startup. Set to a path to auto-open, or None to start with no project.
DEFAULT_PROJECT_PATH = "D:/A - Project Data/TarotSolver/Test 1"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Tarot Solver")
        # Option 4: target 1080p; fullscreen windowed (maximized)
        self.setMinimumSize(1920, 1080)
        self.resize(1920, 1080)

        self._league_worker: Optional[LeagueRunWorker] = None
        self._league_control: Optional[LeagueRunControl] = None

        tabs = QtWidgets.QTabWidget()

        league_tab = make_league_tab()
        if DEFAULT_PROJECT_PATH:
            league_tab.open_project(DEFAULT_PROJECT_PATH)
        league_state = league_tab.state()

        dashboard_tab, run_section = self._make_dashboard_tab(league_state)
        self._run_section = run_section
        self._league_tab = league_tab
        tabs.addTab(dashboard_tab, "Dashboard")
        tabs.addTab(league_tab, "League Parameters")
        tabs.addTab(self._make_agents_tab(), "Agents")
        tabs.addTab(self._make_play_tab(), "Play")
        tabs.addTab(self._make_settings_tab(), "Settings")

        self.setCentralWidget(tabs)
        self._connect_run_controls()

    @staticmethod
    def _make_centered_label(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignCenter)
        return label

    def _make_dashboard_tab(self, league_state: LeagueTabState) -> tuple[QtWidgets.QWidget, RunSectionWidget]:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        layout.addWidget(self._make_centered_label("Dashboard overview (placeholder)."))

        run_section = RunSectionWidget(league_state)
        layout.addWidget(run_section)

        layout.addStretch(1)
        return widget, run_section

    def _connect_run_controls(self) -> None:
        self._run_section.start_clicked.connect(self._on_league_start)
        self._run_section.pause_clicked.connect(self._on_league_pause)
        self._run_section.cancel_clicked.connect(self._on_league_cancel)

    def _on_league_start(self) -> None:
        state = self._league_tab.state()
        if not state.project_path:
            return
        pop = state.build_population()
        if not pop.agents:
            QtWidgets.QMessageBox.warning(
                self,
                "League run",
                "Population is empty. Add at least one group of agents in the League Parameters tab.",
            )
            return
        cfg = self._league_tab.get_league_config()
        num_generations = self._league_tab.get_num_generations()
        control = LeagueRunControl()
        self._league_control = control
        worker = LeagueRunWorker(
            pop=pop,
            cfg=cfg,
            num_generations=num_generations,
            project_path=state.project_path,
            control=control,
            rng_seed=None,
            parent=self,
        )
        self._league_worker = worker
        worker.generation_done.connect(self._on_league_generation_done)
        worker.finished_run.connect(self._on_league_finished)
        self._run_section.set_buttons_running(True)
        worker.start()

    def _on_league_generation_done(self, gen_idx: int, population: object, summary: object) -> None:
        from tarot.tournament import Population as TarotPopulation
        if not isinstance(population, TarotPopulation):
            pop = TarotPopulation()
            if hasattr(population, "agents"):
                for a in population.agents.values():
                    pop.add(a)
        else:
            pop = population
        summary_dict = summary if isinstance(summary, dict) else {}
        self._league_tab.apply_population_from_run(pop, gen_idx, summary_dict)
        self._league_tab._refresh_table()
        self._league_tab._update_project_label()
        state = self._league_tab.state()
        project_save(
            state.project_path,
            groups=self._league_tab._groups_tuples(),
            league_config=self._league_tab.get_league_config(),
            generation_index=state.generation_index,
            last_summary=state.last_summary,
            league_ui=self._league_tab.get_league_ui(),
        )
        self._run_section.update_metrics()

    def _on_league_finished(self, cancelled: bool, paused: bool) -> None:
        self._run_section.set_buttons_running(False)
        self._league_worker = None
        self._league_control = None
        if cancelled:
            QtWidgets.QMessageBox.information(self, "League run", "Run cancelled.")
        elif paused:
            QtWidgets.QMessageBox.information(self, "League run", "Run paused at end of generation.")

    def _on_league_pause(self) -> None:
        if self._league_worker:
            self._league_worker.request_pause()

    def _on_league_cancel(self) -> None:
        if self._league_control:
            self._league_control.request_cancel()

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

        storage_group = QtWidgets.QGroupBox("Storage")
        storage_layout = QtWidgets.QFormLayout(storage_group)
        projects_row = QtWidgets.QHBoxLayout()
        self._edit_projects_folder = QtWidgets.QLineEdit()
        self._edit_projects_folder.setText(get_projects_folder())
        self._edit_projects_folder.setPlaceholderText("Path to projects directory")
        projects_row.addWidget(self._edit_projects_folder)
        btn_browse = QtWidgets.QPushButton("Browse...")
        btn_browse.clicked.connect(self._on_browse_projects_folder)
        projects_row.addWidget(btn_browse)
        storage_layout.addRow("Projects folder:", projects_row)
        layout.addWidget(storage_group)

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

    def _on_browse_projects_folder(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select projects folder", self._edit_projects_folder.text()
        )
        if path:
            self._edit_projects_folder.setText(path)
            save_projects_folder(path)


def main(argv: Optional[list[str]] = None) -> None:
    import sys

    app = QtWidgets.QApplication(argv or sys.argv)
    theme = get_saved_theme()
    apply_theme(app, theme)
    win = MainWindow()
    win.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

