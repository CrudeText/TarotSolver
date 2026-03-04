"""
Tarot Solver GUI entrypoint (placeholder implementation).

This creates a simple window with tabbed sections corresponding to the
planned GUI structure (Dashboard, League, Agents, Play, Settings).

All views are currently placeholders: layouts, buttons, and labels only,
with no real wiring to the backend yet.
"""
from __future__ import annotations

import time
from typing import Optional

from PySide6 import QtCore, QtWidgets
from pathlib import Path

from .dashboard_blocks import (
    ChartsAreaWidget,
    ELOBlockWidget,
    ExportBlockWidget,
    RLPerformanceBlockWidget,
)
from .league_tab import (
    LeagueRunWorker,
    LeagueTabState,
    LeagueTabWidget,
    RunSectionWidget,
    _ResizeFilter,
    make_league_tab,
)
from .project_dialog import OpenProjectDialog
from .run_log import RunLogManager, parse_run_log_entry, build_run_log_entry
from tarot.league import LeagueRunControl
from tarot.project import project_save
from .themes import (
    DARK,
    LIGHT,
    apply_theme,
    get_projects_folder,
    get_saved_device,
    get_saved_theme,
    save_device,
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
        self._run_log_manager = RunLogManager()
        # High-resolution ELO / RL time series for the current run (one entry per match).
        self._step_entries: list[dict] = []

        tabs = QtWidgets.QTabWidget()

        league_tab = make_league_tab()
        if DEFAULT_PROJECT_PATH:
            league_tab.open_project(DEFAULT_PROJECT_PATH)
        league_state = league_tab.state()

        dashboard_tab, run_section = self._make_dashboard_tab(league_state, self._run_log_manager)
        self._run_section = run_section
        run_section.run_log_loaded.connect(self._on_run_log_loaded)
        run_section.load_population_clicked.connect(self._on_load_population_clicked)
        self._league_tab = league_tab
        # Configure auto-save location based on initial project, if any.
        self._run_section.configure_auto_save_for_project(league_state.project_path)
        # If the project has an existing run log, show its ELO graph immediately.
        self._load_project_run_log_into_dashboard(league_state.project_path)
        tabs.addTab(dashboard_tab, "Dashboard")
        tabs.addTab(league_tab, "League Parameters")
        tabs.addTab(self._make_agents_tab(), "Agents")
        tabs.addTab(self._make_play_tab(), "Play")
        tabs.addTab(self._make_settings_tab(), "Settings")

        self.setCentralWidget(tabs)
        self._connect_run_controls()

    def _wrap_tab_in_scroll(self, content_widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        """Option C: wrap tab content in a scroll area; inner widget max width = viewport (no horizontal scroll)."""
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setWidget(content_widget)

        def on_viewport_resize() -> None:
            vw = max(scroll.viewport().width(), 400)
            content_widget.setMaximumWidth(vw)

        scroll.viewport().installEventFilter(_ResizeFilter(scroll, on_viewport_resize))
        on_viewport_resize()
        return scroll

    @staticmethod
    def _make_centered_label(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setAlignment(QtCore.Qt.AlignCenter)
        return label

    def _dashboard_placeholder(self, title: str, min_height: int) -> QtWidgets.QWidget:
        """One Dashboard block placeholder (QGroupBox with centered label)."""
        box = QtWidgets.QGroupBox(title)
        box.setMinimumHeight(min_height)
        layout = QtWidgets.QVBoxLayout(box)
        label = QtWidgets.QLabel(f"({title} — placeholder)")
        label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label)
        return box

    def _make_dashboard_tab(
        self,
        league_state: LeagueTabState,
        run_log_manager: RunLogManager,
    ) -> tuple[QtWidgets.QWidget, RunSectionWidget]:
        # Option C: fixed heights so tab fits in ~1000px viewport (with scrolling).
        # Increased a bit to give controls more vertical breathing room.
        # Order: Run (with File box), ELO/RL, Game, Export, Charts.
        DASH_RUN_BAR = 200  # Run controls + status + inline compute + File box
        DASH_ELO = 400  # summary + time series chart (taller for readability)
        DASH_RL = 225   # RL performance table (increased height)
        DASH_GAME = 110
        DASH_EXPORT = 90
        DASH_CHARTS = 400

        inner = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(inner)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # 1. Run bar (Start/Pause/Cancel, status, inline compute, log file name + File box)
        run_section = RunSectionWidget(league_state, run_log_manager=run_log_manager)
        run_section.setMinimumHeight(DASH_RUN_BAR)
        layout.addWidget(run_section)

        # 2. ELO block (observational: min/mean/max/std + time series chart)
        row_elo_rl = QtWidgets.QHBoxLayout()
        self._elo_block = ELOBlockWidget()
        self._elo_block.setMinimumHeight(DASH_ELO)
        row_elo_rl.addWidget(self._elo_block, 1)

        # 3. RL performance block (Top-N, W/L, high risers)
        self._rl_block = RLPerformanceBlockWidget()
        self._rl_block.setMinimumHeight(DASH_RL)
        row_elo_rl.addWidget(self._rl_block, 1)
        layout.addLayout(row_elo_rl)

        # 4. Charts area (ELO evolution, loaded logs checkboxes, banner, slider)
        # Charts box removed from the Dashboard layout; keep an instance for
        # internal wiring so code paths using _charts_area remain valid.
        self._charts_area = ChartsAreaWidget()

        scroll = self._wrap_tab_in_scroll(inner)
        return scroll, run_section

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
        self._run_log_manager.clear_current()
        self._step_entries = []
        self._run_section.clear_run_output()
        self._run_section.update_run_log_buttons()
        self._elo_block.set_entries([])
        self._rl_block.set_entries([])
        self._run_section.clear_compute_metrics()
        self._charts_area.set_current_entries([])
        self._run_start_time = time.time()
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
            device=get_saved_device(),
            parent=self,
        )
        self._league_worker = worker
        worker.generation_done.connect(self._on_league_generation_done)
        worker.match_done.connect(self._on_league_match_done)
        worker.finished_run.connect(self._on_league_finished)
        self._run_section.set_buttons_running(True)
        # Fix ELO chart X axis to the planned number of generations for this run.
        self._elo_block.set_total_generations(num_generations)
        self._charts_area.set_total_generations(num_generations)
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
        # Lightweight updates only: keep heavy disk I/O and full table refresh
        # out of the generation callback to keep the UI responsive.
        self._run_log_manager.append_generation(gen_idx, pop, summary_dict)
        # Do not overwrite league parameters population; user exports from Dashboard Export block.
        self._run_section.set_run_output(
            self._league_tab.state().project_path,
            pop,
            gen_idx,
            self._league_tab.get_league_config(),
        )
        self._run_section.update_metrics()
        self._run_section.update_run_log_buttons()
        # Update status line: Generation X of Y, Elapsed, ETA (only on generation done)
        total_gens = self._league_tab.get_num_generations()
        elapsed = time.time() - self._run_start_time
        completed = gen_idx + 1
        avg_time_per_gen = elapsed / completed if completed else 0.0
        remaining = total_gens - completed
        eta_sec = (remaining * avg_time_per_gen) if remaining > 0 else 0.0
        # First gens: show ETA as "calculating…"; after that show real ETA
        self._run_section.update_run_status(
            gen_idx,
            total_gens,
            elapsed,
            eta_sec if gen_idx >= 1 else None,
        )
        self._run_section.update_compute_metrics(
            elapsed,
            eta_sec if gen_idx >= 1 else None,
            avg_time_per_gen,
        )
        # For per-match resolution, prefer step entries if available.
        entries = self._step_entries or self._run_log_manager.get_current_entries()
        self._elo_block.set_entries(entries)
        self._rl_block.set_entries(entries)
        self._charts_area.set_current_entries(entries)
        self._charts_area.set_loaded_logs([
            (log.id, log.path, log.entries) for log in self._run_log_manager.get_loaded_logs()
        ])

    def _on_run_log_loaded(self, log_id: str) -> None:
        """Update charts area when user loads a run log."""
        self._charts_area.set_loaded_logs([
            (log.id, log.path, log.entries) for log in self._run_log_manager.get_loaded_logs()
        ])
        # Loaded logs use generation-level resolution.
        self._step_entries = []

    def _on_load_population_clicked(self) -> None:
        """Handle 'Load League Project' from the Dashboard File box."""
        base = get_projects_folder()
        if not base:
            QtWidgets.QMessageBox.warning(
                self,
                "Projects folder not set",
                "No Projects folder is configured.\n\n"
                "Go to the Settings tab, set a Projects folder, then try again.",
            )
            return
        base_path = Path(base)
        if not base_path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Projects folder unavailable",
                "The Projects folder does not exist:\n"
                f"{base}\n\n"
                "Please go to the Settings tab, choose a valid Projects folder,\n"
                "and try again.",
            )
            return
        dlg = OpenProjectDialog(str(base_path), self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        path = dlg.result_path()
        if not path:
            return
        if self._league_tab.open_project(path, show_message=True):
            # Refresh state + auto-save config
            state = self._league_tab.state()
            self._run_section.update_start_enabled()
            self._run_section.configure_auto_save_for_project(state.project_path)
            # If the loaded project has an existing run log, reflect it in the dashboard.
            self._load_project_run_log_into_dashboard(state.project_path)

    def _on_league_match_done(
        self,
        gen_idx: int,
        round_idx: int,
        population: object,
        round_summary: object,
    ) -> None:
        """
        Per-match callback from LeagueRunWorker.

        Build a transient run-log-style entry capturing the current Population snapshot
        and ELO summary after this match, append it to the in-memory step series,
        and update the Dashboard ELO / RL / Game metrics blocks and chart.
        """
        from tarot.tournament import Population as TarotPopulation

        if not isinstance(population, TarotPopulation):
            pop = TarotPopulation()
            if hasattr(population, "agents"):
                for a in population.agents.values():
                    pop.add(a)
        else:
            pop = population

        summary_dict = round_summary if isinstance(round_summary, dict) else {}
        entry = build_run_log_entry(gen_idx, pop, summary_dict)
        self._step_entries.append(entry)
        entries = self._step_entries
        self._elo_block.set_entries(entries)
        self._rl_block.set_entries(entries)
        self._charts_area.set_current_entries(entries)

    def _load_project_run_log_into_dashboard(self, project_path: Optional[str]) -> None:
        """
        When a project is loaded, try to load its auto-saved run log (if any)
        and immediately reflect it in the Dashboard ELO/RL/Game metrics blocks
        and Charts area.
        """
        if not project_path:
            return
        logs_dir = Path(project_path) / "logs"
        if not logs_dir.exists():
            return
        # Use the same default as League tab auto-save if no custom name is set.
        log_filename = ""
        if hasattr(self, "_run_section"):
            try:
                log_filename = self._run_section.get_log_filename().strip()
            except Exception:
                log_filename = ""
        if not log_filename:
            log_filename = "league_run.jsonl"
        log_path = logs_dir / log_filename
        if not log_path.exists():
            return
        entries = []
        try:
            with log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entries.append(parse_run_log_entry(line))
        except Exception:
            return
        if not entries:
            return
        # When loading from disk we have generation-level entries only.
        self._step_entries = list(entries)
        self._elo_block.set_entries(entries)
        self._rl_block.set_entries(entries)
        self._charts_area.set_current_entries(entries)

    def _on_league_finished(self, cancelled: bool, paused: bool) -> None:
        self._run_section.set_buttons_running(False)
        self._run_section.update_run_status(-1, 0, 0.0, None)
        self._run_section.clear_compute_metrics()
        self._league_worker = None
        self._league_control = None
        # Persist final state and refresh League tab visuals once at the end,
        # instead of on every generation, to avoid UI stalls during the run.
        state = self._league_tab.state()
        if state.project_path:
            try:
                project_save(
                    state.project_path,
                    groups=self._league_tab._groups_tuples(),
                    league_config=self._league_tab.get_league_config(),
                    generation_index=state.generation_index,
                    last_summary=state.last_summary,
                    league_ui=self._league_tab.get_league_ui(),
                    hof_agents=state.hof_agents,
                )
                self._league_tab._update_project_label()
                self._league_tab._refresh_table()
            except Exception:
                # Non-fatal: failures here should not crash the UI.
                pass
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
        inner = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(inner)

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
        return self._wrap_tab_in_scroll(inner)

    def _make_play_tab(self) -> QtWidgets.QWidget:
        inner = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(inner)

        layout.addWidget(self._make_centered_label("Play tab (custom tables, spectate, play-vs-AI)."))

        buttons_row = QtWidgets.QHBoxLayout()
        buttons_row.addWidget(QtWidgets.QPushButton("New Custom Table (placeholder)"))
        buttons_row.addWidget(QtWidgets.QPushButton("Play vs AI (placeholder)"))
        buttons_row.addWidget(QtWidgets.QPushButton("Open Replay (placeholder)"))
        buttons_row.addStretch(1)
        layout.addLayout(buttons_row)

        layout.addStretch(1)
        return self._wrap_tab_in_scroll(inner)

    def _make_settings_tab(self) -> QtWidgets.QWidget:
        inner = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(inner)

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

        from tarot.device import get_available_devices
        compute_group = QtWidgets.QGroupBox("Compute")
        compute_layout = QtWidgets.QFormLayout(compute_group)
        self._combo_device = QtWidgets.QComboBox()
        devices = get_available_devices()
        saved_device = get_saved_device()
        for i, (display_name, device_str) in enumerate(devices):
            self._combo_device.addItem(display_name, device_str)
            if device_str == saved_device:
                self._combo_device.setCurrentIndex(i)
        if self._combo_device.currentData() != saved_device:
            self._combo_device.setCurrentIndex(0)  # fallback to CPU
        self._combo_device.setToolTip("Device for league runs (policy inference and PPO fine-tuning). CUDA is used when available.")
        self._combo_device.currentIndexChanged.connect(self._on_device_changed)
        compute_layout.addRow("Default device:", self._combo_device)
        compute_layout.addRow("Max parallel jobs:", QtWidgets.QSpinBox())
        layout.addWidget(compute_group)

        layout.addStretch(1)
        return self._wrap_tab_in_scroll(inner)

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

    def _on_device_changed(self) -> None:
        if hasattr(self, "_combo_device") and self._combo_device.currentData():
            save_device(self._combo_device.currentData())


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

