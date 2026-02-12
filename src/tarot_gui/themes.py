"""
Theme support for the Tarot Solver GUI.

Provides dark (default) and light stylesheets, persisted via QSettings.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

SETTINGS_ORG = "TarotSolver"
SETTINGS_APP = "TarotSolver"
THEME_KEY = "theme"

DARK = "dark"
LIGHT = "light"
THEMES = (DARK, LIGHT)

DARK_STYLESHEET = """
QWidget { background-color: #2d2d2d; color: #e0e0e0; }
QMainWindow { background-color: #2d2d2d; }
QGroupBox {
    background-color: #363636;
    border: 1px solid #505050;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton {
    background-color: #404040;
    color: #e0e0e0;
    border: 1px solid #505050;
    border-radius: 4px;
    padding: 6px 12px;
}
QPushButton:hover { background-color: #505050; }
QPushButton:pressed { background-color: #606060; }
QPushButton:disabled { background-color: #353535; color: #808080; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #3d3d3d;
    color: #e0e0e0;
    border: 1px solid #505050;
    border-radius: 4px;
    padding: 4px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background-color: #3d3d3d; color: #e0e0e0; }
QTableWidget {
    background-color: #2d2d2d;
    color: #e0e0e0;
    gridline-color: #404040;
}
QTableWidget::item { padding: 4px; }
QHeaderView::section {
    background-color: #404040;
    color: #e0e0e0;
    padding: 6px;
    border: 1px solid #505050;
}
QScrollArea { background-color: #2d2d2d; }
QTabWidget::pane { background-color: #2d2d2d; border: 1px solid #505050; }
QTabBar::tab {
    background-color: #363636;
    color: #e0e0e0;
    padding: 8px 16px;
    margin-right: 2px;
}
QTabBar::tab:selected { background-color: #404040; }
QDialog { background-color: #2d2d2d; }
QLabel { color: #e0e0e0; }
QCheckBox { color: #e0e0e0; }
QFormLayout label { color: #e0e0e0; }
"""

LIGHT_STYLESHEET = ""  # Use default Qt (light) palette


def get_saved_theme() -> str:
    settings = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
    return settings.value(THEME_KEY, DARK, type=str)


def save_theme(theme: str) -> None:
    settings = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
    settings.setValue(THEME_KEY, theme)


def apply_theme(app: QtWidgets.QApplication, theme: str) -> None:
    if theme == DARK:
        app.setStyleSheet(DARK_STYLESHEET)
    else:
        app.setStyleSheet(LIGHT_STYLESHEET)
