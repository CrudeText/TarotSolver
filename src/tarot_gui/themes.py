"""
Theme support for the Tarot Solver GUI.

Provides dark (default) and light stylesheets, persisted via QSettings.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

SETTINGS_ORG = "TarotSolver"
SETTINGS_APP = "TarotSolver"
THEME_KEY = "theme"
PROJECTS_FOLDER_KEY = "projects_folder"
DEVICE_KEY = "default_device"

# Default projects folder (parent of typical project dir)
_DEFAULT_PROJECTS_FOLDER = "D:/1 - Project Data/TarotSolver"

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
    /* Small outer margin; reserve extra padding inside the top for the title */
    margin-top: 6px;
    padding-top: 22px;
    font-weight: bold;
}
QGroupBox::title {
    /* Place the title inside the box, top-left, instead of floating above */
    subcontrol-origin: padding;
    subcontrol-position: top left;
    left: 10px;
    top: 4px;
    padding: 0 4px 2px 4px;
    background-color: transparent;
    font-size: 14px;
}
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
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    background-color: #2d2d2d;
    color: #606060;
    border-color: #404040;
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
QLabel { color: #e0e0e0; background-color: transparent; }
QLabel:disabled { color: #606060; background-color: transparent; }
QCheckBox { color: #e0e0e0; background-color: transparent; }
QFormLayout label { color: #e0e0e0; background-color: transparent; }
"""

LIGHT_STYLESHEET = ""  # Use default Qt (light) palette


def get_saved_theme() -> str:
    settings = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
    return settings.value(THEME_KEY, DARK, type=str)


def save_theme(theme: str) -> None:
    settings = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
    settings.setValue(THEME_KEY, theme)


def get_projects_folder() -> str:
    settings = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
    return settings.value(PROJECTS_FOLDER_KEY, _DEFAULT_PROJECTS_FOLDER, type=str)


def save_projects_folder(path: str) -> None:
    settings = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
    settings.setValue(PROJECTS_FOLDER_KEY, path)


def get_saved_device() -> str:
    """Return saved default device string (e.g. 'cpu', 'cuda'). Uses 'cuda' if available and not set."""
    settings = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
    value = settings.value(DEVICE_KEY, None, type=str)
    if value:
        return value
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def save_device(device_str: str) -> None:
    settings = QtCore.QSettings(SETTINGS_ORG, SETTINGS_APP)
    settings.setValue(DEVICE_KEY, device_str or "cpu")


def apply_theme(app: QtWidgets.QApplication, theme: str) -> None:
    if theme == DARK:
        app.setStyleSheet(DARK_STYLESHEET)
    else:
        app.setStyleSheet(LIGHT_STYLESHEET)
