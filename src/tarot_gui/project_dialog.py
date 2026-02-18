"""
New Project / Open Project dialog.

Shows projects folder, list of existing projects, and name input for creating new projects.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtWidgets

from tarot.project import PROJECT_JSON


def _list_existing_projects(base_dir: Path) -> list[str]:
    """Return sorted list of project folder names (subdirs containing project.json)."""
    if not base_dir.exists() or not base_dir.is_dir():
        return []
    result = []
    for child in base_dir.iterdir():
        if child.is_dir() and (child / PROJECT_JSON).exists():
            result.append(child.name)
    return sorted(result)


def _is_valid_project_name(name: str) -> bool:
    """Check that name is non-empty and valid for filesystem."""
    if not name or not name.strip():
        return False
    # Disallow chars that are problematic on Windows/Unix
    invalid = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
    return not invalid.search(name.strip())


class NewProjectDialog(QtWidgets.QDialog):
    """
    Dialog for creating a new project or opening an existing one.

    Shows: projects folder (read-only), list of existing projects,
    name input for create, Create and Open buttons.
    """

    def __init__(
        self,
        projects_folder: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._projects_folder = Path(projects_folder)
        self._result_path: Optional[str] = None  # Set on Create or Open
        self.setWindowTitle("New / Open Project")
        self.setMinimumSize(450, 380)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # Projects folder (read-only)
        folder_group = QtWidgets.QGroupBox("Projects folder")
        folder_layout = QtWidgets.QVBoxLayout(folder_group)
        self._label_folder = QtWidgets.QLabel(str(self._projects_folder))
        self._label_folder.setWordWrap(True)
        self._label_folder.setStyleSheet("color: #888; font-size: 11px;")
        folder_layout.addWidget(self._label_folder)
        layout.addWidget(folder_group)

        # Existing projects list
        list_group = QtWidgets.QGroupBox("Existing projects")
        list_layout = QtWidgets.QVBoxLayout(list_group)
        self._list = QtWidgets.QListWidget()
        self._list.setMinimumHeight(120)
        self._list.itemDoubleClicked.connect(self._on_open_selected)
        self._refresh_list()
        list_layout.addWidget(self._list)
        layout.addWidget(list_group)

        # Create new
        create_group = QtWidgets.QGroupBox("Create new project")
        create_layout = QtWidgets.QFormLayout(create_group)
        self._edit_name = QtWidgets.QLineEdit()
        self._edit_name.setPlaceholderText("Enter project name...")
        self._edit_name.textChanged.connect(self._on_name_changed)
        create_layout.addRow("Project name:", self._edit_name)
        layout.addWidget(create_group)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch(1)
        self._btn_open = QtWidgets.QPushButton("Open selected")
        self._btn_open.clicked.connect(self._on_open_selected)
        self._btn_open.setEnabled(False)
        self._list.itemSelectionChanged.connect(
            lambda: self._btn_open.setEnabled(bool(self._list.currentItem()))
        )
        btn_layout.addWidget(self._btn_open)
        self._btn_create = QtWidgets.QPushButton("Create")
        self._btn_create.clicked.connect(self._on_create)
        self._btn_create.setEnabled(False)
        btn_layout.addWidget(self._btn_create)
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

    def _refresh_list(self) -> None:
        self._list.clear()
        projects = _list_existing_projects(self._projects_folder)
        for name in projects:
            self._list.addItem(name)
        if not projects and self._projects_folder.exists():
            self._list.addItem("(no projects yet)")
            self._list.item(0).setFlags(QtCore.Qt.ItemFlag.NoItemFlags)

    def _on_name_changed(self) -> None:
        name = self._edit_name.text().strip()
        if not name:
            self._btn_create.setEnabled(False)
            self._edit_name.setToolTip("")
            return
        if not _is_valid_project_name(name):
            self._btn_create.setEnabled(False)
            self._edit_name.setToolTip("Invalid characters. Avoid: < > : \" / \\ | ? *")
            return
        exists = (self._projects_folder / name).exists()
        self._btn_create.setEnabled(not exists)
        self._edit_name.setToolTip(
            "Project name already exists. Select it above to Open."
            if exists
            else ""
        )

    def _on_create(self) -> None:
        name = self._edit_name.text().strip()
        if not name or not _is_valid_project_name(name):
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid name",
                "Please enter a valid project name.",
            )
            return
        path = self._projects_folder / name
        if path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Name exists",
                f"Project '{name}' already exists. Select it above to open.",
            )
            return
        try:
            self._projects_folder.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Cannot create folder",
                f"The projects folder could not be created or used:\n\n{self._projects_folder}\n\n{e}\n\nChoose a different folder in Settings, or create the folder manually.",
            )
            return
        self._result_path = str(path)
        self.accept()

    def _on_open_selected(self) -> None:
        item = self._list.currentItem()
        if not item or not (item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled):
            return
        name = item.text()
        if name == "(no projects yet)":
            return
        path = self._projects_folder / name
        if path.exists() and (path / PROJECT_JSON).exists():
            self._result_path = str(path)
            self.accept()

    def result_path(self) -> Optional[str]:
        """Return the chosen project path, or None if cancelled."""
        return self._result_path
