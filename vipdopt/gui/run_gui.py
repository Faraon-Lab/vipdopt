"""Entrypoint running the GUI. Mainly for testing right now."""
import logging
import sys

from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
)

from vipdopt.gui.config_editor import ConfigModel
from vipdopt.gui.ui_settings import Ui_MainWindow as Ui_SettingsWindow
from vipdopt.gui.ui_status import Ui_MainWindow as Ui_StatusWindow
from vipdopt.project import Project
from vipdopt.utils import PathLike, read_config_file


class SettingsWindow(QMainWindow, Ui_SettingsWindow):
    """Wrapper class for the settings window."""
    def __init__(self):
        """Initialize a SettingsWindow."""
        super().__init__()
        self.setupUi(self)


        self.config_model = ConfigModel()
        self.config_treeView.setModel(self.config_model)
        self.sim_model = ConfigModel()
        self.sim_treeView.setModel(self.sim_model)

        self.actionOpen.triggered.connect(self.open_project)

        self.config_pushButton.clicked.connect(self.load_yaml)
        self.sim_pushButton.clicked.connect(self.load_json)

        # Initialize Backend
        self.project = Project()

    # General Methods
    def new_project(self):
        """Create a new project."""
        self.project = Project()
        self._update_values()

    def open_project(self):
        """Load optimization project into the GUI."""
        proj_dir = QFileDialog.getExistingDirectory(
            self,
            'Select Project Directory',
            './',
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        self.project.load_project(proj_dir)
        self._update_values()
        logging.info(f'Loaded project from {proj_dir}')

    def _update_values(self):
        """Update GUI to match values in Project."""
        # Config Tab
        self.config_model.load(self.project.config)

        # Simulation Tab
        self.sim_model.load(self.project.base_sim.as_dict())

        # FoM Tab

        # Optimization Tab
        self.opt_iter_lineEdit.setText(str(self.project.optimization.iteration))
        self.opt_iter_per_epoch_lineEdit.setText(str(self.project.optimization.iter_per_epoch))
        self.opt_max_epoch_lineEdit.setText(str(self.project.optimization.max_epochs))


    # Configuration Tab
    def load_yaml(self):
        """Load a yaml config file into the configuration tab."""
        fname, _ = QFileDialog.getOpenFileName(
            self,
            'Select Configuration File',
            './',
            'YAML (*.yaml *.yml);;All Files(*.*)',
            'YAML (*.yaml *.yml)',
        )
        if fname:
            cfg = read_config_file(fname)
            self.config_lineEdit.setText(fname)
            self.config_model.load(cfg)

    # Simulation Tab
    def load_json(self):
        """Load a yaml config file into the configuration tab."""
        fname, _ = QFileDialog.getOpenFileName(
            self,
            'Select Configuration File',
            './',
            'JSON (*.json);;All Files(*.*)',
            'JSON (*.json)',
        )
        if fname:
            cfg = read_config_file(fname)
            self.sim_lineEdit.setText(fname)
            self.sim_model.load(cfg)

    # FoM Tab
    def add_fom_row(self):
        """Add a FoM to the FoM tab."""

    def remove_fom_row(self):
        """Remove the selected FoM from the FoM tab."""

    # Optimization Tab
    def load_optimization_settings(self, fname: PathLike):
        """Load settings for the optimization tab."""


class StatusDashboard(QMainWindow, Ui_StatusWindow):
    """Wrapper class for the status window."""
    def __init__(self):
        """Initialize a StatusWindow."""
        super().__init__()
        self.setupUi(self)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    app = QApplication(sys.argv)

    # Create application windows
    status_window = StatusDashboard()
    status_window.show()
    settings_window = SettingsWindow()
    settings_window.show()
    app.exec()
