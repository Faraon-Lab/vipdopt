"""Entrypoint running the GUI. Mainly for testing right now."""
import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
)

from vipdopt.gui.ui_settings import Ui_MainWindow as Ui_SettingsWindow
from vipdopt.utils import read_config_file, PathLike
from vipdopt.gui.config_editor import ConfigModel


class SettingsWindow(QMainWindow, Ui_SettingsWindow):
    """Wrapper class for the settings window."""
    def __init__(self):
        """Initialize a settigns window."""
        super().__init__()
        self.setupUi(self)

        self.config_model = ConfigModel()
        self.config_treeView.setModel(self.config_model)
        self.sim_model = ConfigModel()
        self.sim_treeView.setModel(self.sim_model)

        self.actionOpen.triggered.connect(self.open_project)

        self.config_pushButton.clicked.connect(self.load_yaml)
        self.sim_pushButton.clicked.connect(self.load_json)

    # General Methods
    def open_project(self):
        """Load optimization project into the GUI."""
        proj_dir = QFileDialog.getExistingDirectory(
            self,
            'Select Project Directory',
            './',
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        logging.info(f'Loaded project from {proj_dir}')

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
        pass

    def remove_fom_row(self):
        pass

    # Optimization Tab
    def load_optimization_settings(self, fname: PathLike):
        pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    app = QApplication(sys.argv)

    window = SettingsWindow()
    window.show()
    app.exec()
