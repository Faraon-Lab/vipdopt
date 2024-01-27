"""Entrypoint running the GUI. Mainly for testing right now."""
import logging
import sys

from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QTableWidgetItem,
)

from vipdopt.gui.ui_settings import Ui_MainWindow as Ui_SettingsWindow
from vipdopt.utils import read_config_file


class SettingsWindow(QMainWindow, Ui_SettingsWindow):
    """Wrapper class for the settings window."""
    def __init__(self):
        """Initialize a settigns window."""
        super().__init__()
        self.setupUi(self)

        self.actionOpen.triggered.connect(self.open_project)

        self.pushButton.clicked.connect(self.load_config)

    def open_project(self):
        """Load optimization project into the GUI."""
        proj_dir = QFileDialog.getExistingDirectory(
            self,
            'Select Project Directory',
            './',
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )
        logging.info(f'Loaded project from {proj_dir}')

    def load_config(self):
        """Load a config file into the configuration tab."""
        fname, _ = QFileDialog.getOpenFileName(
            self,
            'Select Configuration File',
            './',
            'YAML (*.yaml *.yml);;All Files(*.*)',
            'YAML (*.yaml *.yml)',
        )
        cfg = read_config_file(fname)
        for i, (key, val) in enumerate(cfg.items()):
            self.tableWidget.setItem(i, 0, QTableWidgetItem(key))
            self.tableWidget.setItem(i, 1, QTableWidgetItem(val))


    # def new_project(self):
    #         self,
    #         "Select Project Directory",
    #         "./",
    #         QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
    #     if self.settings_window is None:



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    app = QApplication(sys.argv)

    window = SettingsWindow()
    window.show()
    app.exec()
