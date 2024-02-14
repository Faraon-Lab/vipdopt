"""Entrypoint running the GUI. Mainly for testing right now."""
from __future__ import annotations
import logging
import sys

from PySide6.QtCore import Qt 

from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QListWidget,
    QAbstractItemView,
    QTreeWidget,
    QTreeWidgetItem,
    QDialog,
    QWidget,
)

from vipdopt.gui.config_editor import ConfigModel
from vipdopt.gui.ui_settings import Ui_MainWindow as Ui_SettingsWindow
from vipdopt.gui.ui_status import Ui_MainWindow as Ui_StatusWindow
from vipdopt.gui.ui_fom_dialog import Ui_Dialog as Ui_FomDialog
from vipdopt.project import Project
from vipdopt.optimization import FoM
from vipdopt.simulation import LumericalSimulation
from vipdopt.monitor import Monitor
from vipdopt.utils import PathLike, read_config_file, subclasses


class FomDialog(QDialog, Ui_FomDialog):
    def __init__(self, parent: QWidget | None, f: Qt.WindowType = Qt.WindowType.Dialog) -> None:
        super().__init__(parent, f)

        self.setupUi(self)

    def populate_tree(self, sims: list[LumericalSimulation]):
        self.treeWidget.clear()
        sim: LumericalSimulation
        for i, sim in enumerate(sims):
            sim_node = QTreeWidgetItem((f'Simulation {i}', ))
            source_nodes = [QTreeWidgetItem((src,)) for src in sim.monitor_names()]
            self.uncheck_items(*source_nodes)
            sim_node.addChildren(source_nodes)
            self.treeWidget.addTopLevelItem(sim_node)
    
    def check_items(self, *items: QTreeWidgetItem):
        for item in items:
            item.setCheckState(1, Qt.CheckState.Checked)

    def uncheck_items(self, *items: QTreeWidgetItem):
        for item in items:
            item.setCheckState(1, Qt.CheckState.Unchecked)
            item.setExpanded(False)

    def check_monitors(self, src_to_sim_map: dict[str, LumericalSimulation], *monitors: Monitor):
        all_children:list[QTreeWidgetItem] = self.treeWidget.findChildren(QTreeWidgetItem)
        self.uncheck_items(*all_children)
        mon_items = []
        for monitor in monitors:
            mon_nodes = self.treeWidget.findItems(monitor.monitor_name, Qt.MatchFlag.MatchRecursive | Qt.MatchFlag.MatchExactly, 0)
            mon_node = mon_nodes[list(src_to_sim_map.keys()).index(monitor.source_name)]
            mon_node.parent().setExpanded(True)

            mon_items.append(mon_node)
        self.check_items(*mon_items)


def _set_list_size(l: QListWidget):
    l.setMinimumWidth(l.sizeHintForColumn(0)) 


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
        self.device_model = ConfigModel()
        self.device_treeView.setModel(self.device_model)


        self.actionOpen.triggered.connect(self.open_project)

        self.config_pushButton.clicked.connect(self.load_yaml)
        self.sim_pushButton.clicked.connect(self.load_json)

        self.fom_fom_pushButton_0.clicked.connect(lambda: self.fom_dialog(0, 'fom'))
        self.fom_grad_pushButton_0.clicked.connect(lambda: self.fom_dialog(0, 'grad'))

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
        if proj_dir:
            self.project.load_project(proj_dir)
            print(f'Loaded project from {proj_dir}')
            self._update_values()
            print(f'Updated GUI with values from {proj_dir}')
            print(self.config_model)
    
    def fom_dialog(self, idx: int, mode: str='fom'):
        dialog = FomDialog(self)
        dialog.populate_tree(self.project.optimization.sims)
        fom = self.project.foms[idx]
        match mode.lower():
            case 'fom':
                monitors = fom.fom_monitors
            case 'grad':
                monitors = fom.grad_monitors
            case _:
                raise ValueError(f'fom_dialog can only be called with mode "fom" or "grad"')

        dialog.check_monitors(self.project.src_to_sim_map, *monitors)
        dialog.exec()
    
    def _update_values(self):
        """Update GUI to match values in Project."""
        # Config Tab
        self.config_model.load(self.project.config)

        # Simulation Tab
        self.sim_model.load(self.project.base_sim.as_dict())

        self.sim_config_treeWidget.clear()
        sim: LumericalSimulation
        for i, sim in enumerate(self.project.optimization.sims):
            sim_node = QTreeWidgetItem((f'Simulation {i}', ))
            source_nodes = [QTreeWidgetItem((src,)) for src in sim.source_names()]
            for src in source_nodes:
                src.setCheckState(1, Qt.CheckState.Unchecked)
            source_nodes[i].setCheckState(1, Qt.CheckState.Checked)
            sim_node.addChildren(source_nodes)
            self.sim_config_treeWidget.addTopLevelItem(sim_node)

        # FoM Tab
        fom_types = subclasses(FoM)
        for i, fom in enumerate(self.project.foms):
            self.fom_name_lineEdit_0.setText(fom.name)
            self.fom_type_comboBox_0.clear()
            self.fom_type_comboBox_0.addItems(fom_types)
            self.fom_type_comboBox_0.setCurrentIndex(fom_types.index(type(fom).__name__))

            # self.fom_fom_mon_listWidget_0.clear()
            # self.fom_fom_mon_listWidget_0.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
            # self.fom_fom_mon_listWidget_0.addItems(self.project.base_sim.monitor_names())
            # for mname in [mon.monitor_name for mon in fom.fom_monitors]:
            #     for item in self.fom_fom_mon_listWidget_0.findItems(mname, Qt.MatchFlag.MatchExactly):
            #         item.setSelected(True)

            # self.fom_grad_mon_listWidget_0.clear()
            # self.fom_grad_mon_listWidget_0.addItems(self.project.base_sim.monitor_names())
            # self.fom_grad_mon_listWidget_0.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
            # for mname in [mon.monitor_name for mon in fom.grad_monitors]:
            #     for item in self.fom_grad_mon_listWidget_0.findItems(mname, Qt.MatchFlag.MatchExactly):
            #         item.setSelected(True)
            self.fom_weight_lineEdit_0.setText(str(self.project.weights[i]))

        # Device Tab
        self.device_model.load(self.project.device.as_dict())

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
