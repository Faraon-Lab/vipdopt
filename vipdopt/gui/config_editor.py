"""Widget for editing configuration files. Based on the Qt JSON editor example."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any

from overrides import override
from PySide6.QtCore import QAbstractItemModel, QModelIndex, QObject, Qt
from PySide6.QtWidgets import QApplication, QHeaderView, QTreeView

from vipdopt.utils import read_config_file


class TreeItem:
    """A config item corresponding to a line in QTreeView."""

    def __init__(self, parent: TreeItem = None):
        """Initialize a TreeItem."""
        self._parent = parent
        self._key = ''
        self._value = ''
        self._value_type = None
        self._children = []

    def append_child(self, item: TreeItem):
        """Add item as a child."""
        self._children.append(item)

    def child(self, row: int) -> TreeItem:
        """Return the child of the current item from the given row."""
        return self._children[row]

    def parent(self) -> TreeItem:
        """Return the parent of the current item."""
        return self._parent

    def child_count(self) -> int:
        """Return the number of children of the current item."""
        return len(self._children)

    def row(self) -> int:
        """Return the row where the current item occupies in the parent."""
        return self._parent._children.index(self) if self._parent else 0  # noqa: SLF001

    @property
    def key(self) -> str:
        """Return the key name."""
        return self._key

    @key.setter
    def key(self, key: str):
        """Set key name of the current item."""
        self._key = key

    @property
    def value(self) -> str:
        """Return the value name of the current item."""
        return self._value

    @value.setter
    def value(self, value: str):
        """Set value name of the current item."""
        self._value = value

    @property
    def value_type(self):
        """Return the python type of the item's value."""
        return self._value_type

    @value_type.setter
    def value_type(self, value):
        """Set the python type of the item's value."""
        self._value_type = value

    @classmethod
    def load(
        cls, value: list | tuple | dict, parent: TreeItem = None, sort=True
    ) -> TreeItem:
        """Create a 'root' TreeItem from a nested list or a nested dictonary.

        Examples:
            with open("file.json") as file:
                data = json.dump(file)
                root = TreeItem.load(data)

        This method is a recursive function that calls itself.

        Returns:
            TreeItem: TreeItem
        """
        root_item = TreeItem(parent)
        root_item.key = 'root'

        if isinstance(value, dict):
            items = sorted(value.items()) if sort else value.items()

            for key, value in items:
                child = cls.load(value, root_item)
                child.key = key
                child.value_type = type(value)
                root_item.append_child(child)
        elif isinstance(value, list | tuple):
            for index, val in enumerate(value):
                child = cls.load(val, root_item)
                child.key = index
                child.value_type = type(val)
                root_item.append_child(child)
        else:
            root_item.value = value
            root_item.value_type = type(value)

        return root_item


class ConfigModel(QAbstractItemModel):
    """An editable model of configuration data."""

    def __init__(self, parent: QObject = None):
        """Initialize a ConfigModel."""
        super().__init__(parent)

        self._rootItem = TreeItem()
        self._headers = ('key', 'value')

    def clear(self):
        """Clear data from the model."""
        self.load({})

    def load(self, document: Mapping):
        """Load model from a nested dictionary returned by utils.read_config_file.

        Arguments:
            document (dict): JSON/YAML-compatible dictionary
        """
        self.beginResetModel()

        self._rootItem = TreeItem.load(document)
        self._rootItem.value_type = type(document)

        self.endResetModel()

        return True

    @override
    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> Any:
        """Override from QAbstractItemModel.

        Return data from a config item according index and role

        """
        if not index.isValid():
            return None

        item = index.internalPointer()

        if role == Qt.DisplayRole:
            if index.column() == 0:
                return item.key

            if index.column() == 1:
                return item.value

        if role == Qt.EditRole and index.column() == 1:
            return item.value

        return None

    @override
    def setData(self, index: QModelIndex, value: Any, role: Qt.ItemDataRole):
        """Override from QAbstractItemModel.

        Set config item according index and role

        Args:
            index (QModelIndex)
            value (Any)
            role (Qt.ItemDataRole)

        """
        if role == Qt.EditRole and index.column() == 1:
            item = index.internalPointer()
            item.value = str(value)

            self.dataChanged.emit(index, index, [Qt.EditRole])

            return True

        return False

    @override
    def headerData(
        self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole
    ):
        """Override from QAbstractItemModel.

        For the ConfigModel, it returns only data for columns (orientation = Horizontal)

        """
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return self._headers[section]
        return None

    @override
    def index(self, row: int, column: int, parent=None) -> QModelIndex:
        """Override from QAbstractItemModel.

        Return index according row, column and parent

        """
        if parent is None:
            parent = QModelIndex()

        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            parent_item = self._rootItem
        else:
            parent_item = parent.internalPointer()

        child_item = parent_item.child(row)
        if child_item:
            return self.createIndex(row, column, child_item)
        return QModelIndex()

    @override
    def parent(self, index: QModelIndex) -> QModelIndex:
        """Override from QAbstractItemModel.

        Return parent index of index

        """
        if not index.isValid():
            return QModelIndex()

        child_item = index.internalPointer()
        parent_item = child_item.parent()

        if parent_item == self._rootItem:
            return QModelIndex()

        return self.createIndex(parent_item.row(), 0, parent_item)

    @override
    def rowCount(self, parent=None):
        """Override from QAbstractItemModel.

        Return row count from parent index
        """
        if parent is None:
            parent = QModelIndex()

        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parent_item = self._rootItem
        else:
            parent_item = parent.internalPointer()

        return parent_item.child_count()

    @override
    def columnCount(self, parent=None):  # noqa: ARG002
        """Override from QAbstractItemModel.

        Return column number. For the model, it always return 2 columns
        """
        return 2

    @override
    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        """Override from QAbstractItemModel.

        Return flags of index
        """
        flags = super().flags(index)

        if index.column() == 1:
            return Qt.ItemIsEditable | flags
        return flags

    def to_json(self, item=None):
        """Convert widget data to a JSON-compatible dictionary."""
        if item is None:
            item = self._rootItem

        nchild = item.child_count()

        if item.value_type is dict:
            document = {}
            for i in range(nchild):
                ch = item.child(i)
                document[ch.key] = self.to_json(ch)
            return document

        if item.value_type in (list, tuple):
            document = []
            for i in range(nchild):
                ch = item.child(i)
                document.append(self.to_json(ch))
            return document

        return item.value


if __name__ == '__main__':

    app = QApplication(sys.argv)
    view = QTreeView()
    model = ConfigModel()

    view.setModel(model)

    config_path = 'sim.json'

    with open(config_path) as file:
        document = read_config_file(config_path)
        model.load(document)

    view.show()
    view.header().setSectionResizeMode(0, QHeaderView.Stretch)
    view.setAlternatingRowColors(True)
    view.resize(500, 300)
    app.exec()
