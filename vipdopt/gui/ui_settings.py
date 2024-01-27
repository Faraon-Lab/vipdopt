
################################################################################
## Form generated from reading UI file 'vipdopt_configs.ui'
##
## Created by: Qt User Interface Compiler version 6.6.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import QCoreApplication, QMetaObject, QRect, Qt
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMenuBar,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class Ui_MainWindow:
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName('MainWindow')
        MainWindow.resize(819, 626)
        self.actionSave = QAction(MainWindow)
        self.actionSave.setObjectName('actionSave')
        self.actionOpen = QAction(MainWindow)
        self.actionOpen.setObjectName('actionOpen')
        self.actionNew = QAction(MainWindow)
        self.actionNew.setObjectName('actionNew')
        self.actionSave_as = QAction(MainWindow)
        self.actionSave_as.setObjectName('actionSave_as')
        self.actionQuit = QAction(MainWindow)
        self.actionQuit.setObjectName('actionQuit')
        self.actionUndo = QAction(MainWindow)
        self.actionUndo.setObjectName('actionUndo')
        self.actionRedo = QAction(MainWindow)
        self.actionRedo.setObjectName('actionRedo')
        self.actionCopy = QAction(MainWindow)
        self.actionCopy.setObjectName('actionCopy')
        self.actionPaste = QAction(MainWindow)
        self.actionPaste.setObjectName('actionPaste')
        self.actionCut = QAction(MainWindow)
        self.actionCut.setObjectName('actionCut')
        self.actionDelete = QAction(MainWindow)
        self.actionDelete.setObjectName('actionDelete')
        self.actionvipdopt_help = QAction(MainWindow)
        self.actionvipdopt_help.setObjectName('actionvipdopt_help')
        self.actionAbout_Vipdopt = QAction(MainWindow)
        self.actionAbout_Vipdopt.setObjectName('actionAbout_Vipdopt')
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName('centralwidget')
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName('tabWidget')
        self.tabWidget.setGeometry(QRect(0, 0, 821, 581))
        self.tab = QWidget()
        self.tab.setObjectName('tab')
        self.verticalLayoutWidget = QWidget(self.tab)
        self.verticalLayoutWidget.setObjectName('verticalLayoutWidget')
        self.verticalLayoutWidget.setGeometry(QRect(10, 10, 801, 531))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName('verticalLayout')
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName('horizontalLayout')
        self.label = QLabel(self.verticalLayoutWidget)
        self.label.setObjectName('label')

        self.horizontalLayout.addWidget(self.label)

        self.lineEdit = QLineEdit(self.verticalLayoutWidget)
        self.lineEdit.setObjectName('lineEdit')

        self.horizontalLayout.addWidget(self.lineEdit)

        self.pushButton = QPushButton(self.verticalLayoutWidget)
        self.pushButton.setObjectName('pushButton')

        self.horizontalLayout.addWidget(self.pushButton)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.tableWidget = QTableWidget(self.verticalLayoutWidget)
        if (self.tableWidget.columnCount() < 2):
            self.tableWidget.setColumnCount(2)
        __qtablewidgetitem = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        if (self.tableWidget.rowCount() < 20):
            self.tableWidget.setRowCount(20)
        self.tableWidget.setObjectName('tableWidget')
        self.tableWidget.setAutoFillBackground(False)
        self.tableWidget.setRowCount(20)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(False)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(300)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)

        self.verticalLayout.addWidget(self.tableWidget)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName('horizontalLayout_3')
        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_3)

        self.pushButton_3 = QPushButton(self.verticalLayoutWidget)
        self.pushButton_3.setObjectName('pushButton_3')

        self.horizontalLayout_3.addWidget(self.pushButton_3)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.tabWidget.addTab(self.tab, '')
        self.tab_4 = QWidget()
        self.tab_4.setObjectName('tab_4')
        self.verticalLayoutWidget_3 = QWidget(self.tab_4)
        self.verticalLayoutWidget_3.setObjectName('verticalLayoutWidget_3')
        self.verticalLayoutWidget_3.setGeometry(QRect(9, 9, 801, 531))
        self.verticalLayout_4 = QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_4.setObjectName('verticalLayout_4')
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName('horizontalLayout_2')
        self.label_3 = QLabel(self.verticalLayoutWidget_3)
        self.label_3.setObjectName('label_3')

        self.horizontalLayout_2.addWidget(self.label_3)

        self.lineEdit_2 = QLineEdit(self.verticalLayoutWidget_3)
        self.lineEdit_2.setObjectName('lineEdit_2')

        self.horizontalLayout_2.addWidget(self.lineEdit_2)

        self.pushButton_2 = QPushButton(self.verticalLayoutWidget_3)
        self.pushButton_2.setObjectName('pushButton_2')

        self.horizontalLayout_2.addWidget(self.pushButton_2)


        self.verticalLayout_4.addLayout(self.horizontalLayout_2)

        self.label_2 = QLabel(self.verticalLayoutWidget_3)
        self.label_2.setObjectName('label_2')

        self.verticalLayout_4.addWidget(self.label_2)

        self.treeWidget = QTreeWidget(self.verticalLayoutWidget_3)
        __qtreewidgetitem = QTreeWidgetItem(self.treeWidget)
        __qtreewidgetitem1 = QTreeWidgetItem(__qtreewidgetitem)
        __qtreewidgetitem1.setCheckState(0, Qt.Checked)
        __qtreewidgetitem2 = QTreeWidgetItem(__qtreewidgetitem)
        __qtreewidgetitem2.setCheckState(0, Qt.Unchecked)
        __qtreewidgetitem3 = QTreeWidgetItem(self.treeWidget)
        __qtreewidgetitem4 = QTreeWidgetItem(__qtreewidgetitem3)
        __qtreewidgetitem4.setCheckState(0, Qt.Unchecked)
        __qtreewidgetitem5 = QTreeWidgetItem(__qtreewidgetitem3)
        __qtreewidgetitem5.setCheckState(0, Qt.Checked)
        self.treeWidget.setObjectName('treeWidget')
        self.treeWidget.setUniformRowHeights(True)
        self.treeWidget.header().setCascadingSectionResizes(False)
        self.treeWidget.header().setStretchLastSection(False)

        self.verticalLayout_4.addWidget(self.treeWidget)

        self.tabWidget.addTab(self.tab_4, '')
        self.tab_2 = QWidget()
        self.tab_2.setObjectName('tab_2')
        self.verticalLayoutWidget_2 = QWidget(self.tab_2)
        self.verticalLayoutWidget_2.setObjectName('verticalLayoutWidget_2')
        self.verticalLayoutWidget_2.setGeometry(QRect(10, 10, 801, 531))
        self.verticalLayout_2 = QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setObjectName('verticalLayout_2')
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName('gridLayout_2')
        self.label_18 = QLabel(self.verticalLayoutWidget_2)
        self.label_18.setObjectName('label_18')

        self.gridLayout_2.addWidget(self.label_18, 0, 3, 1, 1)

        self.lineEdit_15 = QLineEdit(self.verticalLayoutWidget_2)
        self.lineEdit_15.setObjectName('lineEdit_15')
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_15.sizePolicy().hasHeightForWidth())
        self.lineEdit_15.setSizePolicy(sizePolicy)

        self.gridLayout_2.addWidget(self.lineEdit_15, 1, 4, 1, 1)

        self.label_17 = QLabel(self.verticalLayoutWidget_2)
        self.label_17.setObjectName('label_17')

        self.gridLayout_2.addWidget(self.label_17, 0, 2, 1, 1)

        self.label_16 = QLabel(self.verticalLayoutWidget_2)
        self.label_16.setObjectName('label_16')

        self.gridLayout_2.addWidget(self.label_16, 0, 1, 1, 1)

        self.lineEdit_11 = QLineEdit(self.verticalLayoutWidget_2)
        self.lineEdit_11.setObjectName('lineEdit_11')
        sizePolicy1 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.lineEdit_11.sizePolicy().hasHeightForWidth())
        self.lineEdit_11.setSizePolicy(sizePolicy1)

        self.gridLayout_2.addWidget(self.lineEdit_11, 1, 0, 1, 1)

        self.comboBox_2 = QComboBox(self.verticalLayoutWidget_2)
        self.comboBox_2.setObjectName('comboBox_2')

        self.gridLayout_2.addWidget(self.comboBox_2, 1, 1, 1, 1)

        self.label_15 = QLabel(self.verticalLayoutWidget_2)
        self.label_15.setObjectName('label_15')

        self.gridLayout_2.addWidget(self.label_15, 0, 0, 1, 1)

        self.comboBox_4 = QComboBox(self.verticalLayoutWidget_2)
        self.comboBox_4.setObjectName('comboBox_4')

        self.gridLayout_2.addWidget(self.comboBox_4, 1, 3, 1, 1)

        self.comboBox_3 = QComboBox(self.verticalLayoutWidget_2)
        self.comboBox_3.setObjectName('comboBox_3')
        self.comboBox_3.setEditable(False)

        self.gridLayout_2.addWidget(self.comboBox_3, 1, 2, 1, 1)

        self.label_19 = QLabel(self.verticalLayoutWidget_2)
        self.label_19.setObjectName('label_19')

        self.gridLayout_2.addWidget(self.label_19, 0, 4, 1, 1)


        self.verticalLayout_2.addLayout(self.gridLayout_2)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName('horizontalLayout_5')
        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_4)

        self.toolButton = QToolButton(self.verticalLayoutWidget_2)
        self.toolButton.setObjectName('toolButton')

        self.horizontalLayout_5.addWidget(self.toolButton)

        self.toolButton_2 = QToolButton(self.verticalLayoutWidget_2)
        self.toolButton_2.setObjectName('toolButton_2')

        self.horizontalLayout_5.addWidget(self.toolButton_2)


        self.verticalLayout_2.addLayout(self.horizontalLayout_5)

        self.tabWidget.addTab(self.tab_2, '')
        self.tab_3 = QWidget()
        self.tab_3.setObjectName('tab_3')
        self.verticalLayoutWidget_4 = QWidget(self.tab_3)
        self.verticalLayoutWidget_4.setObjectName('verticalLayoutWidget_4')
        self.verticalLayoutWidget_4.setGeometry(QRect(9, 9, 801, 531))
        self.verticalLayout_5 = QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_5.setObjectName('verticalLayout_5')
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.label_14 = QLabel(self.verticalLayoutWidget_4)
        self.label_14.setObjectName('label_14')
        font = QFont()
        font.setPointSize(20)
        self.label_14.setFont(font)
        self.label_14.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.label_14)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName('formLayout')
        self.label_4 = QLabel(self.verticalLayoutWidget_4)
        self.label_4.setObjectName('label_4')

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_4)

        self.lineEdit_3 = QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_3.setObjectName('lineEdit_3')

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.lineEdit_3)

        self.label_6 = QLabel(self.verticalLayoutWidget_4)
        self.label_6.setObjectName('label_6')

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.label_6)

        self.lineEdit_5 = QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_5.setObjectName('lineEdit_5')

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.lineEdit_5)

        self.label_5 = QLabel(self.verticalLayoutWidget_4)
        self.label_5.setObjectName('label_5')

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.label_5)

        self.lineEdit_4 = QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_4.setObjectName('lineEdit_4')

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.lineEdit_4)

        self.label_7 = QLabel(self.verticalLayoutWidget_4)
        self.label_7.setObjectName('label_7')

        self.formLayout.setWidget(4, QFormLayout.LabelRole, self.label_7)

        self.lineEdit_6 = QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_6.setObjectName('lineEdit_6')

        self.formLayout.setWidget(4, QFormLayout.FieldRole, self.lineEdit_6)

        self.label_11 = QLabel(self.verticalLayoutWidget_4)
        self.label_11.setObjectName('label_11')

        self.formLayout.setWidget(5, QFormLayout.LabelRole, self.label_11)

        self.lineEdit_8 = QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_8.setObjectName('lineEdit_8')

        self.formLayout.setWidget(5, QFormLayout.FieldRole, self.lineEdit_8)


        self.verticalLayout_5.addLayout(self.formLayout)

        self.line = QFrame(self.verticalLayoutWidget_4)
        self.line.setObjectName('line')
        self.line.setLineWidth(6)
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_5.addWidget(self.line)

        self.label_8 = QLabel(self.verticalLayoutWidget_4)
        self.label_8.setObjectName('label_8')
        self.label_8.setFont(font)
        self.label_8.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.label_8)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName('horizontalLayout_7')
        self.label_9 = QLabel(self.verticalLayoutWidget_4)
        self.label_9.setObjectName('label_9')

        self.horizontalLayout_7.addWidget(self.label_9)

        self.comboBox = QComboBox(self.verticalLayoutWidget_4)
        self.comboBox.addItem('')
        self.comboBox.setObjectName('comboBox')

        self.horizontalLayout_7.addWidget(self.comboBox)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_2)


        self.verticalLayout_5.addLayout(self.horizontalLayout_7)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName('gridLayout')
        self.lineEdit_10 = QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_10.setObjectName('lineEdit_10')

        self.gridLayout.addWidget(self.lineEdit_10, 1, 2, 1, 1)

        self.label_10 = QLabel(self.verticalLayoutWidget_4)
        self.label_10.setObjectName('label_10')

        self.gridLayout.addWidget(self.label_10, 0, 1, 1, 1)

        self.lineEdit_7 = QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_7.setObjectName('lineEdit_7')

        self.gridLayout.addWidget(self.lineEdit_7, 0, 2, 1, 1)

        self.lineEdit_9 = QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_9.setObjectName('lineEdit_9')

        self.gridLayout.addWidget(self.lineEdit_9, 3, 2, 1, 1)

        self.label_12 = QLabel(self.verticalLayoutWidget_4)
        self.label_12.setObjectName('label_12')

        self.gridLayout.addWidget(self.label_12, 3, 1, 1, 1)

        self.label_13 = QLabel(self.verticalLayoutWidget_4)
        self.label_13.setObjectName('label_13')

        self.gridLayout.addWidget(self.label_13, 1, 1, 1, 1)

        self.horizontalSpacer = QSpacerItem(150, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.gridLayout.addItem(self.horizontalSpacer, 0, 0, 1, 1)


        self.verticalLayout_5.addLayout(self.gridLayout)

        self.tabWidget.addTab(self.tab_3, '')
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName('menubar')
        self.menubar.setGeometry(QRect(0, 0, 819, 19))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName('menuFile')
        self.menuEdit = QMenu(self.menubar)
        self.menuEdit.setObjectName('menuEdit')
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName('menuHelp')
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName('statusbar')
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())
        self.menuFile.addAction(self.actionNew)
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_as)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit)
        self.menuEdit.addAction(self.actionUndo)
        self.menuEdit.addAction(self.actionRedo)
        self.menuEdit.addSeparator()
        self.menuEdit.addAction(self.actionCopy)
        self.menuEdit.addAction(self.actionPaste)
        self.menuEdit.addAction(self.actionCut)
        self.menuEdit.addAction(self.actionDelete)
        self.menuHelp.addAction(self.actionvipdopt_help)
        self.menuHelp.addSeparator()
        self.menuHelp.addAction(self.actionAbout_Vipdopt)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate('MainWindow', 'vipdopt', None))
        self.actionSave.setText(QCoreApplication.translate('MainWindow', 'Save', None))
#if QT_CONFIG(shortcut)
        self.actionSave.setShortcut(QCoreApplication.translate('MainWindow', 'Ctrl+S', None))
        self.actionOpen.setText(QCoreApplication.translate('MainWindow', 'Open...', None))
#if QT_CONFIG(shortcut)
        self.actionOpen.setShortcut(QCoreApplication.translate('MainWindow', 'Ctrl+O', None))
        self.actionNew.setText(QCoreApplication.translate('MainWindow', 'New...', None))
#if QT_CONFIG(shortcut)
        self.actionNew.setShortcut(QCoreApplication.translate('MainWindow', 'Ctrl+N', None))
        self.actionSave_as.setText(QCoreApplication.translate('MainWindow', 'Save As...', None))
        self.actionQuit.setText(QCoreApplication.translate('MainWindow', 'Quit', None))
#if QT_CONFIG(shortcut)
        self.actionQuit.setShortcut(QCoreApplication.translate('MainWindow', 'Ctrl+Q', None))
        self.actionUndo.setText(QCoreApplication.translate('MainWindow', 'Undo', None))
#if QT_CONFIG(shortcut)
        self.actionUndo.setShortcut(QCoreApplication.translate('MainWindow', 'Ctrl+Z', None))
        self.actionRedo.setText(QCoreApplication.translate('MainWindow', 'Redo', None))
#if QT_CONFIG(shortcut)
        self.actionRedo.setShortcut(QCoreApplication.translate('MainWindow', 'Ctrl+Shift+Z, Ctrl+Y', None))
        self.actionCopy.setText(QCoreApplication.translate('MainWindow', 'Copy', None))
#if QT_CONFIG(shortcut)
        self.actionCopy.setShortcut(QCoreApplication.translate('MainWindow', 'Ctrl+C', None))
        self.actionPaste.setText(QCoreApplication.translate('MainWindow', 'Paste', None))
#if QT_CONFIG(shortcut)
        self.actionPaste.setShortcut(QCoreApplication.translate('MainWindow', 'Ctrl+V', None))
        self.actionCut.setText(QCoreApplication.translate('MainWindow', 'Cut', None))
#if QT_CONFIG(shortcut)
        self.actionCut.setShortcut(QCoreApplication.translate('MainWindow', 'Ctrl+X', None))
        self.actionDelete.setText(QCoreApplication.translate('MainWindow', 'Delete', None))
        self.actionvipdopt_help.setText(QCoreApplication.translate('MainWindow', 'Vipdopt Help', None))
#if QT_CONFIG(shortcut)
        self.actionvipdopt_help.setShortcut(QCoreApplication.translate('MainWindow', 'Ctrl+H', None))
        self.actionAbout_Vipdopt.setText(QCoreApplication.translate('MainWindow', 'About Vipdopt', None))
        self.label.setText(QCoreApplication.translate('MainWindow', 'Load Configuration File:', None))
        self.pushButton.setText(QCoreApplication.translate('MainWindow', 'Browse...', None))
        ___qtablewidgetitem = self.tableWidget.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate('MainWindow', 'Property', None))
        ___qtablewidgetitem1 = self.tableWidget.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate('MainWindow', 'Value', None))
        self.pushButton_3.setText(QCoreApplication.translate('MainWindow', 'Save...', None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QCoreApplication.translate('MainWindow', 'Configuration', None))
        self.label_3.setText(QCoreApplication.translate('MainWindow', 'Load Base Simulation File:', None))
        self.pushButton_2.setText(QCoreApplication.translate('MainWindow', 'Browse...', None))
        self.label_2.setText(QCoreApplication.translate('MainWindow', 'Simulation Configuration:', None))
        ___qtreewidgetitem = self.treeWidget.headerItem()
        ___qtreewidgetitem.setText(1, QCoreApplication.translate('MainWindow', 'Enabled Sources', None))
        ___qtreewidgetitem.setText(0, QCoreApplication.translate('MainWindow', 'Simulation', None))

        __sortingEnabled = self.treeWidget.isSortingEnabled()
        self.treeWidget.setSortingEnabled(False)
        ___qtreewidgetitem1 = self.treeWidget.topLevelItem(0)
        ___qtreewidgetitem1.setText(0, QCoreApplication.translate('MainWindow', 'Simulation 1', None))
        ___qtreewidgetitem2 = ___qtreewidgetitem1.child(0)
        ___qtreewidgetitem2.setText(0, QCoreApplication.translate('MainWindow', 'Source 1', None))
        ___qtreewidgetitem3 = ___qtreewidgetitem1.child(1)
        ___qtreewidgetitem3.setText(0, QCoreApplication.translate('MainWindow', 'Source 2', None))
        ___qtreewidgetitem4 = self.treeWidget.topLevelItem(1)
        ___qtreewidgetitem4.setText(0, QCoreApplication.translate('MainWindow', 'Simulation 2', None))
        ___qtreewidgetitem5 = ___qtreewidgetitem4.child(0)
        ___qtreewidgetitem5.setText(0, QCoreApplication.translate('MainWindow', 'Source 1', None))
        ___qtreewidgetitem6 = ___qtreewidgetitem4.child(1)
        ___qtreewidgetitem6.setText(0, QCoreApplication.translate('MainWindow', 'Source 2', None))
        self.treeWidget.setSortingEnabled(__sortingEnabled)

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), QCoreApplication.translate('MainWindow', 'Simulation', None))
        self.label_18.setText(QCoreApplication.translate('MainWindow', 'Gradient Monitors', None))
        self.lineEdit_15.setPlaceholderText(QCoreApplication.translate('MainWindow', '1.0', None))
        self.label_17.setText(QCoreApplication.translate('MainWindow', 'Figure of Merit Monitors', None))
        self.label_16.setText(QCoreApplication.translate('MainWindow', 'Type', None))
        self.lineEdit_11.setPlaceholderText(QCoreApplication.translate('MainWindow', 'FoM 0', None))
        self.comboBox_2.setPlaceholderText(QCoreApplication.translate('MainWindow', 'Select...', None))
        self.label_15.setText(QCoreApplication.translate('MainWindow', 'Name', None))
        self.comboBox_4.setPlaceholderText(QCoreApplication.translate('MainWindow', 'Select Monitors...', None))
        self.comboBox_3.setPlaceholderText(QCoreApplication.translate('MainWindow', 'Select Monitors...', None))
        self.label_19.setText(QCoreApplication.translate('MainWindow', 'Weight', None))
        self.toolButton.setText(QCoreApplication.translate('MainWindow', 'Add Row', None))
        self.toolButton_2.setText(QCoreApplication.translate('MainWindow', 'Remove Row', None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QCoreApplication.translate('MainWindow', 'Figure of Merit', None))
        self.label_14.setText(QCoreApplication.translate('MainWindow', 'Optimization Settings', None))
        self.label_4.setText(QCoreApplication.translate('MainWindow', 'Maximum Iterations:', None))
        self.label_6.setText(QCoreApplication.translate('MainWindow', 'Iterations per Epoch:', None))
        self.label_5.setText(QCoreApplication.translate('MainWindow', 'Maximum Epochs:', None))
        self.label_7.setText(QCoreApplication.translate('MainWindow', 'Binarization Frequency:', None))
        self.label_11.setText(QCoreApplication.translate('MainWindow', 'Step Size:', None))
        self.label_8.setText(QCoreApplication.translate('MainWindow', 'Optimizer Settings', None))
        self.label_9.setText(QCoreApplication.translate('MainWindow', 'Optimizer:', None))
        self.comboBox.setItemText(0, QCoreApplication.translate('MainWindow', 'Adam', None))

        self.label_10.setText(QCoreApplication.translate('MainWindow', 'Beta 1:', None))
        self.label_12.setText(QCoreApplication.translate('MainWindow', 'Epsilon:', None))
        self.label_13.setText(QCoreApplication.translate('MainWindow', 'Beta 2:', None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), QCoreApplication.translate('MainWindow', 'Optimization Options', None))
        self.menuFile.setTitle(QCoreApplication.translate('MainWindow', 'File', None))
        self.menuEdit.setTitle(QCoreApplication.translate('MainWindow', 'Edit', None))
        self.menuHelp.setTitle(QCoreApplication.translate('MainWindow', 'Help', None))
    # retranslateUi

