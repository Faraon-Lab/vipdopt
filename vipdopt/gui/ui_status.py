
################################################################################
## Form generated from reading UI file 'vipdopt_progress.ui'
##
## Created by: Qt User Interface Compiler version 6.6.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import QCoreApplication, QMetaObject, QRect, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMenuBar,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


class Ui_MainWindow:
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName('MainWindow')
        MainWindow.resize(800, 600)
        self.actionNew = QAction(MainWindow)
        self.actionNew.setObjectName('actionNew')
        self.actionOpen = QAction(MainWindow)
        self.actionOpen.setObjectName('actionOpen')
        self.actionSave = QAction(MainWindow)
        self.actionSave.setObjectName('actionSave')
        self.actionSave_As = QAction(MainWindow)
        self.actionSave_As.setObjectName('actionSave_As')
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
        self.actionVipdopt_Help = QAction(MainWindow)
        self.actionVipdopt_Help.setObjectName('actionVipdopt_Help')
        self.actionAbout_Vipdopt = QAction(MainWindow)
        self.actionAbout_Vipdopt.setObjectName('actionAbout_Vipdopt')
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName('centralwidget')
        self.horizontalLayoutWidget = QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setObjectName('horizontalLayoutWidget')
        self.horizontalLayoutWidget.setGeometry(QRect(9, 9, 781, 531))
        self.horizontalLayout = QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setObjectName('horizontalLayout')
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName('gridLayout')
        self.label = QLabel(self.horizontalLayoutWidget)
        self.label.setObjectName('label')
        self.label.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.label_3 = QLabel(self.horizontalLayoutWidget)
        self.label_3.setObjectName('label_3')
        self.label_3.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)

        self.label_4 = QLabel(self.horizontalLayoutWidget)
        self.label_4.setObjectName('label_4')
        self.label_4.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_4, 1, 1, 1, 1)

        self.label_2 = QLabel(self.horizontalLayoutWidget)
        self.label_2.setObjectName('label_2')
        self.label_2.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.label_2, 0, 1, 1, 1)


        self.horizontalLayout.addLayout(self.gridLayout)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName('verticalLayout_2')
        self.label_5 = QLabel(self.horizontalLayoutWidget)
        self.label_5.setObjectName('label_5')
        self.label_5.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.label_5)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName('formLayout')
        self.label_10 = QLabel(self.horizontalLayoutWidget)
        self.label_10.setObjectName('label_10')

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label_10)

        self.label_11 = QLabel(self.horizontalLayoutWidget)
        self.label_11.setObjectName('label_11')

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.label_11)

        self.label_12 = QLabel(self.horizontalLayoutWidget)
        self.label_12.setObjectName('label_12')

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_12)

        self.label_13 = QLabel(self.horizontalLayoutWidget)
        self.label_13.setObjectName('label_13')

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.label_13)

        self.label_15 = QLabel(self.horizontalLayoutWidget)
        self.label_15.setObjectName('label_15')

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.label_15)

        self.label_14 = QLabel(self.horizontalLayoutWidget)
        self.label_14.setObjectName('label_14')

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.label_14)


        self.verticalLayout_2.addLayout(self.formLayout)

        self.formLayout_2 = QFormLayout()
        self.formLayout_2.setObjectName('formLayout_2')
        self.label_6 = QLabel(self.horizontalLayoutWidget)
        self.label_6.setObjectName('label_6')

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.label_6)

        self.label_7 = QLabel(self.horizontalLayoutWidget)
        self.label_7.setObjectName('label_7')

        self.formLayout_2.setWidget(0, QFormLayout.FieldRole, self.label_7)

        self.label_8 = QLabel(self.horizontalLayoutWidget)
        self.label_8.setObjectName('label_8')

        self.formLayout_2.setWidget(1, QFormLayout.LabelRole, self.label_8)

        self.label_9 = QLabel(self.horizontalLayoutWidget)
        self.label_9.setObjectName('label_9')

        self.formLayout_2.setWidget(1, QFormLayout.FieldRole, self.label_9)


        self.verticalLayout_2.addLayout(self.formLayout_2)

        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName('gridLayout_2')
        self.pushButton_3 = QPushButton(self.horizontalLayoutWidget)
        self.pushButton_3.setObjectName('pushButton_3')

        self.gridLayout_2.addWidget(self.pushButton_3, 1, 2, 1, 1)

        self.pushButton = QPushButton(self.horizontalLayoutWidget)
        self.pushButton.setObjectName('pushButton')

        self.gridLayout_2.addWidget(self.pushButton, 0, 2, 1, 1)

        self.pushButton_2 = QPushButton(self.horizontalLayoutWidget)
        self.pushButton_2.setObjectName('pushButton_2')

        self.gridLayout_2.addWidget(self.pushButton_2, 2, 2, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer, 0, 0, 1, 1)


        self.verticalLayout_2.addLayout(self.gridLayout_2)


        self.horizontalLayout.addLayout(self.verticalLayout_2)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName('menubar')
        self.menubar.setGeometry(QRect(0, 0, 800, 19))
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
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit)
        self.menuEdit.addAction(self.actionUndo)
        self.menuEdit.addAction(self.actionRedo)
        self.menuEdit.addSeparator()
        self.menuEdit.addAction(self.actionCopy)
        self.menuEdit.addAction(self.actionPaste)
        self.menuEdit.addAction(self.actionCut)
        self.menuEdit.addAction(self.actionDelete)
        self.menuHelp.addAction(self.actionVipdopt_Help)
        self.menuHelp.addSeparator()
        self.menuHelp.addAction(self.actionAbout_Vipdopt)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate('MainWindow', 'vipdopt', None))
        self.actionNew.setText(QCoreApplication.translate('MainWindow', 'New...', None))
#if QT_CONFIG(shortcut)
        self.actionNew.setShortcut(QCoreApplication.translate('MainWindow', 'Ctrl+N', None))
        self.actionOpen.setText(QCoreApplication.translate('MainWindow', 'Open...', None))
#if QT_CONFIG(shortcut)
        self.actionOpen.setShortcut(QCoreApplication.translate('MainWindow', 'Ctrl+O', None))
        self.actionSave.setText(QCoreApplication.translate('MainWindow', 'Save', None))
#if QT_CONFIG(shortcut)
        self.actionSave.setShortcut(QCoreApplication.translate('MainWindow', 'Ctrl+S', None))
        self.actionSave_As.setText(QCoreApplication.translate('MainWindow', 'Save As...', None))
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
        self.actionVipdopt_Help.setText(QCoreApplication.translate('MainWindow', 'Vipdopt Help', None))
        self.actionAbout_Vipdopt.setText(QCoreApplication.translate('MainWindow', 'About Vipdopt', None))
        self.label.setText(QCoreApplication.translate('MainWindow', 'Plot 1', None))
        self.label_3.setText(QCoreApplication.translate('MainWindow', 'Plot 3', None))
        self.label_4.setText(QCoreApplication.translate('MainWindow', 'Plot 4', None))
        self.label_2.setText(QCoreApplication.translate('MainWindow', 'Plot 2', None))
        self.label_5.setText(QCoreApplication.translate('MainWindow', 'Current Device', None))
        self.label_10.setText(QCoreApplication.translate('MainWindow', 'Current Epoch', None))
        self.label_11.setText(QCoreApplication.translate('MainWindow', 'value...', None))
        self.label_12.setText(QCoreApplication.translate('MainWindow', 'Current Iteration:', None))
        self.label_13.setText(QCoreApplication.translate('MainWindow', 'value...', None))
        self.label_15.setText(QCoreApplication.translate('MainWindow', 'value...', None))
        self.label_14.setText(QCoreApplication.translate('MainWindow', 'Estimated time Remaining:', None))
        self.label_6.setText(QCoreApplication.translate('MainWindow', 'Average Power:', None))
        self.label_7.setText(QCoreApplication.translate('MainWindow', 'value...', None))
        self.label_8.setText(QCoreApplication.translate('MainWindow', 'Average E Field:', None))
        self.label_9.setText(QCoreApplication.translate('MainWindow', 'value...', None))
        self.pushButton_3.setText(QCoreApplication.translate('MainWindow', 'Edit Optimization...', None))
        self.pushButton.setText(QCoreApplication.translate('MainWindow', 'Start / Stop', None))
        self.pushButton_2.setText(QCoreApplication.translate('MainWindow', 'Save', None))
        self.menuFile.setTitle(QCoreApplication.translate('MainWindow', 'File', None))
        self.menuEdit.setTitle(QCoreApplication.translate('MainWindow', 'Edit', None))
        self.menuHelp.setTitle(QCoreApplication.translate('MainWindow', 'Help', None))
    # retranslateUi

