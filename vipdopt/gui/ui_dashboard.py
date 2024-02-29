# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'dashboard.ui'
##
## Created by: Qt User Interface Compiler version 6.6.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QFormLayout, QGridLayout, QHBoxLayout,
    QLabel, QMainWindow, QMenu, QMenuBar,
    QPushButton, QSizePolicy, QSpacerItem, QStatusBar,
    QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 600)
        self.actionNew = QAction(MainWindow)
        self.actionNew.setObjectName(u"actionNew")
        self.actionOpen = QAction(MainWindow)
        self.actionOpen.setObjectName(u"actionOpen")
        self.actionSave = QAction(MainWindow)
        self.actionSave.setObjectName(u"actionSave")
        self.actionSave_As = QAction(MainWindow)
        self.actionSave_As.setObjectName(u"actionSave_As")
        self.actionQuit = QAction(MainWindow)
        self.actionQuit.setObjectName(u"actionQuit")
        self.actionUndo = QAction(MainWindow)
        self.actionUndo.setObjectName(u"actionUndo")
        self.actionRedo = QAction(MainWindow)
        self.actionRedo.setObjectName(u"actionRedo")
        self.actionCopy = QAction(MainWindow)
        self.actionCopy.setObjectName(u"actionCopy")
        self.actionPaste = QAction(MainWindow)
        self.actionPaste.setObjectName(u"actionPaste")
        self.actionCut = QAction(MainWindow)
        self.actionCut.setObjectName(u"actionCut")
        self.actionDelete = QAction(MainWindow)
        self.actionDelete.setObjectName(u"actionDelete")
        self.actionVipdopt_Help = QAction(MainWindow)
        self.actionVipdopt_Help.setObjectName(u"actionVipdopt_Help")
        self.actionAbout_Vipdopt = QAction(MainWindow)
        self.actionAbout_Vipdopt.setObjectName(u"actionAbout_Vipdopt")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayoutWidget = QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(9, 9, 781, 531))
        self.horizontalLayout = QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")

        self.horizontalLayout.addLayout(self.gridLayout)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.label_10 = QLabel(self.horizontalLayoutWidget)
        self.label_10.setObjectName(u"label_10")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label_10)

        self.epoch_label = QLabel(self.horizontalLayoutWidget)
        self.epoch_label.setObjectName(u"epoch_label")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.epoch_label)

        self.label_12 = QLabel(self.horizontalLayoutWidget)
        self.label_12.setObjectName(u"label_12")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_12)

        self.iter_label = QLabel(self.horizontalLayoutWidget)
        self.iter_label.setObjectName(u"iter_label")

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.iter_label)

        self.time_est_label = QLabel(self.horizontalLayoutWidget)
        self.time_est_label.setObjectName(u"time_est_label")

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.time_est_label)

        self.label_14 = QLabel(self.horizontalLayoutWidget)
        self.label_14.setObjectName(u"label_14")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.label_14)


        self.verticalLayout_2.addLayout(self.formLayout)

        self.formLayout_2 = QFormLayout()
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.label_6 = QLabel(self.horizontalLayoutWidget)
        self.label_6.setObjectName(u"label_6")

        self.formLayout_2.setWidget(0, QFormLayout.LabelRole, self.label_6)

        self.avg_power_label = QLabel(self.horizontalLayoutWidget)
        self.avg_power_label.setObjectName(u"avg_power_label")

        self.formLayout_2.setWidget(0, QFormLayout.FieldRole, self.avg_power_label)

        self.label_8 = QLabel(self.horizontalLayoutWidget)
        self.label_8.setObjectName(u"label_8")

        self.formLayout_2.setWidget(1, QFormLayout.LabelRole, self.label_8)

        self.avg_e_label = QLabel(self.horizontalLayoutWidget)
        self.avg_e_label.setObjectName(u"avg_e_label")

        self.formLayout_2.setWidget(1, QFormLayout.FieldRole, self.avg_e_label)


        self.verticalLayout_2.addLayout(self.formLayout_2)

        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.edit_pushButton = QPushButton(self.horizontalLayoutWidget)
        self.edit_pushButton.setObjectName(u"edit_pushButton")

        self.gridLayout_2.addWidget(self.edit_pushButton, 1, 2, 1, 1)

        self.start_stop_pushButton = QPushButton(self.horizontalLayoutWidget)
        self.start_stop_pushButton.setObjectName(u"start_stop_pushButton")

        self.gridLayout_2.addWidget(self.start_stop_pushButton, 0, 2, 1, 1)

        self.save_pushButton = QPushButton(self.horizontalLayoutWidget)
        self.save_pushButton.setObjectName(u"save_pushButton")

        self.gridLayout_2.addWidget(self.save_pushButton, 2, 2, 1, 1)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Minimum, QSizePolicy.Minimum)

        self.gridLayout_2.addItem(self.horizontalSpacer, 0, 0, 1, 1)


        self.verticalLayout_2.addLayout(self.gridLayout_2)


        self.horizontalLayout.addLayout(self.verticalLayout_2)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 22))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuEdit = QMenu(self.menubar)
        self.menuEdit.setObjectName(u"menuEdit")
        self.menuHelp = QMenu(self.menubar)
        self.menuHelp.setObjectName(u"menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
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
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"vipdopt", None))
        self.actionNew.setText(QCoreApplication.translate("MainWindow", u"New...", None))
#if QT_CONFIG(shortcut)
        self.actionNew.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+N", None))
#endif // QT_CONFIG(shortcut)
        self.actionOpen.setText(QCoreApplication.translate("MainWindow", u"Open...", None))
#if QT_CONFIG(shortcut)
        self.actionOpen.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+O", None))
#endif // QT_CONFIG(shortcut)
        self.actionSave.setText(QCoreApplication.translate("MainWindow", u"Save", None))
#if QT_CONFIG(shortcut)
        self.actionSave.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+S", None))
#endif // QT_CONFIG(shortcut)
        self.actionSave_As.setText(QCoreApplication.translate("MainWindow", u"Save As...", None))
        self.actionQuit.setText(QCoreApplication.translate("MainWindow", u"Quit", None))
#if QT_CONFIG(shortcut)
        self.actionQuit.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Q", None))
#endif // QT_CONFIG(shortcut)
        self.actionUndo.setText(QCoreApplication.translate("MainWindow", u"Undo", None))
#if QT_CONFIG(shortcut)
        self.actionUndo.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Z", None))
#endif // QT_CONFIG(shortcut)
        self.actionRedo.setText(QCoreApplication.translate("MainWindow", u"Redo", None))
#if QT_CONFIG(shortcut)
        self.actionRedo.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+Shift+Z, Ctrl+Y", None))
#endif // QT_CONFIG(shortcut)
        self.actionCopy.setText(QCoreApplication.translate("MainWindow", u"Copy", None))
#if QT_CONFIG(shortcut)
        self.actionCopy.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+C", None))
#endif // QT_CONFIG(shortcut)
        self.actionPaste.setText(QCoreApplication.translate("MainWindow", u"Paste", None))
#if QT_CONFIG(shortcut)
        self.actionPaste.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+V", None))
#endif // QT_CONFIG(shortcut)
        self.actionCut.setText(QCoreApplication.translate("MainWindow", u"Cut", None))
#if QT_CONFIG(shortcut)
        self.actionCut.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+X", None))
#endif // QT_CONFIG(shortcut)
        self.actionDelete.setText(QCoreApplication.translate("MainWindow", u"Delete", None))
        self.actionVipdopt_Help.setText(QCoreApplication.translate("MainWindow", u"Vipdopt Help", None))
        self.actionAbout_Vipdopt.setText(QCoreApplication.translate("MainWindow", u"About Vipdopt", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Current Epoch", None))
        self.epoch_label.setText(QCoreApplication.translate("MainWindow", u"value...", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Current Iteration:", None))
        self.iter_label.setText(QCoreApplication.translate("MainWindow", u"value...", None))
        self.time_est_label.setText(QCoreApplication.translate("MainWindow", u"value...", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Estimated time Remaining:", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Average Power:", None))
        self.avg_power_label.setText(QCoreApplication.translate("MainWindow", u"value...", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Average E Field:", None))
        self.avg_e_label.setText(QCoreApplication.translate("MainWindow", u"value...", None))
        self.edit_pushButton.setText(QCoreApplication.translate("MainWindow", u"Edit Optimization...", None))
        self.start_stop_pushButton.setText(QCoreApplication.translate("MainWindow", u"Start / Stop", None))
        self.save_pushButton.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuEdit.setTitle(QCoreApplication.translate("MainWindow", u"Edit", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"Help", None))
    # retranslateUi

