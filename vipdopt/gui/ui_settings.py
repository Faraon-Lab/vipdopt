# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'settings.ui'
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
from PySide6.QtWidgets import (QApplication, QComboBox, QFormLayout, QFrame,
    QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMainWindow, QMenu, QMenuBar,
    QPushButton, QSizePolicy, QSpacerItem, QStatusBar,
    QTabWidget, QToolButton, QTreeView, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(819, 626)
        self.actionSave = QAction(MainWindow)
        self.actionSave.setObjectName(u"actionSave")
        self.actionOpen = QAction(MainWindow)
        self.actionOpen.setObjectName(u"actionOpen")
        self.actionNew = QAction(MainWindow)
        self.actionNew.setObjectName(u"actionNew")
        self.actionSave_as = QAction(MainWindow)
        self.actionSave_as.setObjectName(u"actionSave_as")
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
        self.actionvipdopt_help = QAction(MainWindow)
        self.actionvipdopt_help.setObjectName(u"actionvipdopt_help")
        self.actionAbout_Vipdopt = QAction(MainWindow)
        self.actionAbout_Vipdopt.setObjectName(u"actionAbout_Vipdopt")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setGeometry(QRect(0, 0, 821, 581))
        self.config_tab = QWidget()
        self.config_tab.setObjectName(u"config_tab")
        self.verticalLayoutWidget = QWidget(self.config_tab)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(10, 10, 801, 531))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(self.verticalLayoutWidget)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.config_lineEdit = QLineEdit(self.verticalLayoutWidget)
        self.config_lineEdit.setObjectName(u"config_lineEdit")

        self.horizontalLayout.addWidget(self.config_lineEdit)

        self.config_pushButton = QPushButton(self.verticalLayoutWidget)
        self.config_pushButton.setObjectName(u"config_pushButton")

        self.horizontalLayout.addWidget(self.config_pushButton)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.config_treeView = QTreeView(self.verticalLayoutWidget)
        self.config_treeView.setObjectName(u"config_treeView")
        self.config_treeView.setAlternatingRowColors(True)
        self.config_treeView.header().setCascadingSectionResizes(True)

        self.verticalLayout.addWidget(self.config_treeView)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_3)

        self.pushButton_3 = QPushButton(self.verticalLayoutWidget)
        self.pushButton_3.setObjectName(u"pushButton_3")

        self.horizontalLayout_3.addWidget(self.pushButton_3)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.tabWidget.addTab(self.config_tab, "")
        self.sim_tab = QWidget()
        self.sim_tab.setObjectName(u"sim_tab")
        self.verticalLayoutWidget_3 = QWidget(self.sim_tab)
        self.verticalLayoutWidget_3.setObjectName(u"verticalLayoutWidget_3")
        self.verticalLayoutWidget_3.setGeometry(QRect(9, 9, 801, 531))
        self.verticalLayout_4 = QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_3 = QLabel(self.verticalLayoutWidget_3)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_2.addWidget(self.label_3)

        self.sim_lineEdit = QLineEdit(self.verticalLayoutWidget_3)
        self.sim_lineEdit.setObjectName(u"sim_lineEdit")

        self.horizontalLayout_2.addWidget(self.sim_lineEdit)

        self.sim_pushButton = QPushButton(self.verticalLayoutWidget_3)
        self.sim_pushButton.setObjectName(u"sim_pushButton")

        self.horizontalLayout_2.addWidget(self.sim_pushButton)


        self.verticalLayout_4.addLayout(self.horizontalLayout_2)

        self.sim_treeView = QTreeView(self.verticalLayoutWidget_3)
        self.sim_treeView.setObjectName(u"sim_treeView")
        self.sim_treeView.setAlternatingRowColors(True)
        self.sim_treeView.header().setCascadingSectionResizes(True)

        self.verticalLayout_4.addWidget(self.sim_treeView)

        self.label_2 = QLabel(self.verticalLayoutWidget_3)
        self.label_2.setObjectName(u"label_2")

        self.verticalLayout_4.addWidget(self.label_2)

        self.sim_config_treeWidget = QTreeWidget(self.verticalLayoutWidget_3)
        __qtreewidgetitem = QTreeWidgetItem(self.sim_config_treeWidget)
        __qtreewidgetitem1 = QTreeWidgetItem(__qtreewidgetitem)
        __qtreewidgetitem1.setCheckState(0, Qt.Checked);
        __qtreewidgetitem2 = QTreeWidgetItem(__qtreewidgetitem)
        __qtreewidgetitem2.setCheckState(0, Qt.Unchecked);
        __qtreewidgetitem3 = QTreeWidgetItem(self.sim_config_treeWidget)
        __qtreewidgetitem4 = QTreeWidgetItem(__qtreewidgetitem3)
        __qtreewidgetitem4.setCheckState(0, Qt.Unchecked);
        __qtreewidgetitem5 = QTreeWidgetItem(__qtreewidgetitem3)
        __qtreewidgetitem5.setCheckState(0, Qt.Checked);
        self.sim_config_treeWidget.setObjectName(u"sim_config_treeWidget")
        self.sim_config_treeWidget.setAlternatingRowColors(True)
        self.sim_config_treeWidget.setUniformRowHeights(True)
        self.sim_config_treeWidget.header().setCascadingSectionResizes(True)
        self.sim_config_treeWidget.header().setStretchLastSection(False)

        self.verticalLayout_4.addWidget(self.sim_config_treeWidget)

        self.tabWidget.addTab(self.sim_tab, "")
        self.fom_tab = QWidget()
        self.fom_tab.setObjectName(u"fom_tab")
        self.verticalLayoutWidget_2 = QWidget(self.fom_tab)
        self.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.verticalLayoutWidget_2.setGeometry(QRect(10, 10, 801, 531))
        self.verticalLayout_2 = QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.fom_gridLayout = QGridLayout()
        self.fom_gridLayout.setObjectName(u"fom_gridLayout")
        self.fom_weight_label = QLabel(self.verticalLayoutWidget_2)
        self.fom_weight_label.setObjectName(u"fom_weight_label")

        self.fom_gridLayout.addWidget(self.fom_weight_label, 0, 4, 1, 1)

        self.fom_type_label = QLabel(self.verticalLayoutWidget_2)
        self.fom_type_label.setObjectName(u"fom_type_label")

        self.fom_gridLayout.addWidget(self.fom_type_label, 0, 1, 1, 1)

        self.fom_weight_lineEdit_0 = QLineEdit(self.verticalLayoutWidget_2)
        self.fom_weight_lineEdit_0.setObjectName(u"fom_weight_lineEdit_0")
        sizePolicy = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.fom_weight_lineEdit_0.sizePolicy().hasHeightForWidth())
        self.fom_weight_lineEdit_0.setSizePolicy(sizePolicy)

        self.fom_gridLayout.addWidget(self.fom_weight_lineEdit_0, 1, 4, 1, 1)

        self.fom_grad_mon_label = QLabel(self.verticalLayoutWidget_2)
        self.fom_grad_mon_label.setObjectName(u"fom_grad_mon_label")

        self.fom_gridLayout.addWidget(self.fom_grad_mon_label, 0, 3, 1, 1)

        self.fom_fom_mon_label = QLabel(self.verticalLayoutWidget_2)
        self.fom_fom_mon_label.setObjectName(u"fom_fom_mon_label")

        self.fom_gridLayout.addWidget(self.fom_fom_mon_label, 0, 2, 1, 1)

        self.fom_type_comboBox_0 = QComboBox(self.verticalLayoutWidget_2)
        self.fom_type_comboBox_0.setObjectName(u"fom_type_comboBox_0")

        self.fom_gridLayout.addWidget(self.fom_type_comboBox_0, 1, 1, 1, 1)

        self.fom_name_label = QLabel(self.verticalLayoutWidget_2)
        self.fom_name_label.setObjectName(u"fom_name_label")

        self.fom_gridLayout.addWidget(self.fom_name_label, 0, 0, 1, 1)

        self.fom_name_lineEdit_0 = QLineEdit(self.verticalLayoutWidget_2)
        self.fom_name_lineEdit_0.setObjectName(u"fom_name_lineEdit_0")
        sizePolicy1 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.fom_name_lineEdit_0.sizePolicy().hasHeightForWidth())
        self.fom_name_lineEdit_0.setSizePolicy(sizePolicy1)

        self.fom_gridLayout.addWidget(self.fom_name_lineEdit_0, 1, 0, 1, 1)

        self.fom_delrow_toolButton = QToolButton(self.verticalLayoutWidget_2)
        self.fom_delrow_toolButton.setObjectName(u"fom_delrow_toolButton")

        self.fom_gridLayout.addWidget(self.fom_delrow_toolButton, 1, 5, 1, 1)

        self.fom_addrow_toolButton = QToolButton(self.verticalLayoutWidget_2)
        self.fom_addrow_toolButton.setObjectName(u"fom_addrow_toolButton")

        self.fom_gridLayout.addWidget(self.fom_addrow_toolButton, 2, 0, 1, 1)

        self.fom_fom_pushButton_0 = QPushButton(self.verticalLayoutWidget_2)
        self.fom_fom_pushButton_0.setObjectName(u"fom_fom_pushButton_0")

        self.fom_gridLayout.addWidget(self.fom_fom_pushButton_0, 1, 2, 1, 1)

        self.fom_grad_pushButton_0 = QPushButton(self.verticalLayoutWidget_2)
        self.fom_grad_pushButton_0.setObjectName(u"fom_grad_pushButton_0")

        self.fom_gridLayout.addWidget(self.fom_grad_pushButton_0, 1, 3, 1, 1)


        self.verticalLayout_2.addLayout(self.fom_gridLayout)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayout_2.addItem(self.verticalSpacer)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_5.addItem(self.horizontalSpacer_4)


        self.verticalLayout_2.addLayout(self.horizontalLayout_5)

        self.tabWidget.addTab(self.fom_tab, "")
        self.device_tab = QWidget()
        self.device_tab.setObjectName(u"device_tab")
        self.horizontalLayoutWidget = QWidget(self.device_tab)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(9, 9, 801, 531))
        self.horizontalLayout_4 = QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.device_treeView = QTreeView(self.horizontalLayoutWidget)
        self.device_treeView.setObjectName(u"device_treeView")
        self.device_treeView.setAlternatingRowColors(True)
        self.device_treeView.header().setCascadingSectionResizes(True)

        self.verticalLayout_3.addWidget(self.device_treeView)


        self.horizontalLayout_4.addLayout(self.verticalLayout_3)

        self.tabWidget.addTab(self.device_tab, "")
        self.opt_tab = QWidget()
        self.opt_tab.setObjectName(u"opt_tab")
        self.verticalLayoutWidget_4 = QWidget(self.opt_tab)
        self.verticalLayoutWidget_4.setObjectName(u"verticalLayoutWidget_4")
        self.verticalLayoutWidget_4.setGeometry(QRect(9, 9, 801, 531))
        self.verticalLayout_5 = QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.label_14 = QLabel(self.verticalLayoutWidget_4)
        self.label_14.setObjectName(u"label_14")
        font = QFont()
        font.setPointSize(20)
        self.label_14.setFont(font)
        self.label_14.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.label_14)

        self.formLayout = QFormLayout()
        self.formLayout.setObjectName(u"formLayout")
        self.label_4 = QLabel(self.verticalLayoutWidget_4)
        self.label_4.setObjectName(u"label_4")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_4)

        self.opt_iter_lineEdit = QLineEdit(self.verticalLayoutWidget_4)
        self.opt_iter_lineEdit.setObjectName(u"opt_iter_lineEdit")

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.opt_iter_lineEdit)

        self.label_6 = QLabel(self.verticalLayoutWidget_4)
        self.label_6.setObjectName(u"label_6")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.label_6)

        self.opt_iter_per_epoch_lineEdit = QLineEdit(self.verticalLayoutWidget_4)
        self.opt_iter_per_epoch_lineEdit.setObjectName(u"opt_iter_per_epoch_lineEdit")

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.opt_iter_per_epoch_lineEdit)

        self.label_5 = QLabel(self.verticalLayoutWidget_4)
        self.label_5.setObjectName(u"label_5")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.label_5)

        self.opt_max_epoch_lineEdit = QLineEdit(self.verticalLayoutWidget_4)
        self.opt_max_epoch_lineEdit.setObjectName(u"opt_max_epoch_lineEdit")

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.opt_max_epoch_lineEdit)

        self.label_7 = QLabel(self.verticalLayoutWidget_4)
        self.label_7.setObjectName(u"label_7")

        self.formLayout.setWidget(4, QFormLayout.LabelRole, self.label_7)

        self.opt_binarfreq_lineEdit = QLineEdit(self.verticalLayoutWidget_4)
        self.opt_binarfreq_lineEdit.setObjectName(u"opt_binarfreq_lineEdit")

        self.formLayout.setWidget(4, QFormLayout.FieldRole, self.opt_binarfreq_lineEdit)

        self.label_11 = QLabel(self.verticalLayoutWidget_4)
        self.label_11.setObjectName(u"label_11")

        self.formLayout.setWidget(5, QFormLayout.LabelRole, self.label_11)

        self.opt_step_size_lineEdit = QLineEdit(self.verticalLayoutWidget_4)
        self.opt_step_size_lineEdit.setObjectName(u"opt_step_size_lineEdit")

        self.formLayout.setWidget(5, QFormLayout.FieldRole, self.opt_step_size_lineEdit)


        self.verticalLayout_5.addLayout(self.formLayout)

        self.line = QFrame(self.verticalLayoutWidget_4)
        self.line.setObjectName(u"line")
        self.line.setLineWidth(6)
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.verticalLayout_5.addWidget(self.line)

        self.label_8 = QLabel(self.verticalLayoutWidget_4)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setFont(font)
        self.label_8.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.label_8)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.label_9 = QLabel(self.verticalLayoutWidget_4)
        self.label_9.setObjectName(u"label_9")

        self.horizontalLayout_7.addWidget(self.label_9)

        self.opt_comboBox = QComboBox(self.verticalLayoutWidget_4)
        self.opt_comboBox.addItem("")
        self.opt_comboBox.setObjectName(u"opt_comboBox")

        self.horizontalLayout_7.addWidget(self.opt_comboBox)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_2)


        self.verticalLayout_5.addLayout(self.horizontalLayout_7)

        self.opt_optimizer_gridLayout = QGridLayout()
        self.opt_optimizer_gridLayout.setObjectName(u"opt_optimizer_gridLayout")
        self.lineEdit_10 = QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_10.setObjectName(u"lineEdit_10")

        self.opt_optimizer_gridLayout.addWidget(self.lineEdit_10, 1, 2, 1, 1)

        self.label_10 = QLabel(self.verticalLayoutWidget_4)
        self.label_10.setObjectName(u"label_10")

        self.opt_optimizer_gridLayout.addWidget(self.label_10, 0, 1, 1, 1)

        self.lineEdit_7 = QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_7.setObjectName(u"lineEdit_7")

        self.opt_optimizer_gridLayout.addWidget(self.lineEdit_7, 0, 2, 1, 1)

        self.lineEdit_9 = QLineEdit(self.verticalLayoutWidget_4)
        self.lineEdit_9.setObjectName(u"lineEdit_9")

        self.opt_optimizer_gridLayout.addWidget(self.lineEdit_9, 3, 2, 1, 1)

        self.label_12 = QLabel(self.verticalLayoutWidget_4)
        self.label_12.setObjectName(u"label_12")

        self.opt_optimizer_gridLayout.addWidget(self.label_12, 3, 1, 1, 1)

        self.label_13 = QLabel(self.verticalLayoutWidget_4)
        self.label_13.setObjectName(u"label_13")

        self.opt_optimizer_gridLayout.addWidget(self.label_13, 1, 1, 1, 1)

        self.horizontalSpacer = QSpacerItem(150, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.opt_optimizer_gridLayout.addItem(self.horizontalSpacer, 0, 0, 1, 1)


        self.verticalLayout_5.addLayout(self.opt_optimizer_gridLayout)

        self.tabWidget.addTab(self.opt_tab, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 819, 22))
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

        self.tabWidget.setCurrentIndex(2)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"vipdopt", None))
        self.actionSave.setText(QCoreApplication.translate("MainWindow", u"Save", None))
#if QT_CONFIG(shortcut)
        self.actionSave.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+S", None))
#endif // QT_CONFIG(shortcut)
        self.actionOpen.setText(QCoreApplication.translate("MainWindow", u"Open...", None))
#if QT_CONFIG(shortcut)
        self.actionOpen.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+O", None))
#endif // QT_CONFIG(shortcut)
        self.actionNew.setText(QCoreApplication.translate("MainWindow", u"New...", None))
#if QT_CONFIG(shortcut)
        self.actionNew.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+N", None))
#endif // QT_CONFIG(shortcut)
        self.actionSave_as.setText(QCoreApplication.translate("MainWindow", u"Save As...", None))
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
        self.actionvipdopt_help.setText(QCoreApplication.translate("MainWindow", u"Vipdopt Help", None))
#if QT_CONFIG(shortcut)
        self.actionvipdopt_help.setShortcut(QCoreApplication.translate("MainWindow", u"Ctrl+H", None))
#endif // QT_CONFIG(shortcut)
        self.actionAbout_Vipdopt.setText(QCoreApplication.translate("MainWindow", u"About Vipdopt", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Load Configuration File:", None))
        self.config_pushButton.setText(QCoreApplication.translate("MainWindow", u"Browse...", None))
        self.pushButton_3.setText(QCoreApplication.translate("MainWindow", u"Save...", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.config_tab), QCoreApplication.translate("MainWindow", u"Configuration", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Load Base Simulation File:", None))
        self.sim_pushButton.setText(QCoreApplication.translate("MainWindow", u"Browse...", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Simulation Configuration:", None))
        ___qtreewidgetitem = self.sim_config_treeWidget.headerItem()
        ___qtreewidgetitem.setText(1, QCoreApplication.translate("MainWindow", u"Enabled Sources", None));
        ___qtreewidgetitem.setText(0, QCoreApplication.translate("MainWindow", u"Simulation", None));

        __sortingEnabled = self.sim_config_treeWidget.isSortingEnabled()
        self.sim_config_treeWidget.setSortingEnabled(False)
        ___qtreewidgetitem1 = self.sim_config_treeWidget.topLevelItem(0)
        ___qtreewidgetitem1.setText(0, QCoreApplication.translate("MainWindow", u"Simulation 1", None));
        ___qtreewidgetitem2 = ___qtreewidgetitem1.child(0)
        ___qtreewidgetitem2.setText(0, QCoreApplication.translate("MainWindow", u"Source 1", None));
        ___qtreewidgetitem3 = ___qtreewidgetitem1.child(1)
        ___qtreewidgetitem3.setText(0, QCoreApplication.translate("MainWindow", u"Source 2", None));
        ___qtreewidgetitem4 = self.sim_config_treeWidget.topLevelItem(1)
        ___qtreewidgetitem4.setText(0, QCoreApplication.translate("MainWindow", u"Simulation 2", None));
        ___qtreewidgetitem5 = ___qtreewidgetitem4.child(0)
        ___qtreewidgetitem5.setText(0, QCoreApplication.translate("MainWindow", u"Source 1", None));
        ___qtreewidgetitem6 = ___qtreewidgetitem4.child(1)
        ___qtreewidgetitem6.setText(0, QCoreApplication.translate("MainWindow", u"Source 2", None));
        self.sim_config_treeWidget.setSortingEnabled(__sortingEnabled)

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.sim_tab), QCoreApplication.translate("MainWindow", u"Simulation", None))
        self.fom_weight_label.setText(QCoreApplication.translate("MainWindow", u"Weight", None))
        self.fom_type_label.setText(QCoreApplication.translate("MainWindow", u"Type", None))
        self.fom_weight_lineEdit_0.setPlaceholderText(QCoreApplication.translate("MainWindow", u"1.0", None))
        self.fom_grad_mon_label.setText(QCoreApplication.translate("MainWindow", u"Gradient Monitors", None))
        self.fom_fom_mon_label.setText(QCoreApplication.translate("MainWindow", u"Figure of Merit Monitors", None))
        self.fom_type_comboBox_0.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Select...", None))
        self.fom_name_label.setText(QCoreApplication.translate("MainWindow", u"Name", None))
        self.fom_name_lineEdit_0.setPlaceholderText(QCoreApplication.translate("MainWindow", u"FoM 0", None))
        self.fom_delrow_toolButton.setText(QCoreApplication.translate("MainWindow", u"Remove Row", None))
        self.fom_addrow_toolButton.setText(QCoreApplication.translate("MainWindow", u"Add Row", None))
        self.fom_fom_pushButton_0.setText(QCoreApplication.translate("MainWindow", u"Choose Monitors...", None))
        self.fom_grad_pushButton_0.setText(QCoreApplication.translate("MainWindow", u"Choose Monitors...", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.fom_tab), QCoreApplication.translate("MainWindow", u"Figure of Merit", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.device_tab), QCoreApplication.translate("MainWindow", u"Device", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Optimization Settings", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Maximum Iterations:", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Iterations per Epoch:", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Maximum Epochs:", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Binarization Frequency:", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Step Size:", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Optimizer Settings", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Optimizer:", None))
        self.opt_comboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"Adam", None))

        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Beta 1:", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Epsilon:", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Beta 2:", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.opt_tab), QCoreApplication.translate("MainWindow", u"Optimization Options", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuEdit.setTitle(QCoreApplication.translate("MainWindow", u"Edit", None))
        self.menuHelp.setTitle(QCoreApplication.translate("MainWindow", u"Help", None))
    # retranslateUi

