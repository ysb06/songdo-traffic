# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/sbyim/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/songdo_qgis/songdo_qgis_dialog_base.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SongdoQGISDialogBase(object):
    def setupUi(self, SongdoQGISDialogBase: QtWidgets.QDialog):
        SongdoQGISDialogBase.setObjectName("SongdoQGISDialogBase")
        SongdoQGISDialogBase.resize(400, 300)
        self.verticalLayout = QtWidgets.QVBoxLayout(SongdoQGISDialogBase)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget = QtWidgets.QWidget(SongdoQGISDialogBase)
        self.widget.setObjectName("widget")
        self.formLayout = QtWidgets.QFormLayout(self.widget)
        self.formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.formLayout.setHorizontalSpacing(20)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label)
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.label_2)
        self.pathSelectorLabel = QtWidgets.QLabel(self.widget)
        self.pathSelectorLabel.setObjectName("pathSelectorLabel")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.pathSelectorLabel)
        self.nodeSelector = QtWidgets.QComboBox(self.widget)
        self.nodeSelector.setObjectName("nodeSelector")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.nodeSelector)
        self.outputSelector = QtWidgets.QComboBox(self.widget)
        self.outputSelector.setObjectName("outputSelector")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.outputSelector)
        self.pathSelector = QtWidgets.QComboBox(self.widget)
        self.pathSelector.setObjectName("pathSelector")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.pathSelector)
        self.verticalLayout.addWidget(self.widget)
        self.button_box = QtWidgets.QDialogButtonBox(SongdoQGISDialogBase)
        self.button_box.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.button_box.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.button_box.setObjectName("button_box")
        self.verticalLayout.addWidget(self.button_box)

        self.retranslateUi(SongdoQGISDialogBase)
        self.button_box.accepted.connect(SongdoQGISDialogBase.accept) # type: ignore
        self.button_box.rejected.connect(SongdoQGISDialogBase.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(SongdoQGISDialogBase)

    def retranslateUi(self, SongdoQGISDialogBase: QtWidgets.QDialog):
        _translate = QtCore.QCoreApplication.translate
        SongdoQGISDialogBase.setWindowTitle(_translate("SongdoQGISDialogBase", "Songdo QGIS"))
        self.label.setText(_translate("SongdoQGISDialogBase", "Nodes Layer"))
        self.label_2.setText(_translate("SongdoQGISDialogBase", "Output Layer (Links)"))
        self.pathSelectorLabel.setText(_translate("SongdoQGISDialogBase", "Path Data Layer"))