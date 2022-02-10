import sys
import os
import argparse
import traceback

from processor import ImageProcessor

from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QWidget, QDesktopWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


class App(QWidget):
    def __init__(self, config):
        super().__init__()
        self.title = 'Result Viewer'
        self.left = 100
        self.top = 100
        self.width = 1900
        self.height = 1200
        self.config = config
        self.image_processor = ImageProcessor(config)

        self.init_ui()

    def init_ui(self):
        print("Init UI")
        self.setWindowTitle(self.title)
        self.setFixedSize(self.width, self.height)

        self.center_widget()

        self.hicodet_label = QLabel(self)
        self.hicodet_label.move(50, 10)
        self.hicodet_label.setFixedSize(self.width-100, 500)
        self.hicodet_label.setAlignment(Qt.AlignCenter)

        self.open_button = QPushButton(self)
        self.open_button.setText("Open Image")
        self.open_button.clicked.connect(self.open_file_name_dialog)
        self.open_button.move(0, 0)

        self.show()

    def process_image(self, img_file):
        pixmap = QPixmap(img_file)
        pixmap = pixmap.scaled(self.hicodet_label.size(), Qt.KeepAspectRatio)
        self.hicodet_label.setPixmap(pixmap)

        pred_results = self.image_processor.process_image(img_file)
        print(pred_results)
        results = {'boxes_scores': [0.9997149109840393, 0.9549877047538757, 0.9221760034561157, 0.14659665524959564], 'boxes_label': [27, 47, 62], 'boxes_label_names': ['backpack', 'cup', 'chair'], 'pairing': [[0, 0, 0, 0, 0, 0], [1, 1, 2, 2, 3, 3]], 'pairing_scores': [0.060354121029376984, 0.4823215901851654, 0.07706623524427414, 0.10976481437683105, 9.342850535176694e-05, 7.607042789459229e-05], 'pairing_label': [0, 1, 0, 1, 0, 1], 'boxes': [[4.344019889831543, 61.697862922138846, 413.08278808593747, 633.2521546669794], [331.02857666015626, 169.29782774390245, 468.6744873046875, 401.90790630863046], [8.274765014648438, 383.0560726430582, 73.61431274414062, 444.2026999296436], [387.2721313476562, 554.8985694183865, 408.4489013671875, 640.0514481707318]], 'boxes_orientation': [{'front': [0.9138361215591431, -0.09134881943464279, 0.39567533135414124], 'up': [0.08356207609176636, 0.9958187341690063, 0.03691112995147705], 'left': [0.3973926901817322, 0.0006672710878774524, -0.917648434638977]}, {'front': [0.9801550507545471, -0.11695320159196854, 0.16005627810955048], 'up': [0.11523772031068802, 0.9931367635726929, 0.01999105140566826], 'left': [0.1612957864999771, 0.0011498109670355916, -0.9869054555892944]}, {'front': [0.06017287075519562, -0.34833747148513794, 0.9354358315467834], 'up': [0.015106739476323128, 0.9373444318771362, 0.34807640314102173], 'left': [0.9980736374855042, 0.00681337108835578, -0.061664946377277374]}, {'front': [0.054753027856349945, -0.01628202572464943, 0.998367190361023], 'up': [0.0003922013856936246, 0.99986732006073, 0.01628498174250126], 'left': [0.998499870300293, 0.0005000910605303943, -0.05475214868783951]}]}


    def open_file_name_dialog(self):
        fileName, test = QFileDialog.getOpenFileName(self, "Open Image", "D:/Corpora/HICO-DET/hico_20160224_det/images/test2015", "Image Files (*.jpg; *.png)")
        if fileName:
            self.process_image(fileName)

    def center_widget(self):
        qt_rectangle = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        qt_rectangle.moveCenter(center_point)
        self.move(qt_rectangle.topLeft())


def excepthook(exc_type, exc_value, exc_tb):
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print("error catched!:")
    print("error message:\n", tb)
    QApplication.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default=0, type=int)

    parser.add_argument('--box_score_thresh', default=0.8, type=int)

    parser.add_argument('--hoi_model', default="data/models/robust-sweep-8_ckpt_41940_20.pt", type=str)
    parser.add_argument('--pose_model', default="data/models/pose-model.pth", type=str)
    parsed_args = parser.parse_args()


    app = QApplication(sys.argv)

    ex = App(parsed_args)

    sys.excepthook = excepthook
    ret = app.exec_()

    sys.exit(ret)
