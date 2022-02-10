import sys
import os
import argparse
import traceback
import json

from processor import ImageProcessor
from utils import colors
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QWidget, QDesktopWidget, QTextBrowser
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QRect, QLine


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

        self.max_img_width = 800
        self.max_img_hight = self.height - 200

        self.hicodet_label = QLabel(self)
        self.hicodet_label.move(10, 10)
        self.hicodet_label.setFixedSize(self.max_img_width, self.max_img_hight)
        self.hicodet_label.setAlignment(Qt.AlignCenter)


        self.results_textbox = QTextBrowser(self)
        self.results_textbox.move(10, self.max_img_hight + 20)
        self.results_textbox.resize(self.width - 20, self.height - (self.max_img_hight + 20 + 10))

        self.obj_textbox = QTextBrowser(self)
        self.obj_textbox.move(900, 10)
        self.obj_textbox.resize(self.width - 910, 490)

        self.hoi_textbox = QTextBrowser(self)
        self.hoi_textbox.move(900, 510)
        self.hoi_textbox.resize(self.width - 910, 490)
        self.hoi_textbox.setAcceptRichText(True)

        self.open_button = QPushButton(self)
        self.open_button.setText("Open Image")
        self.open_button.clicked.connect(self.open_file_name_dialog)
        self.open_button.move(0, 0)

        self.show()

        self.process_image("D:/Corpora/HICO-DET/hico_20160224_det/images/test2015/HICO_test2015_00000005.jpg")

    def process_image(self, img_file):
        pixmap = QPixmap(img_file)
        img_size = pixmap.size()
        scale_factor = min(self.max_img_hight / img_size.height(), self.max_img_width / img_size.width())

        pixmap = pixmap.scaled(self.hicodet_label.size(), Qt.KeepAspectRatio)

        results = self.image_processor.process_image(img_file)
        #results = {'boxes_scores': [0.9997149109840393, 0.9549877047538757, 0.9221760034561157], 'pairing': [[0, 0, 0, 0], [1, 1, 2, 2]], 'pairing_scores': [0.06059708446264267, 0.5121495723724365, 0.08064398914575577, 0.12172546982765198], 'pairing_label': [0, 1, 0, 1], 'boxes_label': [1, 27, 47], 'boxes_label_names': ['person', 'backpack', 'cup'], 'boxes': [[4.344019889831543, 61.697862922138846, 413.08278808593747, 633.2521546669794], [331.02857666015626, 169.29782774390245, 468.6744873046875, 401.90790630863046], [8.274765014648438, 383.0560726430582, 73.61431274414062, 444.2026999296436]], 'boxes_orientation': [{'front': [0.913848340511322, -0.09131860733032227, 0.39565420150756836], 'up': [0.08353635668754578, 0.9958215355873108, 0.0368945337831974], 'left': [0.3973701000213623, 0.000664496619720012, -0.9176582098007202]}, {'front': [0.9801536202430725, -0.11696866154670715, 0.16005392372608185], 'up': [0.11525282263755798, 0.9931349158287048, 0.019994506612420082], 'left': [0.1612938791513443, 0.0011510220356285572, -0.9869058132171631]}, {'front': [0.060182277113199234, -0.3483457863330841, 0.9354321956634521], 'up': [0.01510847732424736, 0.9373413324356079, 0.3480847179889679], 'left': [0.9980731010437012, 0.006815575994551182, -0.06167431175708771]}]}

        result_text = ""
        for key in sorted(results.keys()):
            # result_text += key + ": " + str(results[key]) + "\n"
            result_text += f"{key} : {results[key]}\n"
        self.results_textbox.setText(result_text)

        obj_text = ""
        for bbox_id, (bbox, bbox_label, bbox_label_name, bbox_score, bbox_ori) in \
                enumerate(zip(results["boxes"], results["boxes_label"], results["boxes_label_names"], results["boxes_scores"], results["boxes_orientation"])):
            obj_text += f"{bbox_id}_{bbox_label_name}: {bbox_score}\n" \
                        f"\tup: {bbox_ori['up']}\n" \
                        f"\tfront: {bbox_ori['front']}\n" \
                        f"\tleft: {bbox_ori['left']}\n\n"
        self.obj_textbox.setText(obj_text)

        painter = QPainter(pixmap)

        for bbox_id, (bbox, bbox_label, bbox_label_name) in \
                enumerate(zip(results["boxes"], results["boxes_label"], results["boxes_label_names"])):

            painter.setPen(QPen(QColor(colors[bbox_label]), 2, Qt.SolidLine))
            rect = QRect(int(bbox[0]*scale_factor), int(bbox[1]*scale_factor),
                         int(bbox[2]*scale_factor) - int(bbox[0]*scale_factor),
                         int(bbox[3]*scale_factor) - int(bbox[1]*scale_factor))
            painter.drawRect(rect)
            painter.drawText(rect, Qt.AlignLeft, str(bbox_id) + "_" + bbox_label_name)

        hoi_text = ""
        for hbox_id, obox_id, pair_label, label_score in \
                zip(results["pairing"][0], results["pairing"][1], results["pairing_label"], results["pairing_scores"]):
            label_score_str = str(label_score) if label_score < self.config.hoi_score_thresh else "===" + str(label_score) + "==="
            if pair_label == 0:
                hoi_text += f"{hbox_id}_{results['boxes_label_names'][hbox_id]} - {obox_id}_{results['boxes_label_names'][obox_id]}\n" \
                            f"\tGibsonian: {label_score_str}\n"
            elif pair_label == 1:
                hoi_text += f"\tTelic: {label_score_str}\n\n"
            else:
                hoi_text += "====================\n"
        self.hoi_textbox.setText(hoi_text)

        for hbox_id, obox_id, pair_label, label_score in \
                zip(results["pairing"][0], results["pairing"][1], results["pairing_label"], results["pairing_scores"]):
            hbox = results["boxes"][hbox_id]
            obox = results["boxes"][obox_id]
            if label_score > 0.2:
                h_box_center = (hbox[0] + (hbox[2] - hbox[0]) / 2, hbox[1] + (hbox[3] - hbox[1]) / 2)
                o_box_center = (obox[0] + (obox[2] - obox[0]) / 2, obox[1] + (obox[3] - obox[1]) / 2)
                if pair_label == 0:
                    painter.setPen(QPen(Qt.blue, 2, Qt.SolidLine))
                elif pair_label == 1:
                    painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
                line = QLine(int(h_box_center[0] * scale_factor),
                             int(h_box_center[1] * scale_factor),
                             int(o_box_center[0] * scale_factor),
                             int(o_box_center[1] * scale_factor))
                line.center()
                painter.drawLine(line)
                painter.drawText(line.center().x(), line.center().y()-10, str(hbox_id) + "_" + str(obox_id))

        painter.end()
        self.hicodet_label.setPixmap(pixmap)


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

    parser.add_argument('--box_score_thresh', default=0.9, type=int)
    parser.add_argument('--hoi_score_thresh', default=0.2, type=int)

    parser.add_argument('--hoi_model', default="data/models/robust-sweep-8_ckpt_41940_20.pt", type=str)
    parser.add_argument('--pose_model', default="data/models/pose-model.pth", type=str)
    parsed_args = parser.parse_args()


    app = QApplication(sys.argv)

    ex = App(parsed_args)

    sys.excepthook = excepthook
    ret = app.exec_()

    sys.exit(ret)
