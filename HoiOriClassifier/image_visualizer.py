import sys
import argparse
import traceback


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

        self.max_img_width = 800
        self.max_img_hight = self.height - 200

        self.config = config
        self.image_processor = ImageProcessor(config)

        self.initUI()

    def initUI(self):
        print("Init UI")
        # Todo: make every size dependent on width and height
        self.setWindowTitle(self.title)
        self.setFixedSize(self.width, self.height)
        self.center_widget()

        # Image Window
        self.hicodet_label = QLabel(self)
        self.hicodet_label.move(10, 10)
        self.hicodet_label.setFixedSize(self.max_img_width, self.max_img_hight)
        self.hicodet_label.setAlignment(Qt.AlignCenter)

        # Bottom Box for JSON Output
        self.results_textbox = QTextBrowser(self)
        self.results_textbox.move(10, self.max_img_hight + 20)
        self.results_textbox.resize(self.width - 20, self.height - (self.max_img_hight + 20 + 10))

        # Top Right Box for Obj Results
        self.obj_textbox = QTextBrowser(self)
        self.obj_textbox.move(900, 10)
        self.obj_textbox.resize(self.width - 910, 490)

        # Bottom Right Box for HOI Results
        self.hoi_textbox = QTextBrowser(self)
        self.hoi_textbox.move(900, 510)
        self.hoi_textbox.resize(self.width - 910, 490)
        #self.hoi_textbox.setAcceptRichText(True)

        # Top Left Buttom
        self.open_button = QPushButton(self)
        self.open_button.setText("Open Image")
        self.open_button.clicked.connect(self.open_file_name_dialog)
        self.open_button.move(0, 0)

        self.show()

        #self.process_image("D:/Corpora/HICO-DET/hico_20160224_det/images/test2015/HICO_test2015_00000005.jpg")

    def update_result_textbox(self, results):
        result_text = ""
        for key in sorted(results.keys()):
            result_text += f"{key} : {results[key]}\n"
        self.results_textbox.setText(result_text)

    def update_obj_textbox(self, results):
        obj_text = ""
        for bbox_id, (bbox, bbox_label, bbox_label_name, bbox_score, bbox_ori) in \
                enumerate(zip(results["boxes"], results["boxes_label"], results["boxes_label_names"], results["boxes_scores"], results["boxes_orientation"])):
            rotation = bbox_ori["rotation"]
            obj_text += f"{bbox_id}_{bbox_label_name}: {bbox_score}\n" \
                        f"\tup: {bbox_ori['up']}\n" \
                        f"\tfront: {bbox_ori['front']}\n" \
                        f"\tleft: {bbox_ori['left']}\n" \
                        f"\t\tazi: {rotation['azi']},ele: {rotation['ele']}, inp: {rotation['inp']}\n\n"
        self.obj_textbox.setText(obj_text)

    def update_hoi_textbox(self, results):
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

    def process_image(self, img_file):
        """
        Args:
            img_file: Path to Image
        Returns: None
        """
        pixmap = QPixmap(img_file)
        img_size = pixmap.size()

        # For rescaling the results between the actual image and the displayed one.
        scale_factor = min(self.max_img_hight / img_size.height(), self.max_img_width / img_size.width())
        pixmap = pixmap.scaled(self.hicodet_label.size(), Qt.KeepAspectRatio)

        # Process Image
        results = self.image_processor.process_image(img_file)
        if len(results) == 0:
            self.hicodet_label.setPixmap(pixmap)
            self.results_textbox.setText("No objects could be detected!")
            self.obj_textbox.setText("")
            self.hoi_textbox.setText("")
            return

        # Update Textboxes
        self.update_result_textbox(results)
        self.update_obj_textbox(results)
        self.update_hoi_textbox(results)

        # Add BBoxes to Image
        painter = QPainter(pixmap)
        for bbox_id, (bbox, bbox_label, bbox_label_name) in \
                enumerate(zip(results["boxes"], results["boxes_label"], results["boxes_label_names"])):

            painter.setPen(QPen(QColor(colors[bbox_label]), 2, Qt.SolidLine))
            rect = QRect(int(bbox[0]*scale_factor), int(bbox[1]*scale_factor),
                         int(bbox[2]*scale_factor) - int(bbox[0]*scale_factor),
                         int(bbox[3]*scale_factor) - int(bbox[1]*scale_factor))
            painter.drawRect(rect)
            painter.drawText(rect, Qt.AlignLeft, str(bbox_id) + "_" + bbox_label_name)

        # Add HOI Links in Image
        for hbox_id, obox_id, pair_label, label_score in \
                zip(results["pairing"][0], results["pairing"][1], results["pairing_label"], results["pairing_scores"]):
            hbox = results["boxes"][hbox_id]
            obox = results["boxes"][obox_id]
            if label_score > self.config.hoi_score_thresh:
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
                painter.drawLine(line)
                painter.drawText(line.center().x(), line.center().y()-10, str(hbox_id) + "_" + str(obox_id))
        painter.end()

        # Display Image
        self.hicodet_label.setPixmap(pixmap)

    def open_file_name_dialog(self):
        """
        Dialoge for opening and processing image
        Returns: None
        """
        file_name, test = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.jpg; *.png)")
        if file_name:
            self.process_image(file_name)

    def center_widget(self):
        """
        Center App
        """
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
