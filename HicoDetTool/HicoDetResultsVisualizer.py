import sys
import os
import json
import configparser
import numpy as np
import traceback
import torch
import upt.util.transforms as T
from PIL import Image

from utils.utils import merge_bboxes
from upt.upt import build_detector

from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QWidget, QDesktopWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines


class ImagePredictor:
    def __init__(self, config):
        self.config = config
        self.upt = build_detector(config)
        checkpoint = torch.load(self.config["MODEL"]["upt_model"], map_location='cpu')
        self.upt.load_state_dict(checkpoint['model_state_dict'])
        self.upt.eval()
        self.transforms = self.prepare_transforms()

    def prepare_transforms(self):
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transforms = T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
        return transforms

    def process_image(self, img_path):
        image = Image.open(img_path).convert('RGB')
        img = self.transforms(image, None)[0]
        output = self.upt([img])[0]
        output.pop("attn_maps")

        boxes = output['boxes']
        boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
        objects = output['objects']
        scores = output['scores']
        verbs = output['labels']

        return {"boxes_h": boxes_h, "boxes_o": boxes_o, "objects": objects, "scores": scores, "verbs": verbs,
                "img": image}


class ResultVisualizer:
    def __init__(self, config):
        self.config = config

        self.anno_dict = {}
        with open(self.config["HICODET"]["anno_list_json"]) as json_file:
            self.anno = json.load(json_file)
        for anno in self.anno:
            self.anno_dict[anno["global_id"]] = anno

        self.hoi_list = []
        with open(self.config["HICODET"]["hoi_list_json"]) as json_file:
            self.hoi_list = json.load(json_file)
        self.hoic_dict = {}
        for hoi in self.hoi_list:
            self.hoic_dict[hoi["id"]] = (hoi["object"], hoi["verb"])

        self.tg_dict = {}
        with open(self.config["HICODET"]["hoi_annotation"]) as file:
            for line in file:
                splitline = line.split()
                if len(splitline) > 3:
                    if splitline[3] == "T":
                        self.tg_dict[(splitline[1], splitline[2])] = "T"
                    elif splitline[3] == "G":
                        self.tg_dict[(splitline[1], splitline[2])] = "G"


class App(QWidget):
    def __init__(self, config, visualizer, predictor):
        super().__init__()
        self.title = 'Result Viewer'
        self.left = 100
        self.top = 100
        self.width = 1900
        self.height = 1200
        self.config = config
        self.visualizer = visualizer
        self.predictor = predictor
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setFixedSize(self.width, self.height)

        self.center_widget()

        self.hicodet_label = QLabel(self)
        self.hicodet_label.move(50, 10)
        self.hicodet_label.setFixedSize(self.width-100, 500)
        self.hicodet_label.setAlignment(Qt.AlignCenter)

        self.aff_label = QLabel(self)
        self.aff_label.move(50, 520)
        self.aff_label.setFixedSize(800, 600)
        self.aff_label.setAlignment(Qt.AlignCenter)

        self.pred_label = QLabel(self)
        self.pred_label.move(900, 520)
        self.pred_label.setFixedSize(800, 600)
        self.pred_label.setAlignment(Qt.AlignCenter)

        self.open_button = QPushButton(self)
        self.open_button.setText("Open Image")
        self.open_button.clicked.connect(self.openFileNameDialog)
        self.open_button.move(0, 0)

        #self.hico_label.setFixedSize(self.width, int(self.height / 4))
        #pixmap = QPixmap('D:/Corpora/HICO-DET/hico_20160224_det/images/test2015/HICO_test2015_00000001.jpg')
        #label.setPixmap(pixmap)
        #self.resize(pixmap.width(), pixmap.height())
        self.show()
        #self.processImage("HICO_train2015_00000001.jpg")

    def center_widget(self):
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    def openFileNameDialog(self):
        hico_images = self.config.get("HICODET", "hico_images")
        fileName, test = QFileDialog.getOpenFileName(self, "Open Image", hico_images, "Image Files (*.jpg; *.png)")
        print(fileName)
        print(test)
        if fileName:
            self.processImage(fileName)

    def processImage(self, img_file):
        print("Open:", img_file)
        filename = os.path.basename(img_file).split(".")[0]

        if filename in self.visualizer.anno_dict:
            hico_annotation = self.visualizer.anno_dict[filename]
            print(hico_annotation)
            hicopath = self.prepare_hicodet_img(hico_annotation)
            pixmap = QPixmap(hicopath)
            pixmap = pixmap.scaled(self.hicodet_label.size(), Qt.KeepAspectRatio)
            self.hicodet_label.setPixmap(pixmap)

            # Merge Human and Object BBoxes
            merged_aff = merge_bboxes(hico_annotation, threshold=0.5)
            tgpath = self.prepare_tg_img(merged_aff)
            pixmap = QPixmap(tgpath)
            pixmap = pixmap.scaled(self.aff_label.size(), Qt.KeepAspectRatio)
            self.aff_label.setPixmap(pixmap)

        pred_hoi = self.predictor.process_image(img_file)
        tgpath = self.prepare_pred_img(pred_hoi, filename)
        pixmap = QPixmap(tgpath)
        pixmap = pixmap.scaled(self.pred_label.size(), Qt.KeepAspectRatio)
        self.pred_label.setPixmap(pixmap)

    def prepare_pred_img(self, pred_hoi, filename):
        print(pred_hoi)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.axis('off')
        rescaledimg = T.resize(pred_hoi["img"], None, 800, max_size=1333)[0]
        ax.imshow(rescaledimg)

        inx = np.array([i for i in range(len(pred_hoi["objects"])) if i % 2 == 0])
        boxes_h_filter = pred_hoi["boxes_h"][inx]
        boxes_o_filter = pred_hoi["boxes_o"][inx]
        objects_filter = pred_hoi["objects"][inx]
        scores_reshape = pred_hoi["scores"].reshape(-1, 2)

        for hbox, obox, score, obj in zip(boxes_h_filter, boxes_o_filter, scores_reshape, objects_filter.reshape(-1, 1)):
            max_score, max_idx = torch.max(score, 0)
            if max_score.item() > 0.1:
                rect = patches.Rectangle((hbox[0], hbox[1]), hbox[2] - hbox[0],
                                         hbox[3] - hbox[1], linewidth=1, edgecolor='r',
                                         facecolor='none')
                ax.add_patch(rect)

                rect = patches.Rectangle((obox[0], obox[1]), obox[2] - obox[0],
                                         obox[3] - obox[1], linewidth=1, edgecolor='b',
                                         facecolor='none')
                ax.add_patch(rect)

                h_box_center = (hbox[0] + (hbox[2] - hbox[0]) / 2, hbox[1] + (hbox[3] - hbox[1]) / 2)
                o_box_center = (obox[0] + (obox[2] - obox[0]) / 2, obox[1] + (obox[3] - obox[1]) / 2)
                ax.plot(h_box_center[0], h_box_center[1], o_box_center[0], o_box_center[1], marker="o", color="g")

                if max_idx.item() == 1:
                    line = lines.Line2D([h_box_center[0], o_box_center[0]],
                                        [h_box_center[1], o_box_center[1]],
                                        lw=2, color='b',
                                        axes=ax)
                else:
                    line = lines.Line2D([h_box_center[0], o_box_center[0]],
                                        [h_box_center[1], o_box_center[1]],
                                        lw=2, color='y',
                                        axes=ax)
                ax.add_line(line)

        save_path = os.path.join("tmp", "PRED_" + filename)
        plt.savefig(save_path)
        plt.clf()
        return save_path


    def prepare_tg_img(self, data):
        print(data)
        image_path_postfix = data["image_path_postfix"]
        full_image_path = os.path.join(self.config["HICODET"]["hico_images"], image_path_postfix)
        img = mpimg.imread(full_image_path)
        human_bboxes = data["human_bboxes"]
        object_bboxes = data["object_bboxes"]

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.axis('off')
        ax.imshow(img)
        # Draw Human BBoxes
        for human_bbox in human_bboxes:
            rect = patches.Rectangle((human_bbox[0], human_bbox[1]), human_bbox[2] - human_bbox[0],
                                     human_bbox[3] - human_bbox[1], linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)

        # Draw Object BBoxes
        for object_bbox in object_bboxes:
            rect = patches.Rectangle((object_bbox[0], object_bbox[1]), object_bbox[2] - object_bbox[0],
                                     object_bbox[3] - object_bbox[1], linewidth=1, edgecolor='b',
                                     facecolor='none')
            ax.add_patch(rect)

        # Map Hois to Telic / Gibsonian
        tg_anno_dict = {}
        for hoi in data["hois"]:
            obj_verb = self.visualizer.hoic_dict[hoi["id"]]
            if obj_verb in self.visualizer.tg_dict:
                tg = self.visualizer.tg_dict[obj_verb]
                for connection in hoi["connections"]:
                    if tuple(connection) in tg_anno_dict:
                        if tg_anno_dict[tuple(connection)] != "T":
                            tg_anno_dict[tuple(connection)] = tg
                    else:
                        tg_anno_dict[tuple(connection)] = tg

        # Draw Telic / Gibsonian Connections
        print(tg_anno_dict)
        for connection, tg in tg_anno_dict.items():
            h_box = human_bboxes[connection[0]]
            h_box_center = (h_box[0] + (h_box[2] - h_box[0]) / 2, h_box[1] + (h_box[3] - h_box[1]) / 2)
            o_box = object_bboxes[connection[1]]
            o_box_center = (o_box[0] + (o_box[2] - o_box[0]) / 2, o_box[1] + (o_box[3] - o_box[1]) / 2)
            ax.plot(h_box_center[0], h_box_center[1], o_box_center[0], o_box_center[1], marker="o", color="g")

            if tg == "T":
                line = lines.Line2D([h_box_center[0], o_box_center[0]],
                                    [h_box_center[1], o_box_center[1]],
                                    lw=2, color='b',
                                    axes=ax)
            else:
                line = lines.Line2D([h_box_center[0], o_box_center[0]],
                                    [h_box_center[1], o_box_center[1]],
                                    lw=2, color='y',
                                    axes=ax)
            ax.add_line(line)

        save_path = os.path.join("tmp", "TG_" + data["global_id"])
        plt.savefig(save_path)
        plt.clf()
        return save_path

    def prepare_hicodet_img(self, data):
        image_path_postfix = data["image_path_postfix"]
        full_image_path = os.path.join(self.config["HICODET"]["hico_images"], image_path_postfix)
        img = mpimg.imread(full_image_path)

        fig, axs = plt.subplots(1, len(data["hois"]), figsize=(5 * len(data["hois"]), 5))

        if not isinstance(axs, np.ndarray) and not isinstance(axs, list):
            axs = [axs]

        for ax, hois in zip(axs, data["hois"]):
            connections = hois["connections"]
            human_bboxes = hois["human_bboxes"]
            object_bboxes = hois["object_bboxes"]
            hoi_id = hois["id"]
            invis = hois["invis"]

            ax.axis('off')
            ax.set_title(self.visualizer.hoic_dict[hoi_id])
            ax.imshow(img)

            for human_bbox in human_bboxes:
                rect = patches.Rectangle((human_bbox[0], human_bbox[1]), human_bbox[2] - human_bbox[0],
                                             human_bbox[3] - human_bbox[1], linewidth=1, edgecolor='r',
                                             facecolor='none')
                ax.add_patch(rect)

            for object_bbox in object_bboxes:
                rect = patches.Rectangle((object_bbox[0], object_bbox[1]), object_bbox[2] - object_bbox[0],
                                             object_bbox[3] - object_bbox[1], linewidth=1, edgecolor='b',
                                             facecolor='none')
                ax.add_patch(rect)

            for connection in connections:
                h_box = human_bboxes[connection[0]]
                h_box_center = (h_box[0] + (h_box[2] - h_box[0]) / 2, h_box[1] + (h_box[3] - h_box[1]) / 2)
                o_box = object_bboxes[connection[1]]
                o_box_center = (o_box[0] + (o_box[2] - o_box[0]) / 2, o_box[1] + (o_box[3] - o_box[1]) / 2)
                ax.plot(h_box_center[0], h_box_center[1], o_box_center[0], o_box_center[1], marker="o", color="g")

                line = lines.Line2D([h_box_center[0], o_box_center[0]],
                                        [h_box_center[1], o_box_center[1]],
                                        lw=1, color='g',
                                        axes=ax)
                ax.add_line(line)
        save_path = os.path.join("tmp", "HOI_" + data["global_id"])
        plt.savefig(save_path)
        plt.clf()
        return save_path


def excepthook(exc_type, exc_value, exc_tb):
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print("error catched!:")
    print("error message:\n", tb)
    QApplication.quit()


if __name__ == '__main__':
    configp = configparser.ConfigParser()
    configp.read('config.ini')

    predictor = ImagePredictor(configp)

    visualizer = ResultVisualizer(configp)

    app = QApplication(sys.argv)
    ex = App(configp, visualizer, predictor)

    sys.excepthook = excepthook
    ret = app.exec_()

    sys.exit(ret)

    #sys.exit(app.exec_())
