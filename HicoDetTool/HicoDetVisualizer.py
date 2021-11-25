import configparser
import json
import os.path
import numpy as np
import torch

from utils.alphapose import vis_frame,vis_frame_fast
import cv2

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines


class Visualizer:
    def __init__(self, config):
        self.config = config

        self.anno = []
        with open(self.config["HICODET"]["anno_list_json"]) as json_file:
            self.anno = json.load(json_file)

        self.hoi_list = []
        with open(self.config["HICODET"]["hoi_list_json"]) as json_file:
            self.hoi_list = json.load(json_file)
        self.hoic_dict = {}
        for hoi in self.hoi_list:
            self.hoic_dict[hoi["id"]] = (hoi["object"], hoi["verb"])

        self.poselist = None
        if self.config["POSE"]["train_dataset"] is not None:
            if os.path.isfile(self.config["POSE"]["train_dataset"]):
                with open(self.config["POSE"]["train_dataset"]) as json_file:
                    self.poselist = json.load(json_file)
            else:
                print("Could not find Pose Data")

            if os.path.isfile(self.config["POSE"]["test_dataset"]):
                with open(self.config["POSE"]["test_dataset"]) as json_file:
                    self.poselist.extend(json.load(json_file))
            else:
                print("Could not find Pose Data")


    def show_image(self, id: int):
        data = self.anno[id]
        image_path_postfix = data["image_path_postfix"]
        full_image_path = os.path.join(self.config["HICODET"]["hico_images"], image_path_postfix)
        img = mpimg.imread(full_image_path)

        if self.poselist is not None:
            img = self.add_pose(img, data["image_path_postfix"])

        fig, axs = plt.subplots(1, len(data["hois"]), figsize=(5*len(data["hois"]), 5))

        if not isinstance(axs, np.ndarray) and not isinstance(axs, list):
            axs = [axs]

        for ax, hois in zip(axs, data["hois"]):
            connections = hois["connections"]
            human_bboxes = hois["human_bboxes"]
            object_bboxes = hois["object_bboxes"]
            hoi_id = hois["id"]
            invis = hois["invis"]

            ax.axis('off')
            ax.set_title(self.hoic_dict[hoi_id])
            ax.imshow(img)

            for human_bbox in human_bboxes:
                rect = patches.Rectangle((human_bbox[0], human_bbox[1]), human_bbox[2] - human_bbox[0], human_bbox[3] - human_bbox[1], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            for object_bbox in object_bboxes:
                rect = patches.Rectangle((object_bbox[0], object_bbox[1]), object_bbox[2] - object_bbox[0], object_bbox[3] - object_bbox[1], linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect)

            for connection in connections:
                h_box = human_bboxes[connection[0]]
                h_box_center = (h_box[0] + (h_box[2] - h_box[0]) / 2, h_box[1] + (h_box[3] - h_box[1]) / 2)
                o_box = object_bboxes[connection[1]]
                o_box_center = (o_box[0] + (o_box[2] - o_box[0]) / 2, o_box[1] + (o_box[3] - o_box[1]) / 2)
                ax.plot(h_box_center[0], h_box_center[1], o_box_center[0], o_box_center[1], marker= "o", color="g")

                line = lines.Line2D([h_box_center[0], o_box_center[0]],
                                    [h_box_center[1], o_box_center[1]],
                                    lw=1, color='g',
                                    axes=ax)
                ax.add_line(line)

        fig.suptitle(data["global_id"], fontsize=16)
        plt.show()

    def add_pose(self, img, image_path_postfix):
        imgid = image_path_postfix.split("/")[1]

        resultjson = []
        for pose_anno in self.poselist:
            if pose_anno["image_id"] == imgid:
                keypoints = pose_anno["keypoints"]
                j_key = []
                kp_score = []

                for i in range(0, len(keypoints), 3):
                    j_key.append([keypoints[i], keypoints[i+1]])
                    kp_score.append([keypoints[i+2]])
                resultjson.append({"keypoints": torch.tensor(j_key), "kp_score": torch.tensor(kp_score)}) #, "box": d["box"]})
        # image channel RGB->BGR https://github.com/MVIG-SJTU/AlphaPose/blob/bcfbc997526bcac464d116356ac2efea9483ff68/scripts/demo_api.py#L192
        # Not needed here. Bet to avoid confusion, if someone saves the image beforehand.
        img = np.array(img, dtype=np.uint8)[:, :, ::-1]
        img = vis_frame(img, {"result": resultjson})  # visulize the pose result
        img = np.array(img, dtype=np.uint8)[:, :, ::-1]
        return img



if __name__ == '__main__':
    configp = configparser.ConfigParser()
    configp.read('config.ini')

    visualizer = Visualizer(configp)
    visualizer.show_image(0)
    exit()
    for id, datain in enumerate(visualizer.anno[0:]):
        for data in datain["hois"]:
            if(len(data["connections"]) > 1):
                visualizer.show_image(id)
                break

