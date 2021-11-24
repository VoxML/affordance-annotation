import configparser
import json
import os.path
import numpy as np

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

    def show_image(self, id: int):
        data = self.anno[id]
        print(data)
        image_path_postfix = data["image_path_postfix"]
        full_image_path = os.path.join(self.config["HICODET"]["hico_images"], image_path_postfix)
        img = mpimg.imread(full_image_path)

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


if __name__ == '__main__':
    configp = configparser.ConfigParser()
    configp.read('config.ini')

    visualizer = Visualizer(configp)
    visualizer.show_image(33)

    for id, datain in enumerate(visualizer.anno[0:]):
        for data in datain["hois"]:
            if(len(data["connections"]) > 1):
                visualizer.show_image(id)
                break

