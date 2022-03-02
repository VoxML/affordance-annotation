import json

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class HicoDetDataset(Dataset):
    def __init__(self, anno_file="", img_path="", train=True):
        self.anno_file = anno_file
        self.img_path = img_path

        self.data = []

        with open(self.anno_file) as json_file:
            anno_data = json.load(json_file)

        self.annotations = []
        for _, annotation in tqdm(anno_data["_via_img_metadata"].items()):
            filename = annotation["filename"]

            for region_id, region in enumerate(annotation["regions"]):

                # Check if Person or Object
                if "category" not in region["region_attributes"]:
                    continue
                elif region["region_attributes"]["category"] == "human":
                    anno_actions = region["region_attributes"]["action"].split(",")
                    anno_affordances = region["region_attributes"]["affordance"].split(",")
                    anno_objs = region["region_attributes"]["obj id"].split(",")
                    if len(anno_actions) == len(anno_affordances) == len(anno_objs):
                        for ac, af, ob in zip(anno_actions, anno_affordances, anno_objs):
                            if af.isnumeric():  # min one annotation had it flipped...
                                af, ob = ob, af
                            af = af.strip()
                            if af not in ["T", "G", "G pot. T"] or not ob.isnumeric():
                                continue
                            obj_anno = annotation["regions"][int(ob)-1]["region_attributes"]
                            if obj_anno["affordance"] == "":
                                obj_anno["affordance"] = af
                            elif obj_anno["affordance"] == "T":
                                continue
                            elif obj_anno["affordance"] == "G" or obj_anno["affordance"] == "-":
                                obj_anno["affordance"] = af
                            elif obj_anno["affordance"] == "G pot. T" and af == "T":
                                obj_anno["affordance"] = "T"


            for region_id, region in enumerate(annotation["regions"]):
                if "category" not in region["region_attributes"]:
                    continue
                elif region["region_attributes"]["category"] == "human":
                    continue
                elif region["region_attributes"]["category"] == "object":
                    obj_name = region["region_attributes"]["obj name"].replace(" ", "_")
                    if region["region_attributes"]["affordance"] != "":
                        bbox = [region["shape_attributes"]["x"], region["shape_attributes"]["y"],
                                region["shape_attributes"]["x"] + region["shape_attributes"]["width"],
                                region["shape_attributes"]["y"] + region["shape_attributes"]["height"]]
                        front_vec = ori_dict_to_vec(region["region_attributes"]["front"])
                        up_vec = ori_dict_to_vec(region["region_attributes"]["up"])
                        rot_mat = self.compute_rotation(front_vec, up_vec)
                        print(front_vec)
                        print(up_vec)
                        print(rot_mat)
                        print("==========")
                        affordance = region["region_attributes"]["affordance"]
                        self.annotations.append({"img_file": filename, "obj_name": obj_name,
                                                 "bbox": bbox, "front": front_vec, "up": up_vec, "affordance": affordance})
                else:
                    print("???????????????????")  # Not human nore object. should never happen.
                    exit()
        #print(self.annotations)
        #print(len(self.annotations))

    def __getitem__(self, idx):
        # ....
        return None

    def __len__(self):
        return len(self.data)

    def compute_rotation(self, front, up):
        rot_mat = np.zeros((3, 3))
        rot_mat[1] = up


        if sum(front) == 0:
            rot_mat[2] = [0, 0, 1]
        else:
            rot_mat[2] = front


        rot_mat[0] = left

        return rot_mat


def ori_dict_to_vec(ori_dict):
    vector = [0, 0, 0]
    keys = ori_dict.keys()
    if "n/a" in keys or len(keys) == 0:
        return vector
    elif "+x" in keys:
        vector[0] = 1
    elif "-x" in keys:
        vector[0] = -1
    elif "+y" in keys:
        vector[1] = 1
    elif "-y" in keys:
        vector[1] = -1
    elif "+z" in keys:
        vector[2] = 1
    elif "-z" in keys:
        vector[2] = -1
    else:
        print(ori_dict)
        print("!!!!!!!!!!!!!!!!!")
    return vector


def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


if __name__ == "__main__":
    data = HicoDetDataset("D:/Corpora/HICO-DET/via234_1200 items_train verified v2.json",
                          "D:/Corpora/HICO-DET/hico_20160224_det/images/merge2015")