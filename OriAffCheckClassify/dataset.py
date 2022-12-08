import json
import os.path
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt

class HicoDetDataset(Dataset):
    def __init__(self, anno_file="", img_path="", train=True, filter_obj=None):
        self.anno_file = anno_file
        self.img_path = img_path

        self.map_affordance_lab = {"G": 0, "G pot. T": 1, "T": 2}
        self.obj_lab_map = {"apple": 0, "bicycle": 1, "bottle": 2, "car": 3, "chair": 4, "cup": 5, "dog": 6, "horse": 7,
                            "knife": 8, "person": 9, "umbrella": 10}
        with open(self.anno_file) as json_file:
            anno_data = json.load(json_file)

        self.annotations = []
        for _, annotation in tqdm(anno_data["_via_img_metadata"].items()):
            filename = annotation["filename"]

            if "test2015" in filename and train:
                continue

            if "train2015" in filename and not train:
                continue

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
                    if obj_name not in self.obj_lab_map:
                        continue

                    if filter_obj is None or filter_obj == obj_name:
                        if region["region_attributes"]["affordance"] != "":
                            bbox = [region["shape_attributes"]["x"], region["shape_attributes"]["y"],
                                    region["shape_attributes"]["x"] + region["shape_attributes"]["width"],
                                    region["shape_attributes"]["y"] + region["shape_attributes"]["height"]]
                            front_vec = ori_dict_to_vec(region["region_attributes"]["front"])
                            up_vec = ori_dict_to_vec(region["region_attributes"]["up"])

                            affordance = region["region_attributes"]["affordance"]
                            if affordance == "None":
                                continue

                            affordance_lab = self.map_affordance_lab[affordance]
                            self.annotations.append({"img_file": filename,
                                                     "obj_name": obj_name, "obj_lab": self.obj_lab_map[obj_name],
                                                     "bbox": np.array(bbox), "front": front_vec, "up": up_vec,
                                                     "affordance": affordance, "affordance_lab": affordance_lab})
                else:
                    print("???????????????????")  # Not human nore object. should never happen.
                    exit()

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.annotations[idx]["img_file"])
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        anno = self.annotations[idx]
        #anno["img"] = img
        return anno

    def __len__(self):
        return len(self.annotations)


class HicoDetRelativeDataset(Dataset):
    def __init__(self, anno_file="", img_path="", train=True, filter_obj=None):
        self.anno_file = anno_file
        self.img_path = img_path

        self.map_affordance_lab = {"G": 0, "G pot. T": 1, "T": 2}
        self.obj_lab_map = {"apple": 0, "bicycle": 1, "bottle": 2, "car": 3, "chair": 4, "cup": 5, "dog": 6, "horse": 7,
                            "knife": 8, "umbrella": 9}
        with open(self.anno_file) as json_file:
            anno_data = json.load(json_file)

        self.annotations = []
        for _, annotation in tqdm(anno_data["_via_img_metadata"].items()):
            filename = annotation["filename"]

            if "test2015" in filename and train:
                continue

            if "train2015" in filename and not train:
                continue
            #print("========================")
            #print(annotation)
            #print("------------------------")
            human_ids = []
            for region_id, region in enumerate(annotation["regions"]):
                if "category" not in region["region_attributes"]:
                    continue
                elif region["region_attributes"]["category"] == "human":
                    #print(region)
                    human_ids.append(region_id)
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
                            obj_attributes = annotation["regions"][int(ob)-1]["region_attributes"]
                            #obj_anno = annotation["regions"][int(ob)-1]["region_attributes"]["affordance"]
                            region_id = str(region_id)
                            #print(obj_attributes)
                            if isinstance(obj_attributes["affordance"], str):
                                obj_attributes["affordance"] = {region_id: af}
                            elif region_id not in obj_attributes["affordance"]:
                                obj_attributes["affordance"][region_id] = af
                            elif obj_attributes["affordance"][region_id] == "T":
                                continue
                            elif obj_attributes["affordance"][region_id] == "G" or obj_attributes["affordance"][region_id] == "-":
                                obj_attributes["affordance"][region_id] = af
                            elif obj_attributes["affordance"][region_id] == "G pot. T" and af == "T":
                                obj_attributes["affordance"][region_id] = "T"

            #print("-----------")
            for region_id, region in enumerate(annotation["regions"]):

                if "category" not in region["region_attributes"]:
                    continue
                elif region["region_attributes"]["category"] == "human":
                    continue
                elif region["region_attributes"]["category"] == "object":

                    obj_name = region["region_attributes"]["obj name"].replace(" ", "_")
                    if obj_name not in self.obj_lab_map:
                        continue
                    #if obj_name != "knife":
                    #    continue

                    #print(region)


                    if filter_obj is None or filter_obj == obj_name:
                        if region["region_attributes"]["affordance"] == "":
                            region["region_attributes"]["affordance"] = {}
                        #    for human_id in human_ids:
                        #        region["region_attributes"]["affordance"][str(human_id)] = "None"
                        for human_id, affordance in region["region_attributes"]["affordance"].items():
                            bbox = [region["shape_attributes"]["x"], region["shape_attributes"]["y"],
                                    region["shape_attributes"]["x"] + region["shape_attributes"]["width"],
                                    region["shape_attributes"]["y"] + region["shape_attributes"]["height"]]
                            front_vec = ori_dict_to_vec(region["region_attributes"]["front"])
                            up_vec = ori_dict_to_vec(region["region_attributes"]["up"])
                            #print(annotation)
                            h_up_vec = ori_dict_to_vec(annotation["regions"][int(human_id)]["region_attributes"]["up"])
                            h_front_vec = ori_dict_to_vec(annotation["regions"][int(human_id)]["region_attributes"]["front"])

                            relative_up = np.zeros(3)
                            if np.array_equal(up_vec, relative_up) or np.array_equal(h_up_vec, relative_up) or np.array_equal(up_vec, h_up_vec):
                                relative_up[0] = 1
                            elif np.array_equal(np.negative(up_vec), h_up_vec):
                                relative_up[0] = -1
                            elif np.array_equal(up_vec, h_front_vec):
                                relative_up[1] = 1
                            elif np.array_equal(np.negative(up_vec), h_front_vec):
                                relative_up[1] = -1
                            else:
                                relative_up[2] = 1

                            relative_front = np.zeros(3)
                            if np.array_equal(front_vec, relative_front) or np.array_equal(h_front_vec, relative_front) or np.array_equal(front_vec, h_front_vec):
                                relative_front[0] = 1
                            elif np.array_equal(np.negative(front_vec), h_front_vec):
                                relative_front[0] = -1
                            elif np.array_equal(front_vec, h_up_vec):
                                relative_front[1] = 1
                            elif np.array_equal(np.negative(front_vec), h_up_vec):
                                relative_front[1] = -1
                            else:
                                relative_front[2] = 1


                            affordance_lab = self.map_affordance_lab[affordance]
                            self.annotations.append({"img_file": filename,
                                                     "obj_name": obj_name, "obj_lab": self.obj_lab_map[obj_name],
                                                     "bbox": np.array(bbox), "front": relative_front, "up": relative_up,
                                                     "affordance": affordance, "affordance_lab": affordance_lab})
                            #print(self.annotations[-1])
                else:
                    print("???????????????????")  # Not human nore object. should never happen.
                    exit()
                ##print("......")
            #print("=============================")

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.annotations[idx]["img_file"])
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        anno = self.annotations[idx]
        #anno["img"] = img
        return anno

    def __len__(self):
        return len(self.annotations)


def ori_dict_to_vec(ori_dict):
    vector = np.zeros(3)
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
    train_data = HicoDetRelativeDataset("D:/Corpora/HICO-DET/via234_1200 items_train verified v2.json",
                          "D:/Corpora/HICO-DET/hico_20160224_det/images/merge2015", train=True, filter_obj=None)

    test_data = HicoDetRelativeDataset("D:/Corpora/HICO-DET/via234_1200 items_train verified v2.json",
                          "D:/Corpora/HICO-DET/hico_20160224_det/images/merge2015", train=False, filter_obj=None)
