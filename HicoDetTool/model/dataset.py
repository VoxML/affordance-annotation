"""
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

from __future__ import print_function, division
from utils.utils import load_hoi_annotation
import os
import json
import torch
import configparser
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.utils import get_iou
from transformers import AutoFeatureExtractor


class HicoDetAllDataset(Dataset):
    def __init__(self, config, train: bool, feature_extractor):
        self.config = config
        self.train = train

        self.image_folder = config.get("HICODET", "hico_images")

        with open(config.get("HICODET", "anno_list_json")) as json_file:
            self.hico_annotation = json.load(json_file)

        with open(self.config["HICODET"]["hoi_list_json"]) as json_file:
            hoi_list = json.load(json_file)
        self.hoic_dict = {}
        for hoi in hoi_list:
            self.hoic_dict[hoi["id"]] = (hoi["object"], hoi["verb"])

        with open(self.config["POSE"]["desc_dataset"]) as json_file:
            self.image_desc = json.load(json_file)

        self.hoi_annotaion = load_hoi_annotation(config.get("HICODET", "hoi_annotation"))

        self.image_list, self.caption_list, self.label_list = self.prepare_data()
        assert len(self.image_list) == len(self.label_list)
        assert len(self.caption_list) == len(self.label_list)

        self.feature_extractor = feature_extractor


    def prepare_data(self):
        image_list = []
        caption_list = []
        label_list = []
        for hico_anno in tqdm(self.hico_annotation):
            if self.train:
                if "test2015" in hico_anno["global_id"]:
                    continue
            else:
                if "train2015" in hico_anno["global_id"]:
                    continue
            image_list.append(hico_anno["image_path_postfix"])

            img_file = hico_anno["image_path_postfix"].split("/")[1]

            for desc in self.image_desc[img_file]:
                if desc['pretrained_model'] == "coco_weights.pt" and (desc['use_beam_search'] == "True") == False:
                    generated_text = desc["generated_text"].strip()
                    generated_text = generated_text[0].upper() + generated_text[1:]
                    if generated_text[-1] != ".":
                        generated_text += "."

                    caption_list.append(generated_text)
                    break

            aff_type_list = []
            for hoi in hico_anno["hois"]:
                hoid_id = hoi["id"]
                obj, verb = self.hoic_dict[hoid_id]
                aff_type_list.append(self.hoi_annotaion[(obj, verb)])
            if "T" in aff_type_list:
                label_list.append(2)
            elif "G" in aff_type_list:
                label_list.append(1)
            else:
                label_list.append(0)

        return image_list, caption_list, label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_folder,
                                self.image_list[idx])

        image = Image.open(img_name).convert('RGB')  # There are some greyscaale images in hicodet
        image = self.feature_extractor(images=image, return_tensors="pt")["pixel_values"][0]

        desc = self.caption_list[idx]

        label = self.label_list[idx]
        label = np.array(label)
        label = label.astype('long')

        sample = {'pixel_values': image, 'caption': desc, 'label': label}

        return sample


class HicoDetAffordanceDataset(Dataset):
    def __init__(self, config, transform, feature_extractor, train: bool):
        self.config = config
        self.train = train

        self.image_folder = config.get("HICODET", "hico_images")

        with open(config.get("HICODET", "anno_list_json")) as json_file:
            self.hico_annotation = json.load(json_file)

        with open(self.config["HICODET"]["hoi_list_json"]) as json_file:
            hoi_list = json.load(json_file)
        self.hoic_dict = {}
        for hoi in hoi_list:
            self.hoic_dict[hoi["id"]] = (hoi["object"], hoi["verb"])

        self.hoi_annotaion = load_hoi_annotation(config.get("HICODET", "hoi_annotation"))

        self.image_list, self.label_list = self.prepare_data()
        assert len(self.image_list) == len(self.label_list)

        self.transform = transform
        self.feature_extractor = feature_extractor

    def prepare_data(self):
        image_list = []
        label_list = []
        for hico_anno in tqdm(self.hico_annotation):
            if self.train:
                if "test2015" in hico_anno["global_id"]:
                    continue
            else:
                if "train2015" in hico_anno["global_id"]:
                    continue
            image_list.append(hico_anno["image_path_postfix"])

            aff_type_list = []
            for hoi in hico_anno["hois"]:
                hoid_id = hoi["id"]
                obj, verb = self.hoic_dict[hoid_id]
                aff_type_list.append(self.hoi_annotaion[(obj, verb)])
            if "T" in aff_type_list:
                label_list.append(2)
            elif "G" in aff_type_list:
                label_list.append(1)
            else:
                label_list.append(0)

        return image_list, label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_folder,
                                self.image_list[idx])

        image = Image.open(img_name).convert('RGB')  # There are some greyscaale images in hicodet

        if self.transform:
            image = self.transform(image)
        elif self.feature_extractor:
            image = self.feature_extractor(images=image, return_tensors="pt")["pixel_values"][0]


        label = self.label_list[idx]
        label = np.array(label)
        label = label.astype('long')

        sample = {'image': image, 'label': label}

        return sample


class HicoDetAffordancePoseDataset(Dataset):
    def __init__(self, config, train: bool, scale: bool, filter_kp_score=False):
        self.config = config
        self.train = train

        self.filter_kp_score = filter_kp_score
        self.scale = scale

        self.image_folder = config.get("HICODET", "hico_images")

        with open(config.get("HICODET", "anno_list_json")) as json_file:
            self.hico_annotation = json.load(json_file)

        with open(self.config["HICODET"]["hoi_list_json"]) as json_file:
            hoi_list = json.load(json_file)
        self.hoic_dict = {}
        for hoi in hoi_list:
            self.hoic_dict[hoi["id"]] = (hoi["object"], hoi["verb"])

        self.hoi_annotaion = load_hoi_annotation(config.get("HICODET", "hoi_annotation"))

        if self.train:
            with open(self.config["POSE"]["train_dataset"]) as json_file:
                pose_anno = json.load(json_file)
        else:
            with open(self.config["POSE"]["test_dataset"]) as json_file:
                pose_anno = json.load(json_file)

        self.pose_anno_dict = {}
        for pose in pose_anno:
            img_id = pose["image_id"]
            if img_id in self.pose_anno_dict:
                self.pose_anno_dict[img_id].append(pose)
            else:
                self.pose_anno_dict[img_id] = [pose]

        self.data_size = -1
        self.pose_list, self.label_list, self.image_list = self.prepare_data()
        assert len(self.pose_list) == len(self.label_list)
        assert len(self.image_list) == len(self.label_list)
        print(len(self.pose_list))

    def prepare_data(self):
        pose_list = []
        label_list = []
        image_list = []
        for hico_anno in tqdm(self.hico_annotation, desc="Prepare Dataset"):
            imgid = hico_anno["image_path_postfix"].split("/")[1]

            if imgid not in self.pose_anno_dict:
                continue

            for pose_anno in self.pose_anno_dict[imgid]:
                keypoints = pose_anno["keypoints"]
                box = pose_anno["box"]
                pose_bbox = [box[0], box[1], box[0]+box[2], box[1]+box[3]]
                aff_type_list = []
                for hoi in hico_anno["hois"]:
                    obj, verb = self.hoic_dict[hoi["id"]]
                    for human_bbox in hoi["human_bboxes"]:
                        iuo = get_iou({"x1": human_bbox[0], "x2": human_bbox[2], "y1": human_bbox[1], "y2": human_bbox[3]}, {"x1": pose_bbox[0], "x2": pose_bbox[2], "y1": pose_bbox[1], "y2": pose_bbox[3]})
                        if iuo > 0.7:
                            aff_type_list.append(self.hoi_annotaion[(obj, verb)])

                if len(aff_type_list) > 0:
                    pose_list.append(keypoints)
                    image_list.append(hico_anno["image_path_postfix"])
                    if "T" in aff_type_list:
                        label_list.append(2)
                    elif "G" in aff_type_list:
                        label_list.append(1)
                    else:
                        label_list.append(0)

        self.data_size = len(pose_list[0])
        return pose_list, label_list, image_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        j_key = []
        kp_score = []

        img_name = os.path.join(self.image_folder, self.image_list[idx])
        size_x, size_y = Image.open(img_name).convert('RGB').size  # There are some greyscaale images in hicodet
        keypoints = self.pose_list[idx]
        for i in range(0, len(keypoints), 3):
            if self.scale:
                j_key.append([keypoints[i] / size_x, keypoints[i + 1] / size_y])
            else:
                j_key.append([keypoints[i], keypoints[i + 1]])

            kp_score.append([keypoints[i + 2]])

        if self.filter_kp_score:
            kp_num = len(kp_score)
            vis_thres = 0.05 if kp_num == 136 else 0.4
            for i, (j, kp) in enumerate(zip(j_key, kp_score)):
                if kp[0] <= vis_thres:
                    j_key[i] = [-1, -1]


        j_key = np.array(j_key)
        j_key = j_key.astype('double')

        kp_score = np.array(kp_score)
        kp_score = kp_score.astype('double')

        label = self.label_list[idx]
        label = np.array(label)
        label = label.astype('long')



        sample = {'j_key': j_key, 'kp_score': kp_score, 'label': label}

        return sample


class HicoDetImgDescDataset(Dataset):
    def __init__(self, config, train: bool, model, beam: bool):
        self.config = config
        self.train = train
        self.model = model  # "conceptual_weights.pt", "coco_weights.pt"
        self.beam = beam

        with open(config.get("HICODET", "anno_list_json")) as json_file:
            self.hico_annotation = json.load(json_file)

        with open(self.config["HICODET"]["hoi_list_json"]) as json_file:
            hoi_list = json.load(json_file)
        self.hoic_dict = {}
        for hoi in hoi_list:
            self.hoic_dict[hoi["id"]] = (hoi["object"], hoi["verb"])

        self.hoi_annotaion = load_hoi_annotation(config.get("HICODET", "hoi_annotation"))


        with open(self.config["POSE"]["desc_dataset"]) as json_file:
            self.image_desc = json.load(json_file)

        self.caption_list, self.label_list = self.prepare_data()
        assert len(self.caption_list) == len(self.label_list)

    def prepare_data(self):
        caption_list = []
        label_list = []
        for hico_anno in tqdm(self.hico_annotation):
            if self.train:
                if "test2015" in hico_anno["global_id"]:
                    continue
            else:
                if "train2015" in hico_anno["global_id"]:
                    continue

            img_file = hico_anno["image_path_postfix"].split("/")[1]

            for desc in self.image_desc[img_file]:
                if desc['pretrained_model'] == self.model and (desc['use_beam_search'] == "True") == self.beam:
                    generated_text = desc["generated_text"].strip()
                    generated_text = generated_text[0].upper() + generated_text[1:]
                    if generated_text[-1] != ".":
                        generated_text += "."

                    caption_list.append(generated_text)
                    break

            aff_type_list = []
            for hoi in hico_anno["hois"]:
                hoid_id = hoi["id"]
                obj, verb = self.hoic_dict[hoid_id]
                aff_type_list.append(self.hoi_annotaion[(obj, verb)])
            if "T" in aff_type_list:
                label_list.append(2)
            elif "G" in aff_type_list:
                label_list.append(1)
            else:
                label_list.append(0)

        return caption_list, label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        desc = self.caption_list[idx]

        label = self.label_list[idx]
        label = np.array(label)
        label = label.astype('long')

        sample = {'caption': desc, 'label': label}
        return sample


if __name__ == "__main__":
    os.chdir("..")
    configp = configparser.ConfigParser()
    configp.read('config.ini')

    extracotre = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    dset = HicoDetAllDataset(configp, False, extracotre)
    data = dset[0]
    exit()
    dset = HicoDetAffordanceDataset(configp, None, None, True)
    lablis = dset.label_list
    print(lablis.count(2))
    print(lablis.count(1))
    print(lablis.count(0))
    exit()

    data = HicoDetImgDescDataset(configp, True, 'coco_weights.pt', True)
    print(data[0])
    exit()

    data = HicoDetAffordancePoseDataset(configp, True, True)
    lablis = data.label_list
    print(lablis.count(2))
    print(lablis.count(1))
    print(lablis.count(0))
    print(data[0])
    exit()
    trans = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    dset = HicoDetAffordanceDataset(configp, trans, None, False)
    lablis = dset.label_list
