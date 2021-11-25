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
#from skimage import io
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms


class HicoDetAffordanceDataset(Dataset):
    def __init__(self, config, transform, train: bool):
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

        self.image_list, self.label_list, self.hoi_list = self.prepare_data()

        self.transform = transform

    def prepare_data(self):
        image_list = []
        label_list = []
        hoi_list = []
        for hico_anno in self.hico_annotation:
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
                hoi_list.append((obj, verb))
            if "T" in aff_type_list:
                label_list.append(2)
            elif "G" in aff_type_list:
                label_list.append(1)
            else:
                label_list.append(0)

        return image_list, label_list, hoi_list

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

        label = self.label_list[idx]
        label = np.array(label)
        label = label.astype('long')

        hoi = self.hoi_list[idx]
        sample = {'image': image, 'label': label, 'hoi_list': hoi}

        return sample


if __name__ == "__main__":
    os.chdir("..")
    configp = configparser.ConfigParser()
    configp.read('config.ini')

    trans = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    dset = HicoDetAffordanceDataset(configp, trans, True)
    lablis = dset.label_list
    print(len(lablis))
    print(lablis.count(2))
    print(lablis.count(1))
    print(lablis.count(0))
    print(dset[0])