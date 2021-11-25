"""
https://huggingface.co/transformers/custom_datasets.html
"""
import json
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from transformers import ViTFeatureExtractor


class HicoDetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Not Ideal. But fits better with the HicoDet Structure ....
        item = {"pixel_values": self.encodings[idx]["pixel_values"]}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_hoi_dict(config):
    with open(config.get("HICODET", "hoi_list_json")) as json_file:
        hoi_list = json.load(json_file)
    hoic_dict = {}
    for hoi in hoi_list:
        hoic_dict[hoi["id"]] = (hoi["object"], hoi["verb"])
    return hoic_dict


def load_test_data(config):
    ENTITY = "boat"
    AFFORDANCE = {"board": "G", "drive": "T", "exit": "G", "inspect": "-", "jump": "-", "launch": "T", "repair": "-",
                  "ride": "T", "row": "T", "sail": "T", "sit-on": "G", "stand_on": "G", "tie": "-", "wash": "-",
                  "no_interaction": "-"}
    LABELMAP = {"-": 0, "G": 1, "T": 2}

    # Load HicoDet Data
    annofile = config.get("HICODET", "anno_list_json")
    with open(annofile) as json_file:
        json_data = json.load(json_file)
    hoic_dict = load_hoi_dict(config)

    # Filter HicoDet Data
    filtered_data = []
    filtered_data_label = []
    for image in json_data:
        for hois in image["hois"]:
            hoiid = hois["id"]
            nomen, verb = hoic_dict[hoiid]
            if nomen == ENTITY and verb in AFFORDANCE.keys() and image["image_size"][2] == 3:  # Filter b/w images
                filtered_data.append(image)
                filtered_data_label.append(LABELMAP[AFFORDANCE[verb]])

    # Preprocess images
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    for image in filtered_data:
        image["img"] = Image.open(os.path.join(config.get("HICODET", "hico_images"), image["image_path_postfix"]))
        image["pixel_values"] = feature_extractor(image["img"])["pixel_values"][0]

    return filtered_data, filtered_data_label
