import json
import configparser
from utils.utils import merge_bboxes


def print_annotation(annotation):
    print(annotation["global_id"])
    for hois in annotation["hois"]:
        print("---")
        print(hois["id"])
        print(hois["human_bboxes"])
        print(hois["object_bboxes"])


if __name__ == "__main__":
    configp = configparser.ConfigParser()
    configp.read('config.ini')
    with open(configp["HICODET"]["anno_list_json"]) as json_file:
        data = json.load(json_file)
    # print_annotation(data[0])
    # print_annotation(data[0])
    print(data[33])
    merged_anno = merge_bboxes(data[37], configp.getfloat("PROCESSING", "bbox_merge_threshold"))
    print(merged_anno)


