import json
import configparser
import os.path

from utils.utils import get_iou

if __name__ == "__main__":
    #docs = ["fast-pose-duc-train-1.json", "fast-pose-duc-train-2.json", "fast-pose-duc-train-3.json", "fast-pose-duc-train-4.json"]
    # outfile = "fast-pose-duc-train.json"
    #docs = ["halpe136-train-1_1.json", "halpe136-train-1_2.json", "halpe136-train-1_3.json",
    #        "halpe136-train-1_4.json", "halpe136-train-1_5.json", "halpe136-train-1_6.json", "halpe136-train-2.json", "halpe136-train-3.json", "halpe136-train-4.json"]

    docs = ["halpe136-train.json", "halpe26-train.json", "fast-pose-duc-train.json"]
    docs = ["halpe136-test.json", "halpe26-test.json", "fast-pose-duc-test.json"]
    outfile = "halpe136-train.json"


    #merged_json = []
    for doc in docs:
        path = os.path.join("data", "poses", doc)
        print("....")
        with open(path, "r") as json_file:
            #print(len(merged_json))
            data = json.load(json_file)
            print(len(data))
            #merged_json.extend(data)
            #print(len(merged_json))

    #with open(os.path.join("data", outfile), 'w') as f:
    #    json.dump(merged_json, f)
