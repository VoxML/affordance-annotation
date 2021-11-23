import json
import configparser


configp = configparser.ConfigParser()
configp.read('config.ini')


with open(configp["HICODET"]["anno_list_json"]) as json_file:
    data = json.load(json_file)

    for annotation in data:
        print("==============")
        print(annotation["global_id"])
        for hois in annotation["hois"]:
            print("---")
            print(hois["id"])
            print(hois["human_bboxes"])
            print(hois["object_bboxes"])
        
