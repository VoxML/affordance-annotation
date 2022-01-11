import os
import json
import torch

def _ori_dict_to_vec(ori_dict):
    vector = [0, 0, 0, 0, 0, 0]
    keys = ori_dict.keys()
    if "n/a" in keys or len(keys) == 0:
        return torch.tensor(vector)
    elif "+x" in keys:
        vector[0] = 1
    elif "-x" in keys:
        vector[1] = 1
    elif "+y" in keys:
        vector[2] = 1
    elif "-y" in keys:
        vector[3] = 1
    elif "+z" in keys:
        vector[4] = 1
    elif "-z" in keys:
        vector[5] = 1
    else:
        print(ori_dict)
        print("!!!!!!!!!!!!!!!!!")
    return torch.tensor(vector)


def prob_for_list_of_tensors(list_of_tensor):
    value_stack = torch.stack(list_of_tensor, dim=0)
    sum_stack = torch.sum(value_stack, dim=0).double()
    probs = sum_stack.softmax(dim=0)
    return probs

train_path = os.path.join("D:/Corpora/HICO-DET", "orientation_annotation", "ALL_train.json")
test_path = os.path.join("D:/Corpora/HICO-DET", "orientation_annotation", "ALL_test.json")

check_most_frequent_front = {}
check_most_frequent_up = {}
with open(train_path, 'r') as json_file:
    data = json.load(json_file)

    for img_id, anno in data["_via_img_metadata"].items():
        for region in anno["regions"]:
            front = _ori_dict_to_vec(region["region_attributes"]["front"])
            up = _ori_dict_to_vec(region["region_attributes"]["up"])
            if "category" not in region["region_attributes"]:
                continue
            elif region["region_attributes"]["category"] == "object":
                obj_name = region["region_attributes"]["obj name"]
            elif region["region_attributes"]["category"] == "human":
                obj_name = "human"
            else:
                print(region["region_attributes"]["category"] + " !!!!!!!!!!!!!!!!")
                exit()

            if not all(v == 0 for v in front):
                if obj_name in check_most_frequent_front:
                    check_most_frequent_front[obj_name].append(front)
                else:
                    check_most_frequent_front[obj_name] = [front]

            if not all(v == 0 for v in up):
                if obj_name in check_most_frequent_up:
                    check_most_frequent_up[obj_name].append(up)
                else:
                    check_most_frequent_up[obj_name] = [up]

print("======================")
map_front = {}
for_all_front = []
for key, values in check_most_frequent_front.items():
    if len(values) > 10:
        map_front[key] = prob_for_list_of_tensors(values)
        for_all_front += values
map_front["all"] = prob_for_list_of_tensors(for_all_front)

print("Probabilities for Front Direction:", map_front)
print("======================")

map_up = {}
for_all_up = []
for key, values in check_most_frequent_up.items():
    if len(values) > 10:
        map_up[key] = prob_for_list_of_tensors(values)
        for_all_up += values
map_up["all"] = prob_for_list_of_tensors(for_all_up)

print("Probabilities for Up Direction:", map_up)
print("======================")
result_dict = dict(up_correct_base=0, up_correct_per_type=0, up_all=0, front_correct_base=0, front_correct_per_type=0, front_all=0)

with open(test_path, 'r') as json_file:
    data = json.load(json_file)

    for img_id, anno in data["_via_img_metadata"].items():
        for region in anno["regions"]:
            front = _ori_dict_to_vec(region["region_attributes"]["front"])
            up = _ori_dict_to_vec(region["region_attributes"]["up"])
            if "category" not in region["region_attributes"]:
                continue
            elif region["region_attributes"]["category"] == "object":
                obj_name = region["region_attributes"]["obj name"]
            elif region["region_attributes"]["category"] == "human":
                obj_name = "human"
            else:
                print(region["region_attributes"]["category"] + " !!!!!!!!!!!!!!!!")
                exit()

            if not all(v == 0 for v in front):
                result_dict["front_all"] += 1
                if obj_name in map_front:
                    if torch.argmax(front) == torch.argmax(map_front[obj_name]):
                        result_dict["front_correct_per_type"] += 1

                if torch.argmax(front) == torch.argmax(map_front["all"]):
                    result_dict["front_correct_base"] += 1

            if not all(v == 0 for v in up):
                result_dict["up_all"] += 1
                if obj_name in map_up:
                    if torch.argmax(up) == torch.argmax(map_up[obj_name]):
                        result_dict["up_correct_per_type"] += 1

                if torch.argmax(up) == torch.argmax(map_up["all"]):
                    result_dict["up_correct_base"] += 1

print("Baseline Front:", result_dict["front_correct_base"] / result_dict["front_all"])
print("Baseline Per Obj Front:", result_dict["front_correct_per_type"] / result_dict["front_all"])
print("...")
print("Baseline Up:", result_dict["up_correct_base"] / result_dict["up_all"])
print("Baseline Per Obj Up:", result_dict["up_correct_per_type"] / result_dict["up_all"])