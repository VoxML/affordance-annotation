import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from sklearn.metrics import cohen_kappa_score
import shutil
plt.rcParams["figure.figsize"] = (20, 5)

from tqdm import tqdm
from utils import get_iou, ori_dict_to_vec, merge_bboxes, id2label, ori_vec_to_binvec


def create_heatmap(df: pd.DataFrame, title: str, subtitle: str = "", cmap="coolwarm", norm=None):
    df["sent"] = df.index

    df = pd.melt(df, id_vars=['sent'], var_name='target', value_name="score")
    #df = df.sort_values(by=["sent", "target"])
    df = df.sort_values(by=["sent"])
    df.rename(columns={"sent": "Objects", "target": "Orientation"}, inplace=True)

    if norm is None:
        scoremax = df["score"].max()
        scoremin = df["score"].min()
    else:
        scoremax = norm[1]
        scoremin = norm[0]

    #plt.figure(1, figsize=(6, 12))
    plt.figure(1, figsize=(8, 16))

    ax = sns.scatterplot(x="Orientation", y="Objects",
                         hue="score", size="score",
                         hue_norm=(scoremin, scoremax), #size_norm=(0, 1),
                         palette=cmap, sizes=(0, 90),
                         marker="s", linewidth=0, legend=False,
                         data=df)
    ax.figure.axes[0].invert_xaxis()
    norm = plt.Normalize(scoremin, scoremax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    ax.figure.colorbar(sm)
    plt.xticks(rotation=90)
    # plt.title(title)
    plt.title(title + '\n' + subtitle)
    # plt.text(0.5, 1, 'the third line', fontsize=13, ha='center')



def generate_matplot_graph(dfall, labels=None, title="multiple stacked bar plot", H="/", sort_vals=None, color_dict={}, **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot.
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)
    axe = plt.subplot(111)

    for df in dfall:  # for each data frame
        if sort_vals != None:
            df.sort_index(inplace=True)

        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      #color=[color_dict.get(x) for x in df.columns],
                      **kwargs)  # make bar plots

    h, l = axe.get_legend_handles_labels()  # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i + n_col]):
            for rect in pa.patches:  # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col))  # edited part
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(dfall[0].index)#, rotation=90)
    axe.set_title(title)

    # Add invisible data to add another legend
    n = []
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i * 2))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.15])
    axe.add_artist(l1)
    if labels is not None:
        l2 = plt.legend(n, labels, loc=[1.01, 0.00])

    return axe


def generate_matplot_graph_with_line(df, box_values, line_values=None):
    # fig = plt.figure()
    ax = df[box_values].plot(kind='bar', use_index=True)
    ax2 = ax.twinx()
    if line_values != None:
        lines = ax2.plot(df[line_values].values, linestyle='-', marker='o', linewidth=2.0)

        # https://matplotlib.org/stable/_images/dflt_style_changes-1.png
        x = np.arange(0, len(df.index), 1)

        color = ["#1f77b4", '#ff7f0e'] * 100
        for line_idx, line_value in enumerate(line_values):
            y_mean = [df[line_value].mean()] * len(df.index)
            ax2.plot(x, y_mean, label=f'{line_value} mean', linestyle='--', color=color[line_idx])

    # y_mean = [df['telic_score'].mean()] * len(df.index)
    # ax2.plot(x, y_mean, label='Telic Mean', linestyle='--', color='#ff7f0e')

        ax2.legend(lines, line_values, loc=1)


def generate_detr_vs_ori_df(config, threshhold=0.8, ori="up"):
    anno_path = os.path.join(config.hicodet_path, "via234_1200 items_train verified.json")
    filter_names = ["apple", "bicycle", "bottle", "car", "chair", "cup", "dog", "horse", "knife", "person", "umbrella"]
    with open(anno_path) as json_file:
        anno_data = json.load(json_file)

    with open(config.processed_path) as json_file:
        pred_data = json.load(json_file)

    error_dfs = {"correct": pd.DataFrame(), "not_det": pd.DataFrame(), "new": pd.DataFrame()}

    for _, annotation in tqdm(anno_data["_via_img_metadata"].items()):
        filename = annotation["filename"]
        preds = pred_data[filename]

        for region in annotation["regions"]:
            bbox = [region["shape_attributes"]["x"], region["shape_attributes"]["y"],
                    region["shape_attributes"]["x"] + region["shape_attributes"]["width"],
                    region["shape_attributes"]["y"] + region["shape_attributes"]["height"]]

            front_vec = ori_dict_to_vec(region["region_attributes"]["front"])
            up_vec = ori_dict_to_vec(region["region_attributes"]["up"])
            if ori == "front-up":
                selected_orientation = f"{front_vec}_{up_vec}"
            elif ori == "front":
                selected_orientation = f"{front_vec}"
            elif ori == "up":
                selected_orientation = f"{up_vec}"
            else:
                print("!!!!!!!!")
                exit()

            if "category" not in region["region_attributes"]:
                continue
            elif region["region_attributes"]["category"] == "human":
                name = "person"
            elif region["region_attributes"]["category"] == "object":
                name = region["region_attributes"]["obj name"].replace(" ", "_")
            else:
                print("???????????????????")
                exit()
            if name not in filter_names:
                continue

            if selected_orientation not in error_dfs["correct"].index:
                for df in error_dfs.values():
                    df.loc[selected_orientation, :] = 0

            if name not in error_dfs["correct"].columns:
                for df in error_dfs.values():
                    df.loc[:, name] = 0

            found = False
            found_other = False
            for pred_bbox, pred_label, pred_bbox_score in zip(preds["boxes"], preds["boxes_label_names"],
                                                              preds["boxes_scores"]):
                if pred_bbox_score < threshhold:
                    continue

                pred_label = pred_label.replace(" ", "_")

                iou = get_iou({"x1": pred_bbox[0], "x2": pred_bbox[2], "y1": pred_bbox[1], "y2": pred_bbox[3]},
                              {"x1": bbox[0], "x2": bbox[2], "y1": bbox[1], "y2": bbox[3]})

                if iou > 0.5 and pred_label == name:
                    found = True
                    error_dfs["correct"].loc[selected_orientation, pred_label] += 1
                    break
                elif iou > 0.5 and pred_label != name:
                    found_other = True

            if not found and not found_other:
                error_dfs["new"].loc[selected_orientation, name] += 1
            elif not found:
                error_dfs["not_det"].loc[selected_orientation, name] += 1

    for key, df in error_dfs.items():
        df = df.transpose()
        df = df.drop("person")
        error_dfs[key] = df

    return error_dfs


def generate_APm_DETR(config):
    anno_path = os.path.join(config.hicodet_path, "instances_train2015.json")
    with open(anno_path) as json_file:
        train_data = json.load(json_file)

    anno_path = os.path.join(config.hicodet_path, "instances_test2015.json")
    with open(anno_path) as json_file:
        eval_data = json.load(json_file)

    hoi_annotation = {}  # {obj#verb: T/G/-} (string)
    with open(os.path.join(config.hicodet_path, "HOI.txt")) as file:
        for line in file:
            splitline = line.split()
            if len(splitline) > 3:
                if splitline[3] == "T":
                    hoi_annotation[(splitline[1], splitline[2])] = 1
                elif splitline[3] == "G":
                    hoi_annotation[(splitline[1], splitline[2])] = 0

    object_names = train_data["objects"]
    object_names = [object_name.replace(" ", "_") for object_name in object_names]
    verb_names = train_data['verbs']

    df = pd.DataFrame(0, index=object_names,
                      columns=["hicodet_obj_train_count", "hicodet_obj_test_count",
                               "hicodet_telic_train_count", "hicodet_gibsonian_train_count",
                               "hicodet_telic_test_count", "hicodet_gibsonian_test_count",
                               "gibsonian_score", "telic_score"])

    for anno in train_data["annotation"]:
        for obj in anno["object"]:
            df["hicodet_obj_train_count"][obj] += 1
    for anno in eval_data["annotation"]:
        for obj in anno["object"]:
            df["hicodet_obj_test_count"][obj] += 1

    eval_results = [0.00, 0.00, 0.21988659, 0.00637816, 0.32196299, 0.57225427
        , 0.30477074, 0.26946023, 0.29352602, 0.63404685, 0.03190559, 0.33536871
        , 0.12791223, 0.30757226, 0.16880263, 0.29212872, 0.08298403, 0.30189208
        , 0.06167575, 0.42664007, 0.00, 0.11363636, 0.53636364, 0.43593074
        , 0.00, 0.00, 0.18274899, 0.02272727, 0.00, 0.42424242
        , 0.51023601, 0.00460969, 0.18234428, 0.10461825, 0.51702546, 0.10313216
        , 0.54625449, 0.16303471, 0.37690076, 0.72149081, 0.39034195, 0.31310025
        , 0.25638154, 0.23803191, 0.58076267, 0.23398578, 0.17300234, 0.37878788
        , 0.4018595, 0.58712121, 0.3039841, 0.51158633, 0.00, 0.00
        , 0.05830465, 0.45147618, 0.52423709, 0.0286887, 0.00, 0.00
        , 0.00, 0.00, 0.26853411, 0.0019406, 0.00327429, 0.43641888
        , 0.32523612, 0.10415449, 0.48089509, 0.10772434, 0.11684999, 0.36379214
        , 0.18249577, 0.51666289, 0.34898088, 0.31722398, 0.12366018, 0.27377885
        , 0.04438164, 0.37450821, 0.03724832, 0.30532292, 0.32749023, 0.56221177
        , 0.49662548, 0.39352541, 0.19327402, 0.41042864, 0.28364, 0.19850593
        , 0.00, 0.00, 0.32881731, 0.26139166, 0.29611321, 0.2188021
        , 0.22231405, 0.00, 0.0229319, 0.17720892, 0.23993784, 0.12303486
        , 0.12205723, 0.20328321, 0.1164393, 0.25834627, 0.30219513, 0.22849272
        , 0.22856531, 0.18660176, 0.40668884, 0.2455452, 0.14566234, 0.26599314
        , 0.13980716, 0.08915691, 0.10495543, 0.1323664, 0.19446854, 0.30616779
        , 0.20631197, 0.32890633, 0.35335088, 0.50988398, 0.27387632, 0.00
        , 0.40476557, 0.00, 0.25510248, 0.00, 0.50061415, 0.29671995
        , 0.00, 0.00, 0.12623162, 0.45551303, 0.00, 0.00
        , 0.00, 0.00, 0.17086656, 0.74770022, 0.00, 0.00
        , 0.00, 0.18684285, 0.20673035, 0.58033889, 0.00, 0.23246753
        , 0.18048392, 0.00, 0.28482166, 0.490595, 0.1080334, 0.1733771
        , 0.00, 0.47315478, 0.01505216, 0.2323027, 0.45454545, 0.29545455
        , 0.01818182, 0.27597403, 0.1958042, 0.42519418, 0.00, 0.00
        , 0.12786195, 0.39799712, 0.01048951, 0.2590783, 0.00, 0.27272727
        , 0.01728512, 0.21769106, 0.58926262, 0.00, 0.07009167, 0.37464115
        , 0.1906608, 0.49947662]

    train_anno_interactions = [0, 0, 679, 167, 2138, 2391, 134, 931, 1265, 1641, 219, 801, 736, 1839, 270, 648, 106,
                               522, 1956, 4965, 20, 48, 1, 41, 0, 0, 25, 1, 0, 42, 2067, 13, 476, 257, 225, 4, 644, 78,
                               826, 1696, 324, 679, 258, 288, 1211, 41, 83, 82, 74, 8, 294, 158, 0, 0, 302, 1044, 2159,
                               38, 0, 0, 0, 0, 206, 6, 32, 858, 360, 48, 797, 93, 481, 1359, 281, 936, 974, 213, 598,
                               952, 245, 741, 38, 199, 1197, 1743, 408, 237, 235, 86, 420, 221, 0, 0, 327, 151, 243,
                               190, 180, 0, 162, 633, 240, 32, 66, 34, 450, 325, 193, 82, 112, 132, 42, 5, 27, 47, 237,
                               64, 183, 254, 173, 253, 238, 177, 464, 436, 1386, 0, 1123, 0, 23, 0, 465, 2, 0, 0, 1791,
                               2324, 0, 0, 0, 0, 60, 25, 0, 0, 0, 237, 74, 729, 27, 48, 197, 0, 57, 221, 330, 496, 0,
                               11, 5, 104, 2, 3, 6, 14, 25, 24, 0, 0, 265, 811, 23, 80, 8, 17, 122, 195, 305, 0, 9, 51,
                               165, 232]
    test_anno_interactions = [0, 0, 189, 22, 553, 589, 35, 260, 355, 484, 42, 239, 215, 538, 67, 132, 46, 182, 354,
                              1176, 10, 18, 6, 16, 0, 0, 7, 3, 0, 11, 416, 10, 151, 56, 74, 5, 152, 58, 216, 517, 80,
                              89, 78, 119, 266, 11, 11, 27, 11, 5, 68, 57, 0, 0, 100, 250, 464, 26, 0, 0, 0, 0, 56, 8,
                              8, 195, 114, 17, 207, 37, 149, 380, 65, 231, 256, 81, 206, 285, 44, 195, 16, 46, 308, 488,
                              133, 74, 49, 35, 74, 93, 0, 0, 71, 66, 54, 66, 57, 0, 45, 183, 56, 22, 21, 6, 94, 116, 65,
                              30, 31, 43, 31, 25, 27, 20, 44, 86, 39, 151, 50, 108, 54, 55, 122, 112, 321, 0, 272, 0,
                              15, 0, 97, 6, 0, 0, 318, 639, 0, 0, 0, 0, 23, 23, 0, 0, 0, 102, 21, 170, 5, 14, 56, 0, 11,
                              52, 65, 168, 0, 11, 5, 35, 2, 17, 2, 10, 10, 10, 0, 0, 99, 217, 4, 21, 3, 10, 20, 64, 66,
                              0, 5, 16, 50, 74]

    for x in range(0, len(eval_results), 2):
        obj_idx = str(int(x / 2))
        obj_name = id2label[obj_idx].replace(" ", "_")
        if obj_name in df.index:
            df.loc[obj_name, "gibsonian_score"] = eval_results[x]
            df.loc[obj_name, "telic_score"] = eval_results[x + 1]
            df.loc[obj_name, "hicodet_gibsonian_train_count"] = train_anno_interactions[x]
            df.loc[obj_name, "hicodet_telic_train_count"] = train_anno_interactions[x + 1]
            df.loc[obj_name, "hicodet_gibsonian_test_count"] = train_anno_interactions[x]
            df.loc[obj_name, "hicodet_telic_test_count"] = train_anno_interactions[x + 1]
    pd.set_option('display.expand_frame_repr', False)

    df["tg_diff"] = df["hicodet_gibsonian_train_count"] - df["hicodet_telic_train_count"]

    # df = df.sort_values(["gibsonian_score"], ascending=False)
    df = df.sort_values(["tg_diff"], ascending=False)
    # df = df.sort_values(["hicodet_gibsonian_train_count"], ascending=False)

    df["obj"] = df.index
    return df


def merge_everything(config):
    anno_path = os.path.join(config.hicodet_path, "instances_train2015.json")
    with open(anno_path) as json_file:
        train_data = json.load(json_file)

    anno_path = os.path.join(config.hicodet_path, "instances_test2015.json")
    with open(anno_path) as json_file:
        eval_data = json.load(json_file)

    anno_path = os.path.join(config.hicodet_path, "via234_1200 items_train verified v2.json")
    with open(anno_path) as json_file:
        anno_data = json.load(json_file)

    object_names = train_data["objects"]
    verb_names = train_data["verbs"]
    merged_results = {}

    # Merge multiple bboxes for one object
    for file_name, annotation in zip(train_data["filenames"], train_data["annotation"]):
        train_data_merged = merge_bboxes(annotation, object_names, verb_names)
        merged_results[file_name] = train_data_merged

    for file_name, annotation in zip(eval_data["filenames"], eval_data["annotation"]):
        train_data_merged = merge_bboxes(annotation, object_names, verb_names)
        merged_results[file_name] = train_data_merged

    error_count = 0
    # Add Anju Annotations
    for _, annotation in tqdm(anno_data["_via_img_metadata"].items()):
        filename = annotation["filename"]

        hicodet_anno = merged_results[filename]

        human_bboxes_ori = [None for _ in range(len(hicodet_anno["human_bboxes"]))]
        object_bboxes_ori = [None for _ in range(len(hicodet_anno["object_bboxes"]))]

        remap_obj_id = {}
        remap_human_id = {}
        annotated_connections = []
        annotated_actions = []
        annotated_affordances = []

        # For every marked object
        for region_id, region in enumerate(annotation["regions"]):
            bbox = [region["shape_attributes"]["x"], region["shape_attributes"]["y"],
                    region["shape_attributes"]["x"] + region["shape_attributes"]["width"],
                    region["shape_attributes"]["y"] + region["shape_attributes"]["height"]]
            front_vec = ori_dict_to_vec(region["region_attributes"]["front"])
            up_vec = ori_dict_to_vec(region["region_attributes"]["up"])

            # Check if Person or Object
            if "category" not in region["region_attributes"]:
                continue
            elif region["region_attributes"]["category"] == "human":
                found = False
                for h_bbox_idx, h_bbox in enumerate(hicodet_anno["human_bboxes"]):
                    iou = get_iou({"x1": bbox[0], "x2": bbox[2], "y1": bbox[1], "y2": bbox[3]},
                                  {"x1": h_bbox[0], "x2": h_bbox[2], "y1": h_bbox[1], "y2": h_bbox[3]})
                    if iou > 0.5:
                        found = True
                        human_bboxes_ori[h_bbox_idx] = [front_vec, up_vec]
                        remap_human_id[region_id] = h_bbox_idx
                        break # Not shure if I sould break ....

                if not found:
                    # Add new annotated human
                    hicodet_anno["human_bboxes"].append(bbox)
                    human_bboxes_ori.append([front_vec, up_vec])
                    remap_human_id[region_id] = len(hicodet_anno["human_bboxes"]) - 1


            elif region["region_attributes"]["category"] == "object":
                found = False
                obj_name = region["region_attributes"]["obj name"].replace(" ", "_")
                for o_bbox_idx, (o_bbox, o_bbox_lab) in enumerate(
                        zip(hicodet_anno["object_bboxes"], hicodet_anno["object_labels"])):
                    iou = get_iou({"x1": bbox[0], "x2": bbox[2], "y1": bbox[1], "y2": bbox[3]},
                                  {"x1": o_bbox[0], "x2": o_bbox[2], "y1": o_bbox[1], "y2": o_bbox[3]})
                    if iou > 0.5 and obj_name == o_bbox_lab:
                        found = True
                        object_bboxes_ori[o_bbox_idx] = [front_vec, up_vec]
                        remap_obj_id[region_id] = o_bbox_idx
                if not found:
                    # Add new annotated object
                    hicodet_anno["object_bboxes"].append(bbox)
                    object_bboxes_ori.append([front_vec, up_vec])
                    hicodet_anno["object_labels"].append(obj_name)
                    remap_obj_id[region_id] = len(hicodet_anno["object_bboxes"]) - 1
            else:
                print("???????????????????") # Not human nore object. should never happen.
                exit()

        # Now add the coresponding links.
        for region_id, region in enumerate(annotation["regions"]):
            if "category" not in region["region_attributes"]:
                continue
            elif region["region_attributes"]["category"] == "human":
                anno_actions = region["region_attributes"]["action"].split(",")
                anno_affordances = region["region_attributes"]["affordance"].split(",")
                anno_objs = region["region_attributes"]["obj id"].split(",")
                if len(anno_actions) == len(anno_affordances) == len(anno_objs): #13 annotations are missing something here ...
                    for ac, af, ob in zip(anno_actions, anno_affordances, anno_objs):
                        if af.isnumeric(): # min one annotation had it flipped...
                            af, ob = ob, af

                        if ob.isnumeric() and "category" in annotation["regions"][int(ob) - 1]["region_attributes"]:
                            if int(ob) - 1 in remap_human_id: # I am currenntly not interested in links between humans
                                continue
                            annotated_connections.append((remap_human_id[region_id], remap_obj_id[int(ob) - 1]))
                            annotated_actions.append(ac)
                            annotated_affordances.append(af.strip())
                else:
                    error_count += 1
        hicodet_anno["human_bboxes_ori"] = human_bboxes_ori
        hicodet_anno["object_bboxes_ori"] = object_bboxes_ori
        hicodet_anno["annotated_connections"] = annotated_connections
        hicodet_anno["annotated_actions"] = annotated_actions
        hicodet_anno["annotated_affordances"] = annotated_affordances
    print(error_count)
    return {"annotations": merged_results, "objects": object_names, "verbs": verb_names}


def generate_anno_hoi_vs_auto_hoi_iaa(annos, object_names, config):
    #object_names = ["apple", "bicycle", "bottle", "car", "chair", "cup", "dog", "horse", "knife", "umbrella"]

    hoi_annotation = {}  # {obj#verb: T/G/-} (string)
    with open(os.path.join(config.hicodet_path, "HOI.txt")) as file:
        for line in file:
            splitline = line.split()
            if len(splitline) > 3:
                if splitline[3] == "T":
                    hoi_annotation[(splitline[1], splitline[2])] = "telic"
                elif splitline[3] == "G":
                    hoi_annotation[(splitline[1], splitline[2])] = "gibsonian"

    annotator1 = []
    annotator2 = []
    anno_map = {"T": "telic", "G": "gibsonian", "G pot. T": "gibsonian"}

    for anno_file, anno in annos["annotations"].items():
        if "annotated_connections" in anno:
            merged_auto_hois = {}
            for connection, verb in zip(anno["connections"], anno["connection_verbs"]):
                obj = anno["object_labels"][connection[1]]
                if (obj, verb) not in hoi_annotation:
                    continue
                if connection in merged_auto_hois and merged_auto_hois[connection] != "telic":
                    merged_auto_hois[connection] = hoi_annotation[(obj, verb)]
                elif connection not in merged_auto_hois:
                    merged_auto_hois[connection] = hoi_annotation[(obj, verb)]

            merged_anno_hois = {}
            for connection, verb in zip(anno["annotated_connections"], anno["annotated_affordances"]):
                if verb == "None":
                    continue
                if connection in merged_anno_hois:
                    if merged_anno_hois[connection] == "gibsonian":
                        merged_anno_hois[connection] = anno_map[verb]
                    elif merged_anno_hois[connection] == "gibs / telic" and anno_map[verb] == "telic":
                        merged_anno_hois[connection] = anno_map[verb]
                elif connection not in merged_anno_hois:
                    merged_anno_hois[connection] = anno_map[verb]

            for anno_conntection, hoi in merged_anno_hois.items():
                obj = anno["object_labels"][anno_conntection[1]]
                if anno_conntection in merged_auto_hois and obj in object_names:
                    annotator1.append(hoi)
                    annotator2.append(merged_auto_hois[anno_conntection])

    return annotator1, annotator2

def generate_anno_hoi_vs_auto_hoi(annos, config):
    object_names = ["apple", "bicycle", "bottle", "car", "chair", "cup", "dog", "horse", "knife", "umbrella"]

    hoi_annotation = {}  # {obj#verb: T/G/-} (string)
    with open(os.path.join(config.hicodet_path, "HOI.txt")) as file:
        for line in file:
            splitline = line.split()
            if len(splitline) > 3:
                if splitline[3] == "T":
                    hoi_annotation[(splitline[1], splitline[2])] = "telic"
                elif splitline[3] == "G":
                    hoi_annotation[(splitline[1], splitline[2])] = "gibsonian"

    auto_gibs_df = pd.DataFrame(0, index=object_names, columns=["gibsonian", "telic"])
    auto_telic_df = pd.DataFrame(0, index=object_names, columns=["gibsonian", "telic"])
    anno_map = {"T": "telic", "G": "gibsonian", "G pot. T": "gibsonian"}

    for anno_file, anno in annos["annotations"].items():
        if "annotated_connections" in anno:
            merged_auto_hois = {}
            for connection, verb in zip(anno["connections"], anno["connection_verbs"]):
                obj = anno["object_labels"][connection[1]]
                if (obj, verb) not in hoi_annotation:
                    continue
                if connection in merged_auto_hois and merged_auto_hois[connection] != "telic":
                    merged_auto_hois[connection] = hoi_annotation[(obj, verb)]
                elif connection not in merged_auto_hois:
                    merged_auto_hois[connection] = hoi_annotation[(obj, verb)]

            merged_anno_hois = {}
            for connection, verb in zip(anno["annotated_connections"], anno["annotated_affordances"]):
                if verb == "None":
                    continue
                if connection in merged_anno_hois:
                    if merged_anno_hois[connection] == "gibsonian":
                        merged_anno_hois[connection] = anno_map[verb]
                    elif merged_anno_hois[connection] == "gibs / telic" and anno_map[verb] == "telic":
                        merged_anno_hois[connection] = anno_map[verb]
                elif connection not in merged_anno_hois:
                    merged_anno_hois[connection] = anno_map[verb]

            for anno_conntection, hoi in merged_anno_hois.items():
                obj = anno["object_labels"][anno_conntection[1]]
                if anno_conntection in merged_auto_hois and obj in object_names:
                    auto_hoi = merged_auto_hois[anno_conntection]
                    if auto_hoi == "telic":
                        auto_telic_df.loc[obj, hoi] += 1
                    elif auto_hoi == "gibsonian":
                        auto_gibs_df.loc[obj, hoi] += 1
                    else:
                        print("ERRO!")
                        exit()

    return auto_gibs_df, auto_telic_df


def Hoi_vs_AnnoOri(annos, config, auto=False, ori="up"):
    object_names = ["apple", "bicycle", "bottle", "car", "chair", "cup", "dog", "horse", "knife", "person", "umbrella"]
    hoi_annotation = {}  # {obj#verb: T/G/-} (string)
    with open(os.path.join(config.hicodet_path, "HOI.txt")) as file:
        for line in file:
            splitline = line.split()
            if len(splitline) > 3:
                if splitline[3] == "T":
                    hoi_annotation[(splitline[1], splitline[2])] = "telic"
                elif splitline[3] == "G":
                    hoi_annotation[(splitline[1], splitline[2])] = "gibsonian"

    dfs = {"gibsonian": pd.DataFrame(), "gibs / telic": pd.DataFrame() , "telic": pd.DataFrame()}

    anno_map = {"T": "telic", "G": "gibsonian", "G pot. T": "gibs / telic"}

    for anno_file, anno in annos["annotations"].items():
        if "annotated_connections" in anno:
            merged_hois = {}
            if auto:
                for connection, verb in zip(anno["connections"], anno["connection_verbs"]):
                    obj = anno["object_labels"][connection[1]]
                    if (obj, verb) not in hoi_annotation:
                        continue
                    if connection in merged_hois and merged_hois[connection] != "telic":
                        merged_hois[connection] = hoi_annotation[(obj, verb)]
                    elif connection not in merged_hois:
                        merged_hois[connection] = hoi_annotation[(obj, verb)]
            else:
                for connection, verb in zip(anno["annotated_connections"], anno["annotated_affordances"]):
                    if verb == "None":
                        continue
                    if connection in merged_hois:
                        if merged_hois[connection] == "gibsonian":
                            merged_hois[connection] = anno_map[verb]
                        elif merged_hois[connection] == "gibs / telic" and anno_map[verb] == "telic":
                            merged_hois[connection] = anno_map[verb]
                    elif connection not in merged_hois:
                        merged_hois[connection] = anno_map[verb]

            for connection, hoi in merged_hois.items():
                obj = anno["object_labels"][connection[1]]
                if obj not in object_names:
                    continue

                obj_ori = anno["object_bboxes_ori"][connection[1]]
                if obj_ori is None:
                    continue

                if ori == "front":
                    selected_orientation = str(obj_ori[0])
                elif ori == "up":
                    selected_orientation = str(obj_ori[1])
                else:
                    selected_orientation = str(obj_ori[0])+"_"+str(obj_ori[1])

                if obj not in dfs["gibsonian"].index:
                    for df in dfs.values():
                        df.loc[obj, :] = 0

                if selected_orientation not in dfs["gibsonian"].columns:
                    for df in dfs.values():
                        df.loc[:, selected_orientation] = 0

                dfs[hoi].loc[obj, selected_orientation] += 1

    return dfs


def AnnoHoi_vs_AnnoOri(annos, config, merge_pot, relative=True):
    object_names = ["apple", "bicycle", "bottle", "car", "chair", "cup", "dog", "horse", "knife", "person", "umbrella"]
    #object_names = ["bicycle", "bottle", "car", "chair", "cup", "dog", "horse", "knife", "person", "umbrella"]

    hoi_annotation = {}  # {obj#verb: T/G/-} (string)
    with open(os.path.join(config.hicodet_path, "HOI.txt")) as file:
        for line in file:
            splitline = line.split()
            if len(splitline) > 3:
                if splitline[3] == "T":
                    hoi_annotation[(splitline[1], splitline[2])] = "telic"
                elif splitline[3] == "G":
                    hoi_annotation[(splitline[1], splitline[2])] = "gibsonian"

    dfs = {"gibsonian": pd.DataFrame(), "gibs / telic": pd.DataFrame() , "telic": pd.DataFrame()}

    anno_map = {"T": "telic", "G": "gibsonian", "G pot. T": "gibs / telic"}

    for anno_file, anno in annos["annotations"].items():
        if "annotated_connections" in anno:
            merged_hois = {}
            for connection, verb in zip(anno["annotated_connections"], anno["annotated_affordances"]):
                if verb == "None":
                    continue
                if connection in merged_hois:
                    if merged_hois[connection] == "gibsonian":
                        merged_hois[connection] = anno_map[verb]
                    elif merged_hois[connection] == "gibs / telic" and anno_map[verb] == "telic":
                        merged_hois[connection] = anno_map[verb]
                elif connection not in merged_hois:
                    merged_hois[connection] = anno_map[verb]

            for connection, hoi in merged_hois.items():
                obj = anno["object_labels"][connection[1]]
                if obj not in object_names:
                    continue

                obj_ori = anno["object_bboxes_ori"][connection[1]]
                if obj_ori is None:
                    continue

                human_ori = anno["human_bboxes_ori"][connection[0]]
                if human_ori is None:
                    continue

                if relative:
                    relative_up, relative_front = utils.relative_orientations(obj_ori[1], obj_ori[0], human_ori[1], human_ori[0])
                    relative_up = relative_up.astype(int).tolist()
                    relative_front = relative_front.astype(int).tolist()
                else:
                    relative_up = obj_ori[1]
                    relative_front = obj_ori[0]
                selected_orientation = str(relative_front)+"_"+str(relative_up)

                #hoi_save = hoi.replace(" / ", "+")
                #shutil.copy(os.path.join(config.hicodet_path, "hico_20160224_det", "images", "merge2015", anno_file),
                #            os.path.join(config.hicodet_path, "hico_20160224_det", "images", "anno_test", f"{obj}-{hoi_save}-{selected_orientation}-{anno_file}"))

                if obj not in dfs["gibsonian"].index:
                    for df in dfs.values():
                        df.loc[obj, :] = 0

                if selected_orientation not in dfs["gibsonian"].columns:
                    for df in dfs.values():
                        df.loc[:, selected_orientation] = 0

                dfs[hoi].loc[obj, selected_orientation] += 1

    if merge_pot:
        gibs_df = dfs["gibsonian"]
        pot_df = dfs["gibs / telic"]
        test = gibs_df.add(pot_df, fill_value=0)
        dfs["gibsonian"] = test
        dfs.pop("gibs / telic")
    return dfs


def ModelHoi_vs_Ori(config, ori="front", threshhold = 0.8):
    with open(config.processed_path) as json_file:
        pred_data = json.load(json_file)

    dfs = {"gibsonian": pd.DataFrame(), "telic": pd.DataFrame()}
    anno_map = {"T": "telic", "G": "gibsonian"}

    for anno_file, anno in tqdm(pred_data.items()):
        if len(anno) == 0:
            continue
        for x in range(0, len(anno["pairing_label"]), 2):
            hum_id = anno["pairing"][0][x]
            hum_id_score = anno["boxes_scores"][hum_id]

            obj_id = anno["pairing"][1][x]
            obj_id_score = anno["boxes_scores"][obj_id]
            obj_name = anno["boxes_label_names"][obj_id]
            g_score = anno["pairing_scores"][x]
            t_score = anno["pairing_scores"][x+1]

            if ori == "front":
                selected_orientation = str(ori_vec_to_binvec(anno["boxes_orientation"][obj_id]["front"]))
            elif ori == "up":
                selected_orientation = str(ori_vec_to_binvec(anno["boxes_orientation"][obj_id]["up"]))
            else:
                selected_orientation = str(ori_vec_to_binvec(anno["boxes_orientation"][obj_id]["front"])) + \
                                       "_" + str(ori_vec_to_binvec(anno["boxes_orientation"][obj_id]["up"]))

            if hum_id_score > threshhold and obj_id_score > threshhold:
                if max(g_score, t_score) > 0.2:
                    if obj_name not in dfs["gibsonian"].index:
                        for df in dfs.values():
                            df.loc[obj_name, :] = 0

                    if selected_orientation not in dfs["gibsonian"].columns:
                        for df in dfs.values():
                            df.loc[:, selected_orientation] = 0

                    if g_score > t_score:
                        dfs["gibsonian"].loc[obj_name, selected_orientation] += 1
                    else:
                        dfs["telic"].loc[obj_name, selected_orientation] += 1

    return dfs


def ModelHoi_vs_ModelOri(config, filter, threshhold=0.8, relative=True):
    object_names = ["apple", "bicycle", "bottle", "car", "chair", "cup", "dog", "horse", "knife", "umbrella"]

    with open(config.processed_path) as json_file:
        pred_data = json.load(json_file)

    dfs = {"gibsonian": pd.DataFrame(), "telic": pd.DataFrame()}
    anno_map = {"T": "telic", "G": "gibsonian"}

    for anno_file, anno in tqdm(pred_data.items()):
        if len(anno) == 0:
            continue
        for x in range(0, len(anno["pairing_label"]), 2):
            hum_id = anno["pairing"][0][x]
            hum_id_score = anno["boxes_scores"][hum_id]
            hum_orientation = anno["boxes_orientation"][hum_id]

            obj_id = anno["pairing"][1][x]
            obj_id_score = anno["boxes_scores"][obj_id]
            obj_name = anno["boxes_label_names"][obj_id]
            obj_orientation = anno["boxes_orientation"][obj_id]

            if obj_name not in object_names and filter:
                continue

            if relative:
                relative_up, relative_front = utils.relative_orientations(
                    ori_vec_to_binvec(obj_orientation["up"]), ori_vec_to_binvec(obj_orientation["front"]),
                    ori_vec_to_binvec(hum_orientation["up"]), ori_vec_to_binvec(hum_orientation["front"]))
                relative_up = relative_up.astype(int).tolist()
                relative_front = relative_front.astype(int).tolist()
            else:
                relative_up = ori_vec_to_binvec(obj_orientation["up"])
                relative_front = ori_vec_to_binvec(obj_orientation["front"])

            selected_orientation = str(relative_front) + "_" + str(relative_up)
            g_score = anno["pairing_scores"][x]
            t_score = anno["pairing_scores"][x+1]


            if hum_id_score > threshhold and obj_id_score > threshhold:
                if max(g_score, t_score) > 0.2:
                    if obj_name not in dfs["gibsonian"].index:
                        for df in dfs.values():
                            df.loc[obj_name, :] = 0

                    if selected_orientation not in dfs["gibsonian"].columns:
                        for df in dfs.values():
                            df.loc[:, selected_orientation] = 0

                    if g_score > t_score:
                        dfs["gibsonian"].loc[obj_name, selected_orientation] += 1
                    else:
                        dfs["telic"].loc[obj_name, selected_orientation] += 1

    return dfs

"""
def generate_anno_hoi_vs_pred_hoi(annos, config):
    object_names = ["apple", "bicycle", "bottle", "car", "chair", "cup", "dog", "horse", "knife", "umbrella"]

    auto_gibs_df = pd.DataFrame(0, index=object_names, columns=["gibsonian", "gibs / telic", "telic"])
    auto_telic_df = pd.DataFrame(0, index=object_names, columns=["gibsonian", "gibs / telic", "telic"])
    anno_map = {"T": "telic", "G": "gibsonian", "G pot. T": "gibs / telic"}

    with open(config.processed_path) as json_file:
        pred_data = json.load(json_file)

    for anno_file, anno in annos["annotations"].items():
        if "annotated_connections" in anno:
            #merged_auto_hois = {}
            pred_results = pred_data[anno_file]
            
            #for connection, verb in zip(anno["connections"], anno["connection_verbs"]):
            #    obj = anno["object_labels"][connection[1]]
            #    if (obj, verb) not in hoi_annotation:
            #        continue
            #    if connection in merged_auto_hois and merged_auto_hois[connection] != "telic":
            #        merged_auto_hois[connection] = hoi_annotation[(obj, verb)]
            #    elif connection not in merged_auto_hois:
            #        merged_auto_hois[connection] = hoi_annotation[(obj, verb)]
            
            merged_anno_hois = {}
            for connection, verb in zip(anno["annotated_connections"], anno["annotated_affordances"]):
                if verb == "None":
                    continue
                if connection in merged_anno_hois:
                    if merged_anno_hois[connection] == "gibsonian":
                        merged_anno_hois[connection] = anno_map[verb]
                    elif merged_anno_hois[connection] == "gibs / telic" and anno_map[verb] == "telic":
                        merged_anno_hois[connection] = anno_map[verb]
                elif connection not in merged_anno_hois:
                    merged_anno_hois[connection] = anno_map[verb]

            for anno_conntection, hoi in merged_anno_hois.items():
                hbox = anno["human_bboxes"][anno_conntection[0]]
                obox = anno["human_bboxes"][anno_conntection[1]]
                for

            for anno_conntection, hoi in merged_anno_hois.items():
                obj = anno["object_labels"][anno_conntection[1]]
                if anno_conntection in merged_auto_hois and obj in object_names:
                    auto_hoi = merged_auto_hois[anno_conntection]
                    if auto_hoi == "telic":
                        auto_telic_df.loc[obj, hoi] += 1
                    elif auto_hoi == "gibsonian":
                        auto_gibs_df.loc[obj, hoi] += 1
                    else:
                        print("ERRO!")
                        exit()

    return auto_gibs_df, auto_telic_df
"""


def create_annotation_gibs_telic_differance(config):
    merged_hicodet = merge_everything(config)
    dfs = AnnoHoi_vs_AnnoOri(merged_hicodet, config, merge_pot=True, relative=True)

    diff_df = pd.DataFrame(columns=dfs["telic"].columns)
    for obj in dfs["telic"].index:
        gibs_df = dfs["gibsonian"].loc[obj, :]
        gibs_df[gibs_df < 2] = 0
        gibs_sum = gibs_df.sum()

        telic_df = dfs["telic"].loc[obj, :]
        telic_df[telic_df < 2] = 0
        telic_sum = telic_df.sum()

        gibs_df[gibs_df < 10] = 0
        gibs_df = gibs_df / gibs_sum
        gibs_df[gibs_df < 0.1] = 0
        gibs_df += 0.000001

        #print(gibs_df)
        #print("============")

        telic_df[telic_df < 10] = 0
        telic_df = telic_df / telic_sum
        telic_df[telic_df < 0.1] = 0
        telic_df += 0.000001
        #print(telic_df)
        #print("===========")
        #print(gibs_df)
        #exit()
        div_df = telic_df.div(gibs_df)
        div_df = np.log(div_df)
        #div_df[div_df < 1] = 0
        diff_df.loc[obj, :] = div_df
        #print(diff_df)

    diff_df = diff_df.transpose()
    return diff_df


def create_pred_gibs_telic_differance(config, filter=True):
    dfs = ModelHoi_vs_ModelOri(config, filter=filter, relative=True)

    diff_df = pd.DataFrame(columns=dfs["telic"].columns)
    for obj in dfs["telic"].index:
        gibs_df = dfs["gibsonian"].loc[obj, :]
        gibs_df[gibs_df < 2] = 0
        gibs_sum = gibs_df.sum()
        telic_df = dfs["telic"].loc[obj, :]
        telic_df[telic_df < 2] = 0
        telic_sum = telic_df.sum()

        gibs_df[gibs_df < 10] = 0
        gibs_df = gibs_df / gibs_sum
        gibs_df[gibs_df < 0.1] = 0
        gibs_df += 0.000001

        telic_df[telic_df < 10] = 0
        telic_df = telic_df / telic_sum
        telic_df[telic_df < 0.1] = 0
        telic_df += 0.000001
        #print(telic_df)
        #print(gibs_df)

        div_df = telic_df.div(gibs_df)
        div_df = np.log2(div_df)
        diff_df.loc[obj, :] = div_df

    diff_df = diff_df.transpose()
    return diff_df


def create_habitat_df(config):
    merged_hicodet = merge_everything(config)

    dfs = AnnoHoi_vs_AnnoOri(merged_hicodet, config, merge_pot=True, relative=True)
    #exit()
    diff_df = pd.DataFrame(columns=dfs["telic"].columns)
    for obj in dfs["telic"].index:
        print(obj)
        #obj = "car"
        gibs_df = dfs["gibsonian"].loc[obj, :]
        gibs_df = gibs_df[gibs_df > 2]
        gibs_sum = gibs_df.sum()

        #telic_df = dfs["telic"].loc[obj, :]
        #telic_sum = telic_df.sum()

        gibs_df = gibs_df / gibs_sum
        gibs_df[gibs_df < 0.2] = 0
        gibs_oris = gibs_df[gibs_df > 0.2].index
        print(gibs_oris)
        #exit()
        #gibs_df += 0.000001


    diff_df = diff_df.transpose()
    return diff_df


def objectnet3d_data_visual():
    df = pd.read_csv("data/object3d_analysis.csv", index_col=0)
    df = df[df.columns[df.sum() > 10]]
    df = df.transpose()
    df = df / df.sum()
    df = df.transpose()
    df = df[df.columns[df.sum() > 0.1]]
    sum = df.sum()
    sum.name = "Total"
    df = df.append(sum)
    df = df.transpose()
    df = df.sort_values("Total")
    df = df.transpose()
    df = df.drop("Total")
    return df


####################################################################################

def generate_browsable_csv(config):
    merged_hicodet = merge_everything(config)
    csv_data = []
    for anno_file, anno in tqdm(merged_hicodet["annotations"].items()):
        #print("==============================")
        #print(anno)
        for connection, verb in zip(anno["connections"], anno["connection_verbs"]):
            #print(connection)
            obj_name = anno["object_labels"][connection[1]]
            csv_data.append({"file": anno_file, "obj": obj_name, "verb": verb})
    df = pd.DataFrame.from_dict(csv_data)
    df.to_csv("HicoDetData.csv")
    exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--hicodet_path', default="D:/Corpora/HICO-DET", type=str)
    parser.add_argument('--processed_path', default='results_hico_merge_2015_v2_with-unary-pair-token_thresh_0_8_lines.json', type=str)
    parsed_args = parser.parse_args()

    """
    for thresh in [0.2, 0.5, 0.8, 0.9]:
        for ori in ["front-up"]: #["front", "up", "front-up"]:
            error_dfs = generate_detr_vs_ori_df(parsed_args, thresh, ori)
            graph = generate_matplot_graph(list(error_dfs.values()), error_dfs.keys(),
                                           title=f"OriAnnotation {ori} vs DETR with threshold {thresh}")
            #plt.tight_layout()
            plt.savefig(f"images/OriAnnotation {ori} vs DETR with threshold {thresh}.png", bbox_inches='tight')
            #plt.savefig(f"images/OriAnnotation {ori} vs DETR with threshold {thresh}.png")
            plt.show()
    """

    """
    df = generate_APm_DETR(parsed_args)
    generate_matplot_graph_with_line(df,
                                     ['hicodet_gibsonian_train_count', 'hicodet_telic_train_count', 'hicodet_obj_train_count'],
                                     ['gibsonian_score', 'telic_score'])
    plt.title("UPT HOI Scores")
    plt.tight_layout()
    plt.savefig("images_final/UPT HOI Scores_200.png", dpi=200)
    plt.show()
    """


    merged_hicodet = merge_everything(parsed_args)
    obj_names = ["apple", "bicycle", "bottle", "car", "chair", "cup", "dog", "horse", "knife", "umbrella"]
    for name in obj_names:
        print(name)
        annotation1, annotation2 = generate_anno_hoi_vs_auto_hoi_iaa(merged_hicodet, [name] ,parsed_args)
        #print(annotation1)
        #print(annotation2)

        score = cohen_kappa_score(annotation1, annotation2)
        print(score)
    print("Total")
    annotation1, annotation2 = generate_anno_hoi_vs_auto_hoi_iaa(merged_hicodet, obj_names, parsed_args)
    score = cohen_kappa_score(annotation1, annotation2)
    print(score)
    exit()
    """
    merged_hicodet = merge_everything(parsed_args)
    auto_gibs_df, auto_telic_df = generate_anno_hoi_vs_auto_hoi(merged_hicodet, parsed_args)
    auto_gibs_df.rename(columns={"telic": "image_telic", "gibsonian": "image_gibsonian"}, inplace=True)
    auto_telic_df.rename(columns={"telic": "image_telic", "gibsonian": "image_gibsonian"}, inplace=True)
    graph = generate_matplot_graph([auto_gibs_df, auto_telic_df], ["text_gibsonain", "text_telic"])
    plt.tight_layout()
    plt.title("Auto Affordance vs Anno Affordance")
    plt.savefig("images_final/Text Hoi vs Image Hoi_100.png", dpi=100)
    plt.show()
    """

    """
    merged_hicodet = merge_everything(parsed_args)
    for ori in ["front", "up", "front-up"]:
        for auto in [True, False]:
            if auto:
                name = "pred. affordance"
            else:
                name = "annotated affordance"

            dfs = Hoi_vs_AnnoOri(merged_hicodet, parsed_args, ori=ori, auto=auto)
            graph = generate_matplot_graph(list(dfs.values()), ["gibs", "gibs-telic", "telic"], sort_vals=True)
            plt.tight_layout()
            plt.title(f"{ori} vs {name}")
            plt.savefig(f"images/Ori {ori} vs {name}.png")
            plt.show()
    """

    """
    for ori in ["front", "up", "front-up"]:
        dfs = ModelHoi_vs_Ori(parsed_args, ori)
        graph = generate_matplot_graph(list(dfs.values()), ["gibs", "telic"])
        plt.tight_layout()
        plt.title(f"Pred Ori {ori} vs Pred Affordance")
        plt.savefig(f"images/Pred Ori {ori} vs Pred Affordance.png")
        plt.show()
    """

    """
    obj_thresh = 0.8
    aff_thresh = 0.4
    eval_df = pd.read_csv(f"data/eval_results_{obj_thresh}_{aff_thresh}.csv", index_col=0)
    graph = generate_matplot_graph([eval_df], ["eval_df"])
    plt.tight_layout()
    plt.title(f"UPT Eval Results Obj={obj_thresh}, Aff={aff_thresh}")
    plt.savefig(f"images/UPT Eval Results Obj={obj_thresh}, Aff={aff_thresh}.png")
    plt.show()
    """


    """
    filter = False
    only_test = False
    dfs = ModelHoi_vs_RelativeOri(parsed_args, filter, only_test)
    for ori in list(dfs.values())[0].head():
        ori_sum = 0
        for df in list(dfs.values()):
            ori_sum += df.loc[:, ori].sum()
        if ori_sum < 5:
            #print("drop", ori)
            for df in list(dfs.values()):
                df.drop(ori, axis=1, inplace=True)
                #print(df)

    graph = generate_matplot_graph(list(dfs.values()), ["gibs", "telic"], sort_vals=True)
    plt.tight_layout()
    plt.title(f"Pred RelativeOri vs Pred Affordance - filtered {filter}")
    plt.savefig(f"images_rel_v2/Pred RelativeOri vs Pred Affordance - filtered {filter}.png", bbox_inches='tight')
    plt.show()
    """

    """
    #create_habitat_df(parsed_args)

    diff_df = create_annotation_gibs_telic_differance(parsed_args)
    create_heatmap(diff_df, title="Heatmap", norm=(-5, 5))
    plt.tight_layout()
    plt.title(f"Annotation Gibs-Telic Heatmap")
    plt.savefig(f"images_new/Annotation Gibs-Telic Heatmap.png", bbox_inches='tight')
    plt.show()


    diff_df = create_pred_gibs_telic_differance(parsed_args)
    create_heatmap(diff_df, title="Heatmap", norm=(-5, 5))
    plt.tight_layout()
    plt.title(f"Pred Gibs-Telic Heatmap")
    plt.savefig(f"images_new/Pred Gibs-Telic Heatmap.png", bbox_inches='tight')
    plt.show()
    """

    """
    diff_df = create_annotation_gibs_telic_differance(parsed_args)
    create_heatmap(diff_df, title="Heatmap", norm=(-5, 5))
    plt.tight_layout()
    plt.title(f"Annotation Gibs-Telic Heatmap")
    plt.savefig(f"images_rel_v2/Annotation Gibs-Telic Heatmap.png", bbox_inches='tight')
    plt.show()
    """


    """
    merge_pot = True
    relative = True

    all_oris = []
    merged_hicodet = merge_everything(parsed_args)
    anno_dfs = AnnoHoi_vs_AnnoOri(merged_hicodet, parsed_args, merge_pot, relative=relative)
    for ori in list(anno_dfs.values())[0].head():
        ori_sum = 0
        for df in list(anno_dfs.values()):
            ori_sum += df.loc[:, ori].sum()
        if ori_sum < 10:
            for df in list(anno_dfs.values()):
                df.drop(ori, axis=1, inplace=True)
        else:
            #all_oris.add(ori)
            if ori not in all_oris:
                all_oris.append(ori)

    '''
    filter = True #Same Objects as Annotation
    model_dfs = ModelHoi_vs_ModelOri(parsed_args, filter, relative=relative)
    for ori in list(model_dfs.values())[0].head():
        ori_sum = 0
        for df in list(model_dfs.values()):
            ori_sum += df.loc[:, ori].sum()
        if ori_sum < 5:
            for df in list(model_dfs.values()):
                df.drop(ori, axis=1, inplace=True)
        else:
            #all_oris.add(ori)
            if ori not in all_oris:
                all_oris.append(ori)
    '''
    colors = sns.color_palette("Paired", n_colors=len(all_oris))  # Set3, Paired
    color_dict = {}
    #for ori, c in zip(sorted(list(all_oris)), colors):
    for ori, c in zip(all_oris, colors):
        color_dict[ori] = c


    graph = generate_matplot_graph(list(anno_dfs.values()), list(anno_dfs.keys()), sort_vals=True, color_dict=color_dict)
    plt.tight_layout()
    plt.title(f"Anno Relative_{relative} Orientation vs Anno Affordance")
    plt.xticks(rotation=0, fontsize=12)
    #plt.xlabel("objects", fontsize=20)
    plt.savefig(f"images_final/habitat/Anno Relative_{relative} Orientation vs Anno Affordance.png", bbox_inches='tight', dpi=400)
    plt.show()
    """




    """
    graph = generate_matplot_graph(list(model_dfs.values()), ["gibs", "telic"], sort_vals=True, color_dict=color_dict)
    plt.tight_layout()
    plt.title(f"Pred Relative_{relative} vs Pred Affordance - filtered {filter} - All")
    plt.savefig(f"images_final/habitat/Pred Relative_{relative} vs Pred Affordance - filtered {filter} - All.png", bbox_inches='tight')
    plt.show()
    """

    """
    df = objectnet3d_data_visual()
    graph = create_heatmap(df, "Object3D Dataset", cmap="flare")
    plt.tight_layout()
    #plt.title(f"Pred Relative_{relative} vs Pred Affordance - filtered {filter} - All")
    plt.savefig(f"images_final/ObjectNet3D Dist_100.png", bbox_inches='tight', dpi=100)
    plt.show()
    """

    #generate_browsable_csv(parsed_args)



