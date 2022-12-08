import argparse
import json
from tqdm import tqdm
import os
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_iou
from ast import literal_eval
np.random.seed(123)
random.seed(123)


def generate_barplot(df, x, y, hue=None, weight = False):
    if weight:
        df['counts'] = df['counts'] / df.groupby('cluster')['counts'].transform('sum')

    g = sns.catplot(
        data=df, kind="bar", x=x, y=y, hue=hue,
        palette="dark", height=6, alpha=.8
    )
    g.despine(left=True)
    g.set_axis_labels("Actions", "Count")
    g.legend.set_title("")

    return g



def merge_data(args):
    df = pd.read_csv(args.pair_data, index_col=0, converters={"hbbox": literal_eval, "obbox": literal_eval})
    df["verbs"] = None
    anno_path = os.path.join(args.hicodet_path, "instances_train2015.json")
    with open(anno_path) as json_file:
        train_data = json.load(json_file)


    verb_names = train_data["verbs"]
    obj_names = train_data["objects"]
    for filename, anno in tqdm(zip(train_data["filenames"], train_data["annotation"])):
        verbs = [verb_names[x] for x in anno["verb"]]
        objs = [obj_names[x] for x in anno["object"]]
        boxes_h = anno["boxes_h"]
        boxes_o = anno["boxes_o"]

        sub_df = df.loc[df['filename'] == filename]
        for index, row in sub_df.iterrows():
            row_obox = row["obbox"]
            row_hbox = row["hbbox"]
            row_obj = row["obj"]
            important_verbs = []
            for v, o, hbox, obox in zip(verbs, objs, boxes_h, boxes_o):
                if o == row_obj:
                    o_iou = get_iou({"x1": row_obox[0], "x2": row_obox[2], "y1": row_obox[1], "y2": row_obox[3]},
                                  {"x1": obox[0], "x2": obox[2], "y1": obox[1], "y2": obox[3]})
                    h_iou = get_iou({"x1": row_hbox[0], "x2": row_hbox[2], "y1": row_hbox[1], "y2": row_hbox[3]},
                                {"x1": hbox[0], "x2": hbox[2], "y1": hbox[1], "y2": hbox[3]})
                    if min(o_iou, h_iou) > 0.5:
                        important_verbs.append(v)
            df["verbs"][index] = np.asarray(important_verbs)



    anno_path = os.path.join(args.hicodet_path, "instances_test2015.json")
    with open(anno_path) as json_file:
        eval_data = json.load(json_file)
    verb_names = eval_data["verbs"]
    obj_names = eval_data["objects"]
    for filename, anno in tqdm(zip(eval_data["filenames"], eval_data["annotation"])):
        verbs = [verb_names[x] for x in anno["verb"]]
        objs = [obj_names[x] for x in anno["object"]]
        boxes_h = anno["boxes_h"]
        boxes_o = anno["boxes_o"]

        sub_df = df.loc[df['filename'] == filename]
        for index, row in sub_df.iterrows():
            row_obox = row["obbox"]
            row_hbox = row["hbbox"]
            row_obj = row["obj"]
            important_verbs = []
            for v, o, hbox, obox in zip(verbs, objs, boxes_h, boxes_o):
                if o == row_obj:
                    o_iou = get_iou({"x1": row_obox[0], "x2": row_obox[2], "y1": row_obox[1], "y2": row_obox[3]},
                                  {"x1": obox[0], "x2": obox[2], "y1": obox[1], "y2": obox[3]})
                    h_iou = get_iou({"x1": row_hbox[0], "x2": row_hbox[2], "y1": row_hbox[1], "y2": row_hbox[3]},
                                {"x1": hbox[0], "x2": hbox[2], "y1": hbox[1], "y2": hbox[3]})
                    if min(o_iou, h_iou) > 0.5:
                        important_verbs.append(v)
            df["verbs"][index] = np.asarray(important_verbs)
            #print(df.iloc[index])
    return df


def check_in_cluster(cluster, point):
    left_down_corner = cluster[0]
    right_up_corner = cluster[1]
    if left_down_corner[0] <= point[0] <= right_up_corner[0]:
        if left_down_corner[1] <= point[1] <= right_up_corner[1]:
            return True
    return False

def generate_cluster_stats(df, obj, clusters):
    df = df.loc[df['obj'] == obj]
    print(df)
    data = []
    for index, row in df.iterrows():
        comp_1 = row["comp-1"]
        comp_2 = row["comp-2"]
        verbs = row["verbs"]
        if verbs is None or len(verbs) == 0:
            continue

        for cluster in clusters:
            if check_in_cluster(cluster, (comp_1, comp_2)):
                for verb in verbs:
                    data.append({'cluster': cluster, 'verb': verb})

    df = pd.DataFrame.from_dict(data)
    df = df.groupby(['cluster', 'verb']).size().reset_index(name='counts')
    df = df.sort_values(["verb", "cluster"])

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hicodet_path', default="D:/Corpora/HICO-DET", type=str)
    parser.add_argument('--pair_data', default='images_new/pair_data/UPT Pair Tokens TSNE Inst_500 10 Objs Telic BBOX.csv', type=str)

    parsed_args = parser.parse_args()

    data = merge_data(parsed_args)

    weighted = True
    # UPT Pair Tokens TSNE Inst_500 10 Objs Telic.csv
    obj = "car"
    clusters = [((-40, -40), (-10, 20)), ((20, -40), (80, 0))]

    #obj = "apple"
    #clusters = [((-40, -40), (-20, 0)), ((-20, -20), (20, 0)), ((0, 0), (40, 40))]

    #obj = "dog"
    #clusters = [((-60, -60), (-35, -25)), ((-35, -40), (-10, -10)), ((0, -60), (20, -20))]

    # UPT Pair Tokens TSNE Inst_5000 10 Objs Telic.csv
    #obj = "car"
    #clusters = [((-70, -20), (-30, 60)), ((-20, 20), (0, 50)), ((-20, 50), (60, 100))]


    # UPT Pair Tokens TSNE Inst_5000 10 Objs Gibs.csv
    #obj = "bicycle"
    #clusters = [((-10, 0), (10, 40)), ((25, -20), (90, 40))]
    #obj = "horse"
    #clusters = [((-10, 0), (40, 80)), ((30, -10), (90, 40))]


    df = generate_cluster_stats(data, obj, clusters)
    graph = generate_barplot(df, x="verb", y="counts", hue="cluster", weight=weighted)
    plt.xticks(rotation=90)
    plt.title = f"Verbs in Cluster - {obj} - weighted_{weighted}"
    plt.tight_layout()

    #    plt.savefig(f"images_new/ObjectNet3D Dist.png", bbox_inches='tight')
    plt.show()
