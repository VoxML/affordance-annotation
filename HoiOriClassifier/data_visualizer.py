import os
import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

from tqdm import tqdm
from utils import get_iou, ori_dict_to_vec


def generate_altair_graph(dfs, names):
    # https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars-with-python-pandas
    def prep_df(df, name):
        df = df.stack().reset_index()
        df.columns = ['c1', 'c2', 'values']
        df['DF'] = name
        return df

    prepared_dfs = []
    for df, name in zip(dfs, names):
        prepared_dfs.append(prep_df(df, name))

    df = pd.concat(prepared_dfs)

    alt.Chart(df).mark_bar().encode(
        # tell Altair which field to group columns on
        x=alt.X('c2:N', title=None),

        # tell Altair which field to use as Y values and how to calculate
        y=alt.Y('sum(values):Q',
                axis=alt.Axis(
                    grid=False,
                    title=None)),

        # tell Altair which field to use to use as the set of columns to be  represented in each group
        column=alt.Column('c1:N', title=None),

        # tell Altair which field to use for color segmentation
        color=alt.Color('DF:N',
                        scale=alt.Scale(
                            # make it look pretty with an enjoyable color pallet
                            range=['#96ceb4', '#ffcc5c', '#ff6f69'],
                        ),
                        )) \
        .configure_view(
        # remove grid lines around column clusters
        strokeOpacity=0
    )



def convert_df_to_long(df, factor=0.9):
    df = df.drop("person")
    df.loc[:, 'obj'] = df.index

    df_filtered = df[["obj", "hico_det_count", f"detr_count_{factor}"]]
    df_filtered_long = pd.melt(df_filtered, id_vars=["obj"], value_vars=["hico_det_count", f"detr_count_{factor}"])
    df_filtered_long.loc[:, "stack"] = 0

    df_filtered = df.loc[:, ("obj", f"detr_count_{factor}_all")]
    df_filtered.loc[:, "stack"] = 1
    df_filtered.loc[:, "variable"] = f"detr_count_{factor}"

    df_filtered = df_filtered.rename(columns={f"detr_count_{factor}_all": "value"})

    df_filtered_2 = df[["obj"]]  # SettingWithCopyWarning is here.
    df_filtered_2.loc[:, "stack"] = 1
    df_filtered_2.loc[:, "value"] = 0
    df_filtered_2.loc[:, "variable"] = "hico_det_count"

    df_filtered_long = pd.concat([df_filtered_long, df_filtered, df_filtered_2])

    df_filtered_long = df_filtered_long.sort_values(["variable", "value"], ascending=False)

    return df_filtered_long


def generate_sns(df):
    #https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars-with-python-pandas
    factor = 0.2
    df = convert_df_to_long(df, factor)
    sns.set(rc={'figure.figsize': (30, 10)})
    for i, g in enumerate(df.groupby("stack")):
        color = None
        if i > 0:
            color = "orange"

        sns.barplot(data=g[1], x="obj", y="value", hue="variable", color=color, zorder=-i, edgecolor="k")

    plt.xticks(rotation=90)
    plt.title(f"HicoDet Annotation vs DETR with threshold {factor}")
    plt.tight_layout()
    plt.savefig(f"images/HicoDet Annotation vs DETR with threshold {factor}.png")
    plt.show()

def generate_detr_hico_sns(df):
    #https://stackoverflow.com/questions/22787209/how-to-have-clusters-of-stacked-bars-with-python-pandas
    df = df.groupby(["obj", "front", "det_type"])["value"].sum().reset_index()
    print(df)
    sns.set(rc={'figure.figsize': (30, 10)})
    for i, g in enumerate(df.groupby("det_type")):
        #print("===")
        #print(g[1])
        #print(g[1].colums)
        #print(g[1].index)
        sns.barplot(data=g[1], x="obj", y="value", hue="front", zorder=-i, edgecolor="k")

    plt.xticks(rotation=90)
    plt.title(f"...........")
    plt.show()


def generate_detr_vs_hicodet_df(config):
    anno_path = os.path.join(config.hicodet_path, "instances_train2015.json")
    with open(anno_path) as json_file:
        train_data = json.load(json_file)

    with open(config.processed_path) as json_file:
        pred_data = json.load(json_file)

    object_names = train_data["objects"]
    object_names = [object_name.replace(" ", "_") for object_name in object_names]
    df = pd.DataFrame(0, index=object_names,
                      columns=["hico_det_count", "detr_count_0.2", "detr_count_0.5", "detr_count_0.8", "detr_count_0.9",
                               "detr_count_0.2_all", "detr_count_0.5_all", "detr_count_0.8_all", "detr_count_0.9_all"])

    for anno in train_data["annotation"]:
        for obj in anno["object"]:
            df["hico_det_count"][obj] += 1

    for anno, img in tqdm(zip(train_data["annotation"], train_data["filenames"]), total=len(train_data["filenames"])):
        bboxes = anno["boxes_h"] + anno["boxes_o"]
        labels = [0 for _ in range(len(anno["boxes_h"]))] + anno["object"]

        preds = pred_data[img]
        if len(preds) == 0:
            continue

        for pred_bbox, pred_label, pred_bbox_score in zip(preds["boxes"], preds["boxes_label_names"], preds["boxes_scores"]):
            found = False
            pred_label = pred_label.replace(" ", "_")
            for bbox, label in zip(bboxes, labels):
                iou = get_iou({"x1": pred_bbox[0], "x2": pred_bbox[2], "y1": pred_bbox[1], "y2": pred_bbox[3]},
                              {"x1": bbox[0], "x2": bbox[2], "y1": bbox[1], "y2": bbox[3]})

                if iou > 0.5 and pred_label == object_names[label]:
                    if pred_bbox_score > 0.9:
                        df["detr_count_0.9"][labels] += 1
                        df["detr_count_0.9_all"][labels] += 1
                    if pred_bbox_score > 0.8:
                        df["detr_count_0.8"][labels] += 1
                        df["detr_count_0.8_all"][labels] += 1
                    if pred_bbox_score > 0.5:
                        df["detr_count_0.5"][labels] += 1
                        df["detr_count_0.5_all"][labels] += 1
                    if pred_bbox_score > 0.2:
                        df["detr_count_0.2"][labels] += 1
                        df["detr_count_0.2_all"][labels] += 1
                    found = True
                    break
            if not found:
                if pred_label not in object_names:
                    df.loc[pred_label] = 0

                if pred_bbox_score > 0.9:
                    df["detr_count_0.9_all"][pred_label] += 1
                if pred_bbox_score > 0.8:
                    df["detr_count_0.8_all"][pred_label] += 1
                if pred_bbox_score > 0.5:
                    df["detr_count_0.5_all"][pred_label] += 1
                if pred_bbox_score > 0.2:
                    df["detr_count_0.2_all"][pred_label] += 1


    df.to_csv("object_detection.csv")
    print(df)


def generate_detr_vs_ori_df(config):
    anno_path = os.path.join(config.hicodet_path, "via234_1200 items_train verified.json")
    with open(anno_path) as json_file:
        anno_data = json.load(json_file)

    with open(config.processed_path) as json_file:
        pred_data = json.load(json_file)

    df = pd.DataFrame(columns=["obj", "front", "up", "det_type", "value"])
    for _, annotation in tqdm(anno_data["_via_img_metadata"].items()):
        filename = annotation["filename"]
        preds = pred_data[filename]

        for region in annotation["regions"]:
            bbox = [region["shape_attributes"]["x"], region["shape_attributes"]["y"],
                    region["shape_attributes"]["x"] + region["shape_attributes"]["width"],
                    region["shape_attributes"]["y"] + region["shape_attributes"]["height"]]

            front_vec = ori_dict_to_vec(region["region_attributes"]["front"])
            up_vec = ori_dict_to_vec(region["region_attributes"]["up"])

            if "category" not in region["region_attributes"]:
                continue
            elif region["region_attributes"]["category"] == "human":
                name = "person"
            elif region["region_attributes"]["category"] == "object":
                name = region["region_attributes"]["obj name"].replace(" ", "_")
            else:
                print("???????????????????")
                exit()
            #if f"{front_vec}_{up_vec}_det" not in df.index:
            #    df.loc[f"{front_vec}_{up_vec}_det", :] = 0
            #    df.loc[f"{front_vec}_{up_vec}_ndet", :] = 0
            #    df.loc[f"{front_vec}_{up_vec}_wdet", :] = 0

            #if name not in df.columns:
            #    df.loc[:, name] = 0

            found = False
            found_other = False
            for pred_bbox, pred_label, pred_bbox_score in zip(preds["boxes"], preds["boxes_label_names"], preds["boxes_scores"]):
                pred_label = pred_label.replace(" ", "_")

                iou = get_iou({"x1": pred_bbox[0], "x2": pred_bbox[2], "y1": pred_bbox[1], "y2": pred_bbox[3]},
                                {"x1": bbox[0], "x2": bbox[2], "y1": bbox[1], "y2": bbox[3]})

                if iou > 0.5 and pred_label == name:
                    found = True
                    df = df.append({"obj": name, "front": str(front_vec), "up": str(up_vec), "det_type": "correct", "value": 1}, ignore_index=True)
                    #df.loc[f"{front_vec}_{up_vec}_det", pred_label] += 1
                    break
                elif iou > 0.5 and pred_label != name:
                    found_other = True

            if not found and not found_other:
                df = df.append({"obj": name, "front": str(front_vec), "up": str(up_vec), "det_type": "wrong_det", "value": 1}, ignore_index=True)
            elif not found:
                df = df.append({"obj": name, "front": str(front_vec), "up": str(up_vec), "det_type": "not_det", "value": 1}, ignore_index=True)
                #df.loc[f"{front_vec}_{up_vec}_ndet", name] += 1

                #df.loc[f"{front_vec}_{up_vec}_wdet", name] += 1
    df["orientation"] = df["up"] + df["front"]
    print(df)
    return df
    #df = df.groupby(["obj", "front", "det_type"])["value"].sum()
    #df = df.transpose()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--hicodet_path', default="D:/Corpora/HICO-DET", type=str)
    parser.add_argument('--processed_path', default='results_hico_merge_2015.json', type=str)
    parsed_args = parser.parse_args()

    #generate_detr_vs_hicodet_df(parsed_args)
    pandas_df = pd.read_csv("object_detection.csv", index_col=0)
    generate_sns(pandas_df)

    # OUTDATED!!! (see v2)
    #pandas_df = generate_detr_vs_ori_df(parsed_args)
    #generate_detr_hico_sns(pandas_df)


