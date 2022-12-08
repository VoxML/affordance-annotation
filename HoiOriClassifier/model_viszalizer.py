import argparse
import json

import pandas
from tqdm import tqdm
import numpy as np
import random
import seaborn as sns
from sklearn.cluster import DBSCAN
import pandas as pd
import random
import copy
import numpy as np
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import pacmap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(123)
random.seed(123)

from utils import visualize_embeddings
from processor import ImageProcessor

#filter_names_filtered = ["bicycle", "bottle", "car", "chair", "cup", "dog", "horse", "knife", "umbrella"]

filter_names = ["apple", "bicycle", "bottle", "car", "chair", "cup", "dog", "horse", "knife", "umbrella"]
filter_names_5000 = ["bench", "backpack", "cup", "chair", "car", "bicycle", "handbag", "bottle", "book", "boat",
                     "dining table", "tie", "motorcycle", "cell phone", "truck", "umbrella", "skateboard"]


def visualize_unary_token(args):
    print("Load File")
    with open(args.processed_path, 'r') as f:
        results = json.load(f)

    feature_vecs = []
    labels = []
    count_inst = {}
    print("Prepare Data")
    filter = None
    if args.subset == "5000":
        filter = filter_names_5000
    elif args.subset == "anno":
        filter = filter_names
    else:
        print("meh")
        exit()


    for file, result in results.items():
        if "boxes_label" not in result:
            continue

        for label_id, label_name, token in zip(result["boxes_label"], result["boxes_label_names"], result["unary_token"]):
            if label_name not in count_inst:
                count_inst[label_name] = 0
            count_inst[label_name] += 1
            if label_name in filter and count_inst[label_name] < args.max_inst:
            #if count_inst[label_name] < args.max_inst:
                feature_vecs.append(token)
                labels.append(label_name)
    print(count_inst)

    print("Generate TSNE")
    feature_vecs = np.array(feature_vecs)

    plt = visualize_embeddings(feature_vecs, labels, tool="tsne", title=f"UPT Unary OBJ Tokens TSNE Inst_{args.max_inst} {args.subset}")
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f"images_new/UPT Unary OBJ Tokens TSNE Inst_{args.max_inst} {args.subset}.png", bbox_inches='tight')
    plt.show()

    print("Generate PACMAC")
    plt = visualize_embeddings(feature_vecs, labels, tool="pacmap", title=f"UPT Unary OBJ Tokens PACMAC Inst_{args.max_inst} {args.subset}")
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f"images_new/UPT Unary OBJ Tokens PACMAC Inst_{args.max_inst} {args.subset}.png", bbox_inches='tight')
    plt.show()


def visualize_pair_token(args, folder):
    print("Load File")
    with open(args.processed_path, 'r') as f:

        feature_vecs = []
        labels = []
        files = []
        h_boxes = []
        o_boxes = []
        count_inst = {}
        print("Prepare Data")
        pbar = tqdm(total=47777)
        for line in f:
            pbar.update(1)
            result = json.loads(line)
            if "boxes_label" not in result:
                continue

            boxes_label_names = result["boxes_label_names"]
            boxes_label = result["boxes_label"]
            filename = result["filename"]

            filter = None
            if args.subset == "5000":
                filter = filter_names_5000
            elif args.subset == "anno":
                filter = filter_names
            else:
                print("meh")
                exit()

            aff_filter = []
            if "Gibs" in args.aff:
                aff_filter.append("g")
            if "Telic" in args.aff:
                aff_filter.append("t")

            for x in range(0, len(result["pairing_scores"]), 2):
                human_id = result["pairing"][0][x]
                obj_id = result["pairing"][1][x]
                gib_score = result["pairing_scores"][x]
                telic_score = result["pairing_scores"][x + 1]
                pair_token = result["pairwise_tokens"][x // 2]
                h_bbox = result["boxes"][human_id]
                o_bbox = result["boxes"][obj_id]

                label_name = boxes_label_names[obj_id]
                label_id = boxes_label[obj_id]
                if label_name in filter:
                    if telic_score > gib_score and telic_score > 0.2:
                        label_name += "_t"
                    if gib_score > telic_score and gib_score > 0.2:
                        label_name += "_g"

                    if label_name != "dog" and label_name[-1] in aff_filter:
                        # and (label_name[-1] == "g" or label_name[-1] == "t"):
                        if label_name not in count_inst:
                            count_inst[label_name] = 0
                        count_inst[label_name] += 1
                        if count_inst[label_name] < args.max_inst:
                            feature_vecs.append(pair_token)
                            labels.append(label_name)
                            #labels.append(label_name[-1])
                            files.append(filename)
                            h_boxes.append(h_bbox)
                            o_boxes.append(o_bbox)
            #if len(files) > 50:
            #    break

    print("Generate TSNE")
    df = pd.DataFrame()
    df["feature_vecs"] = feature_vecs
    df["y"] = labels
    df[['obj', 'affordance']] = df.y.str.split("_", expand=True)
    df["filename"] = files
    df["hbbox"] = h_boxes
    df["obbox"] = o_boxes

    #feature_vecs = np.array(feature_vecs)
    df_copy = df.copy()
    plt, df_copy = visualize_embeddings(df_copy, tool="tsne", title=f"UPT Pair Tokens TSNE Inst_{args.max_inst} {args.subset} {args.aff}", ret_df=True)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f"{folder}/UPT Pair Tokens TSNE Inst_{args.max_inst} {args.subset} {args.aff}.png", bbox_inches='tight')
    plt.show()
    df_copy.drop("feature_vecs", axis=1)
    df_copy.to_csv(f"{folder}/pair_data/UPT Pair Tokens TSNE Inst_{args.max_inst} {args.subset} {args.aff}.csv")

    print("Generate PACMAC")
    df_copy = df.copy()
    plt, df_copy = visualize_embeddings(df_copy, tool="pacmap", title=f"UPT Pair Tokens PACMAC Inst_{args.max_inst} {args.subset} {args.aff}", ret_df=True)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f"{folder}/UPT Pair Tokens PACMAC Inst_{args.max_inst} {args.subset} {args.aff}.png", bbox_inches='tight')
    plt.show()
    df_copy.drop("feature_vecs", axis=1)
    df_copy.to_csv(f"{folder}/pair_data/UPT Pair Tokens PACMAC Inst_{args.max_inst} {args.subset} {args.aff}.csv")


def visualize_pair_token_per_obj(args, folder, tool="tsne", ):
    print("Load File")
    with open(args.processed_path, 'r') as f:

        feature_vecs = []
        labels = []
        files = []
        h_boxes = []
        o_boxes = []
        count_inst = {}
        print("Prepare Data")
        pbar = tqdm(total=47777)
        for line in f:
            pbar.update(1)
            result = json.loads(line)
            if "boxes_label" not in result:
                continue

            boxes_label_names = result["boxes_label_names"]
            boxes_label = result["boxes_label"]
            filename = result["filename"]

            filter = None
            if args.subset == "5000":
                filter = filter_names_5000
            elif args.subset == "anno":
                filter = filter_names
            else:
                print("meh")
                exit()

            for x in range(0, len(result["pairing_scores"]), 2):
                human_id = result["pairing"][0][x]
                obj_id = result["pairing"][1][x]
                gib_score = result["pairing_scores"][x]
                telic_score = result["pairing_scores"][x + 1]
                pair_token = result["pairwise_tokens"][x // 2]
                h_bbox = result["boxes"][human_id]
                o_bbox = result["boxes"][obj_id]

                label_name = boxes_label_names[obj_id]
                label_id = boxes_label[obj_id]
                if label_name in filter:
                    if telic_score > gib_score and telic_score > 0.2:
                        label_name += "_t"
                    if gib_score > telic_score and gib_score > 0.2:
                        label_name += "_g"

                    if label_name[-2:] in ["_g", "_t"]:
                        if label_name not in count_inst:
                            count_inst[label_name] = 0
                        count_inst[label_name] += 1
                        if count_inst[label_name] < args.max_inst:
                            feature_vecs.append(pair_token)
                            labels.append(label_name)
                            files.append(filename)
                            h_boxes.append(h_bbox)
                            o_boxes.append(o_bbox)
            #if len(files) > 50:
            #    break

    df = pd.DataFrame()
    df["feature_vecs"] = feature_vecs
    df["y"] = labels
    df[['obj', 'affordance']] = df.y.str.split("_", expand=True)
    df["filename"] = files
    df["hbbox"] = h_boxes
    df["obbox"] = o_boxes

    if tool == "tsne":
        model = TSNE(n_components=2, verbose=1, random_state=123)
    elif tool == "pacmap":
        model = pacmap.PaCMAP(n_dims=2, random_state=123)
    else:
        print("Could not find", tool)
        exit()
    X = np.array(df["feature_vecs"].values.tolist())
    Z = model.fit_transform(X)

    df["comp-1"] = Z[:, 0]
    df["comp-2"] = Z[:, 1]

    for obj in filter:
        print(obj)
        colormap = {}
        colormap[obj + "_g"] = obj
        colormap[obj + "_t"] = obj
        for a_obj in list(set(labels)):
            if a_obj not in colormap:
                colormap[a_obj] = "x_rest"
        df["color"] = [colormap[x] for x in df["y"]]
        df.sort_values(by="color", inplace=True, ascending=False)
        sns.scatterplot(x="comp-1", y="comp-2", hue="color", style="affordance",
                        # palette=sns.color_palette("Paired", 17), #husl, hls
                        data=df,
                        sizes=(40, 400), alpha=.75
                        # c="color"
                        ).set(title=f"UPT Pair Tokens TSNE Inst_{args.max_inst} {args.subset} {obj} ")

        plt.tight_layout()
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.savefig(f"{folder}/obj/UPT Pair Tokens TSNE Inst_{args.max_inst} {args.subset} {obj}.png",
                    bbox_inches='tight')
        plt.show()
        df_copy = df.copy()
        df_copy.drop("feature_vecs", axis=1)
        df.to_csv(f"{folder}/obj/pair_data/UPT Pair Tokens TSNE Inst_{args.max_inst} {args.subset} {obj}.csv")


def save_visualize_pair_token_clustering(args):
    print("Load File")
    with open(args.processed_path, 'r') as f:
        feature_vecs = []
        labels = []
        files = []
        h_boxes = []
        o_boxes = []
        h_unary = []
        o_unary = []
        count_inst = {}
        print("Prepare Data")
        pbar = tqdm(total=47777)
        for line in f:
            pbar.update(1)
            result = json.loads(line)
            if "boxes_label" not in result:
                continue

            boxes_label_names = result["boxes_label_names"]
            boxes_label = result["boxes_label"]
            filename = result["filename"]

            for x in range(0, len(result["pairing_scores"]), 2):

                human_id = result["pairing"][0][x]
                obj_id = result["pairing"][1][x]
                gib_score = result["pairing_scores"][x]
                telic_score = result["pairing_scores"][x + 1]
                pair_token = result["pairwise_tokens"][x // 2]
                h_bbox = result["boxes"][human_id]
                o_bbox = result["boxes"][obj_id]
                h_unary_token = result["unary_token"][human_id]
                o_unary_token = result["unary_token"][obj_id]
                label_name = boxes_label_names[obj_id]
                label_id = boxes_label[obj_id]

                if telic_score > gib_score and telic_score > 0.2:
                    label_name += "_t"
                if gib_score > telic_score and gib_score > 0.2:
                    label_name += "_g"

                if label_name[-2:] in ["_g", "_t"]:
                    if label_name not in count_inst:
                        count_inst[label_name] = 0
                    count_inst[label_name] += 1
                    if count_inst[label_name] < args.max_inst:
                        feature_vecs.append(pair_token)
                        labels.append(label_name)
                        files.append(filename)
                        h_boxes.append(h_bbox)
                        o_boxes.append(o_bbox)
                        h_unary.append(h_unary_token)
                        o_unary.append(o_unary_token)

        df = pd.DataFrame()
        df["pair_feature_vecs"] = feature_vecs
        df["h_feature_vecs"] = h_unary
        df["o_feature_vecs"] = o_unary
        df["y"] = labels
        df[['obj', 'affordance']] = df.y.str.split("_", expand=True)
        df["filename"] = files
        df["hbbox"] = h_boxes
        df["obbox"] = o_boxes
        df.to_json(f"pair_and_unary_token_pd_results.json")


def process_visualize_pair_token_clustering(args):
    print("Load DF")
    df = pd.read_json("pair_token_pd_results.json")
    subset_df = pd.DataFrame(columns=df.columns)
    for name, group in df.groupby('y'):
        subset_df = subset_df.append(group.head(500))

    print("Prepare X")
    X = np.array(subset_df.feature_vecs.values.tolist())
    print("Cluster")
    #clustering = DBSCAN(eps=3).fit(X)
    #subset_df["cluster"] = clustering.labels_

    from hdbscan import HDBSCAN
    clustering = HDBSCAN().fit_predict(X)
    subset_df["cluster"] = clustering
    print("Save Cluster Labels")

    from sklearn.manifold import TSNE
    import seaborn as sns
    import matplotlib.pyplot as plt
    model = TSNE(n_components=2, verbose=1, random_state=123)
    Z = model.fit_transform(X)

    subset_df["comp-1"] = Z[:, 0]
    subset_df["comp-2"] = Z[:, 1]
    subset_df.sort_values(by="affordance", inplace=True)
    subset_df.sort_values(by="obj", inplace=True)
    subset_df.sort_values(by="cluster", inplace=True)
    sns.scatterplot(x="comp-1", y="comp-2", hue="cluster",
                    palette=sns.color_palette("hls", len(set(clustering.labels_))), #husl, hls
                    data=subset_df,
                    sizes=(40, 400), alpha=.75
                    #c="color"
                    )
    plt.show()

def process_visualize_all_token(args):
    print("Load DF")
    df = pd.read_json("pair_and_unary_token_pd_results.json")
    subset_df = pd.DataFrame(columns=df.columns)
    for name, group in df.groupby('y'):
        print(name)
        if name[:-2] in filter_names:
            subset_df = subset_df.append(group.head(500))
    print(subset_df.size)
    print("Prepare X")
    X_pair = np.array(subset_df.pair_feature_vecs.values.tolist())
    X_h = np.array(subset_df.h_feature_vecs.values.tolist())
    X_o = np.array(subset_df.o_feature_vecs.values.tolist())
    #X = np.concatenate([X_h, X_o], axis=-1)
    #X = np.concatenate([X_pair,X_h, X_o], axis=-1)
    X = X_h

    from sklearn.manifold import TSNE
    import seaborn as sns
    import matplotlib.pyplot as plt
    model = TSNE(n_components=2, verbose=1, random_state=123)
    Z = model.fit_transform(X)

    subset_df["comp-1"] = Z[:, 0]
    subset_df["comp-2"] = Z[:, 1]

    test_df = subset_df.loc[subset_df["comp-1"] > 20]
    test_df = test_df.loc[test_df["comp-2"] > 80]
    print(test_df["filename"])
    exit()

    subset_df.sort_values(by="affordance", inplace=True)
    subset_df.sort_values(by="obj", inplace=True)
    sns.scatterplot(x="comp-1", y="comp-2", hue="affordance", #hue="obj",
                        #style="affordance",
                        data=subset_df,
                        sizes=(40, 400), alpha=.75
                        # c="color"
                        )
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--hicodet_path', default="D:/Corpora/HICO-DET", type=str)
    #parser.add_argument('--processed_path', default='results_hico_merge_2015_v2_with-unary-token.json', type=str)
    #parser.add_argument('--processed_path', default='results_hico_merge_2015_v2_with-unary-pair-token_thresh_0_8_lines.json', type=str)
    #parser.add_argument('--processed_path', default='results_anno_v2_hico_merge_2015_v2_with-unary-pair-token_thresh_0_8_lines.json', type=str)
    parser.add_argument('--processed_path',
                        default='results_anno_v2_hico_merge_2015_v2_telic-intent_with-unary-pair-token_thresh_0_8_lines.json',
                        type=str)

    parser.add_argument('--max_inst', default=5000, type=int)
    parser.add_argument('--subset', default="anno", type=str)  # anno, 5000
    parser.add_argument('--aff', default="Gibs_Telic", type=str)  # Gibs, Telic, Gibs_Telic

    parsed_args = parser.parse_args()

    #visualize_unary_token(parsed_args)

    folder = "images_anno_v2_telic-intent"
    visualize_pair_token(parsed_args, folder)
    parsed_args.max_inst = 1000
    visualize_pair_token(parsed_args, folder)
    parsed_args.max_inst = 500
    visualize_pair_token(parsed_args, folder)

    parsed_args.subset = "5000"
    parsed_args.max_inst = 5000
    visualize_pair_token(parsed_args, folder)
    parsed_args.max_inst = 1000
    visualize_pair_token(parsed_args, folder)
    parsed_args.max_inst = 500
    visualize_pair_token(parsed_args, folder)


    """
    parsed_args.subset = "anno"
    parsed_args.max_inst = 500
    visualize_pair_token_per_obj(parsed_args)
    parsed_args.max_inst = 1000
    visualize_pair_token_per_obj(parsed_args)
    parsed_args.max_inst = 5000
    visualize_pair_token_per_obj(parsed_args)
    parsed_args.max_inst = 9999999
    visualize_pair_token_per_obj(parsed_args)

    parsed_args.subset = "5000"
    parsed_args.max_inst = 500
    visualize_pair_token_per_obj(parsed_args)
    parsed_args.max_inst = 1000
    visualize_pair_token_per_obj(parsed_args)
    parsed_args.max_inst = 5000
    visualize_pair_token_per_obj(parsed_args)
    parsed_args.max_inst = 9999999
    visualize_pair_token_per_obj(parsed_args)
    """


    #save_visualize_pair_token_clustering(parsed_args)
    #process_visualize_all_token(parsed_args)

    #process_visualize_pair_token_clustering(parsed_args)
