import argparse
import json
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
from ast import literal_eval
np.random.seed(123)
random.seed(123)

from utils import visualize_embeddings
from processor import ImageProcessor

filter_names_filtered = ["bicycle", "bottle", "car", "chair", "cup", "dog", "horse", "knife", "umbrella"]

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
    for file, result in results.items():
        if "boxes_label" not in result:
            continue

        for label_id, label_name, token in zip(result["boxes_label"], result["boxes_label_names"], result["unary_token"]):
            if label_name not in count_inst:
                count_inst[label_name] = 0
            count_inst[label_name] += 1
            if label_name in filter_names_5000 and count_inst[label_name] < args.max_inst:
            #if count_inst[label_name] < args.max_inst:
                feature_vecs.append(token)
                labels.append(label_name)
    print(count_inst)

    print("Generate TSNE")
    feature_vecs = np.array(feature_vecs)

    plt = visualize_embeddings(feature_vecs, labels, tool="tsne", title=f"UPT Unary OBJ Tokens TSNE Inst_{args.max_inst} Over_5000")
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f"images_new/UPT Unary OBJ Tokens TSNE Inst_{args.max_inst} Over_5000.png", bbox_inches='tight')
    plt.show()

    print("Generate PACMAC")
    plt = visualize_embeddings(feature_vecs, labels, tool="pacmap", title=f"UPT Unary OBJ Tokens PACMAC Inst_{args.max_inst} Over_5000")
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f"images_new/UPT Unary OBJ Tokens PACMAC Inst_{args.max_inst} Over_5000.png", bbox_inches='tight')
    plt.show()


def generate_pair_token(args):
    print("Load File")
    data = []
    with open(args.processed_path, 'r') as f:
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
                #human_id = result["pairing"][0][x]
                obj_id = result["pairing"][1][x]
                gib_score = result["pairing_scores"][x]
                telic_score = result["pairing_scores"][x+1]
                pair_token = result["pairwise_tokens"][x // 2]

                label_name = boxes_label_names[obj_id]
                #label_id = boxes_label[obj_id]

                affordance = "-"
                if telic_score > gib_score and telic_score > 0.2:
                    affordance = "telic"
                if gib_score > telic_score and gib_score > 0.2:
                    affordance = "gibsonian"

                if label_name in filter_names_5000:
                    data.append({"filename": filename, "object": label_name, "affordance": affordance,
                                                  "pair_token": pair_token})
                #result_df = result_df.append(,
                #                             ignore_index=True)
            if len(data) > 500:
                break
    result_df = pd.DataFrame.from_dict(data)
    result_df.to_csv("pair_token_results-filter_names_5000_test.csv")


def visualize_pair_token(file):
    instances = 500
    subset = ["telic"]

    print("Load File")
    df = pd.read_csv(file, index_col=0, converters={"pair_token": literal_eval})

    print(df.head())
    df = df.sort_values(["object", "affordance"])

    print("Select Affordannce Subset")
    df = df.loc[df['affordance'].isin(subset)]
    classes = len(set(df["object"]))

    print("Concat and Group and SUbset Labels")
    df["label"] = df["object"] + "_" + df["affordance"]
    df = df.groupby("label").head(instances)

    print("Select Vecs and Labels")
    feature_vecs = df["pair_token"].tolist() #to
    feature_vecs = np.asarray(feature_vecs)
    print(feature_vecs)
    labels = df["label"]

    subsetname = "_".join(subset)

    print("Visualize")
    plt = visualize_embeddings(feature_vecs, labels, tool="tsne", title=f"UPT Pair Tokens TSNE {classes} Objects {subsetname}")
    plt.tight_layout()
    #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    #plt.savefig(f"images_new/UPT Pair Tokens TSNE {classes} Classes Telic.png", bbox_inches='tight')
    plt.show()

    print("Generate PACMAC")
    plt = visualize_embeddings(feature_vecs, labels, tool="pacmap", title=f"UPT Pair Tokens PACMAC {classes} Objects {subsetname}")
    plt.tight_layout()
    #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    #plt.savefig(f"images_new/UPT Pair Tokens PACMAC {classes} Classes Telic", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--hicodet_path', default="D:/Corpora/HICO-DET", type=str)
    #parser.add_argument('--processed_path', default='results_hico_merge_2015_v2_with-unary-token.json', type=str)
    parser.add_argument('--processed_path', default='results_hico_merge_2015_v2_with-unary-pair-token_thresh_0_8_lines.json', type=str)

    parser.add_argument('--max_inst', default=500, type=int)

    parsed_args = parser.parse_args()

    #visualize_unary_token(parsed_args)
    #generate_pair_token(parsed_args)
    visualize_pair_token("pair_token_results-filter_names_filtered.csv")
    #visualize_pair_token("pair_token_results-filter_names_5000_test.csv")