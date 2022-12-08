import argparse
import json

from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from utils import visualize_embeddings

np.random.seed(123)
random.seed(123)



filter_names = ["apple", "bicycle", "bottle", "car", "chair", "cup", "dog", "horse", "knife", "umbrella"]
filter_names_5000 = ["bench", "backpack", "cup", "chair", "car", "bicycle", "handbag", "bottle", "book", "boat",
                     "dining table", "tie", "motorcycle", "cell phone", "truck", "umbrella", "skateboard"]


def visualize_unary_token(args):
    print("Load File")
    with open(args.processed_path, 'r') as f:
        results = json.load(f)

    list_data = []
    for file, result in results.items():
        if "boxes_label" not in result:
            continue

        for label_id, label_name, token in zip(result["boxes_label"], result["boxes_label_names"], result["unary_token"]):
            list_data.append({"feature_vec": token, "label": label_name})

    df = pd.DataFrame.from_dict(list_data)

    if args.subset == "5000":
        df = df[df["obj"].isin(filter_names_5000)]
    elif args.subset == "anno":
        df = df[df["obj"].isin(filter_names)]
    else:
        print(f"Subset: {args.subset} is not known ...")
        exit()
    filter_aff = []
    if "Gibs" in args.aff:
        filter_aff.append("g")
    if "Telic" in args.aff:
        filter_aff.append("t")
    if len(filter_aff) == 1:
        df = df[df["affordance"].isin(filter_aff)]

    if args.max_inst is not None:
        df = df.groupby("label").head(args.max_inst)
    df = df.replace("t", "telic")
    df = df.replace("g", "gibsonian")
    df_copy = df.copy()
    plt, df_copy = visualize_embeddings(df_copy, tool="tsne", title=f"UPT Pair Tokens TSNE Inst_{args.max_inst} {args.subset} {args.aff}", ret_df=True)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f"{args.img_folder}/UPT Pair Tokens TSNE Inst_{args.max_inst} {args.subset} {args.aff}.png", bbox_inches='tight')
    plt.show()
    df_copy.drop("pair_token", axis=1)
    df_copy.to_csv(f"{args.img_folder}/pair_data/UPT Pair Tokens TSNE Inst_{args.max_inst} {args.subset} {args.aff}.csv")

    print("Generate PACMAC")
    df_copy = df.copy()
    plt, df_copy = visualize_embeddings(df_copy, tool="pacmap", title=f"UPT Pair Tokens PACMAC Inst_{args.max_inst} {args.subset} {args.aff}", ret_df=True)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f"{args.img_folder}/UPT Pair Tokens PACMAC Inst_{args.max_inst} {args.subset} {args.aff}.png", bbox_inches='tight')
    plt.show()
    df_copy.drop("pair_token", axis=1)
    df_copy.to_csv(f"{args.img_folder}/pair_data/UPT Pair Tokens PACMAC Inst_{args.max_inst} {args.subset} {args.aff}.csv")



def visualize_pair_token(args):
    df = pd.read_json(args.pair_path)

    if args.subset == "5000":
        df = df[df["obj"].isin(filter_names_5000)]
    elif args.subset == "anno":
        df = df[df["obj"].isin(filter_names)]
    else:
        print(f"Subset: {args.subset} is not known ...")
        exit()
    filter_aff = []
    if "Gibs" in args.aff:
        filter_aff.append("g")
    if "Telic" in args.aff:
        filter_aff.append("t")
    if len(filter_aff) == 1:
        df = df[df["affordance"].isin(filter_aff)]


    #df = df.drop_duplicates(subset=['o_unary']) # Does not work ...
    #df = df.sample(frac=1).reset_index(drop=True)

    if args.max_inst is not None:
        df = df.groupby("label").head(args.max_inst)
    df = df.replace("t", "telic")
    df = df.replace("g", "gibsonian")
    df_copy = df.copy()
    plt, df_copy = visualize_embeddings(df_copy, tool="tsne", title=f"UPT Pair Tokens TSNE Inst_{args.max_inst} {args.subset} {args.aff}", ret_df=True)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f"{args.img_folder}/UPT Pair Tokens TSNE Inst_{args.max_inst} {args.subset} {args.aff}.png", bbox_inches='tight', dpi=400)
    plt.show()

    #df_copy.drop("pair_token", axis=1)
    #df_copy.to_csv(f"{args.img_folder}/pair_data/UPT Person Unary Tokens TSNE Inst_{args.max_inst} {args.subset} {args.aff}.csv")

    print("Generate PACMAC")
    df_copy = df.copy()
    plt, df_copy = visualize_embeddings(df_copy, tool="pacmap", title=f"UPT Pair Tokens PACMAC Inst_{args.max_inst} {args.subset} {args.aff}", ret_df=True)
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f"{args.img_folder}/UPT Pair Tokens PACMAC Inst_{args.max_inst} {args.subset} {args.aff}.png", bbox_inches='tight', dpi=400)
    plt.show()
    #df_copy.drop("pair_token", axis=1)
    #df_copy.to_csv(f"{args.img_folder}/pair_data/UPT Person Unary Tokens PACMAC Inst_{args.max_inst} {args.subset} {args.aff}.csv")


def save_pair_tokens(args):
    print("Load File")
    with open(args.processed_path, 'r') as f:
        print("Prepare Data")

        pd_data = []
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
                    data = {"filename": filename, "label": label_name, "pair_token": pair_token,
                            "h_bbox": h_bbox, "o_bbox": o_bbox,
                            "h_unary": h_unary_token, "o_unary": o_unary_token}
                    pd_data.append(data)


        df = pd.DataFrame.from_dict(pd_data)
        df[['obj', 'affordance']] = df.label.str.split("_", expand=True)
        df.to_json(args.pair_path)


def ObjectNet3DTest(args):
    print("Load File")
    count_images = 0
    count_images_with_humans = 0
    counts_per_obj = {}
    counts_per_obj_humans = {}

    df = pd.read_csv("E:/Corpora/ObjectNet3D/ObjectNet3D.txt", sep=",")
    df.index = df["im_name"]
    print(df)


    with open(args.processed_path, 'r') as f:
        print("Prepare Data")

        pd_data = []
        pbar = tqdm(total=90048)
        for line in f:

            pbar.update(1)
            count_images += 1
            result = json.loads(line)

            if "boxes_label" not in result:
                continue

            filename = result["filename"].split(".")[0]

            if filename not in df.index:
                #print(count_images)
                continue

            objnet_annos = df.loc[filename, "cls_name"]
            if isinstance(objnet_annos, str):
                objnet_annos = [objnet_annos]
            if(objnet_annos[0] == "bottle"):
                print(objnet_annos)

            boxes_label = result["boxes_label"]
            boxes_label_scores = result["boxes_scores"]

            for objnet_anno in objnet_annos:
                if objnet_anno not in counts_per_obj:
                    counts_per_obj[objnet_anno] = 0
                    counts_per_obj_humans[objnet_anno] = 0
                counts_per_obj[objnet_anno] += 1

                if boxes_label[0] == 1 and boxes_label_scores[0] > 0.9:
                    counts_per_obj_humans[objnet_anno] += 1

    for key in counts_per_obj.keys():
        print(key, float(counts_per_obj_humans[key] / counts_per_obj[key]))
    #print(counts_per_obj, boxes_label_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--hicodet_path', default="D:/Corpora/HICO-DET", type=str)
    parser.add_argument('--processed_path', default='results_hico_merge_2015_v2_with-unary-pair-token_thresh_0_8_lines.json', type=str)
    #parser.add_argument('--processed_path', default='results_anno_v2_hico_merge_2015_v2_with-unary-pair-token_thresh_0_8_lines.json', type=str)
    #parser.add_argument('--processed_path', default='results_anno_v2_hico_merge_2015_v2_telic-intent_with-unary-pair-token_thresh_0_8_lines_Mod.json', type=str)
    #parser.add_argument('--processed_path', default='results_anno-ori-mod_hico_merge_2015_v2_with-unary-pair-token_thresh_0_8_lines.json', type=str)
    #parser.add_argument('--processed_path', default='results_objectnet3d_defaultmodel.json', type=str)

    #parser.add_argument('--pair_path', default='pair_token_data/pair_token_results_original_mod.json', type=str)
    parser.add_argument('--pair_path', default='pair_token_data/pair_token_results_original.json', type=str)
    parser.add_argument('--img_folder', default='images_final/original', type=str)

    parser.add_argument('--max_inst', default=500, type=int)
    parser.add_argument('--subset', default="anno", type=str)  # anno, 5000
    parser.add_argument('--aff', default="Gibs_Telic", type=str)  # Gibs, Telic, Gibs_Telic

    parsed_args = parser.parse_args()


    #save_pair_tokens(parsed_args)
    visualize_pair_token(parsed_args)

    #ObjectNet3DTest(parsed_args)




