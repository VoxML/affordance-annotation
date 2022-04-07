import argparse
import json

import numpy as np

from utils import visualize_embeddings
from processor import ImageProcessor

def run(args):
    filter_names = ["apple", "bicycle", "bottle", "car", "chair", "cup", "dog", "horse", "knife", "umbrella"]
    filter_names_5000 = ["bench", "backpack", "cup", "chair", "car", "bicycle", "handbag", "bottle", "book", "boat",
                         "dining table", "tie", "motorcycle", "cell phone", "truck", "umbrella", "skateboard"]

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

    plt = visualize_embeddings(feature_vecs, labels, tool="tsne", title="UPT Unary OBJ Tokens TSNE Inst_1000 Over_5000")
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f"images_new/UPT Unary OBJ Tokens TSNE Inst_1000 Over_5000.png", bbox_inches='tight')
    plt.show()

    print("Generate PACMAC")
    plt = visualize_embeddings(feature_vecs, labels, tool="pacmap", title="UPT Unary OBJ Tokens PACMAC Inst_1000 Over_5000")
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig(f"images_new/UPT Unary OBJ Tokens PACMAC Inst_1000 Over_5000.png", bbox_inches='tight')
    plt.show()
    #if args.output_file is not None:
    #    with open(args.output_file, 'w') as f:
    #        json.dump(results, f, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--hicodet_path', default="D:/Corpora/HICO-DET", type=str)
    parser.add_argument('--processed_path', default='results_hico_merge_2015_v2_with-unary-token.json', type=str)
    parser.add_argument('--max_inst', default=1000, type=int)

    parsed_args = parser.parse_args()

    run(parsed_args)