import argparse
import json

from processor import ImageProcessor


def run(args):
    print("Init Models")
    processor = ImageProcessor(args)
    results = processor.process_images_in_folder(args.input_folder, with_feature=True, save=args.output_file)

    #with open(args.output_file, 'w') as f:
    #    json.dump(results, f, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--input_folder', default="data/test_images", type=str)
    #parser.add_argument('--input_folder', default="D:/Corpora/HICO-DET/hico_20160224_det/images/merge2015", type=str)
    parser.add_argument('--input_folder', default="E:/Corpora/ObjectNet3D/Images", type=str)

    #parser.add_argument('--output_file', default="results_anno_v2_hico_merge_2015_v2_telic-intent_with-unary-pair-token_thresh_0_8_lines_Mod_v2.json", type=str)
    #parser.add_argument('--output_file', default="results_objectnet3d_with-unary-token.json", type=str)
    parser.add_argument('--output_file',
                        default="results_objectnet3d_defaultmodel.json",
                        type=str)

    parser.add_argument('--device', default=0, type=int)

    parser.add_argument('--box_score_thresh', default=0.8, type=int)

    parser.add_argument('--hoi_model', default="data/models/robust-sweep-8_ckpt_41940_20.pt", type=str)
    #parser.add_argument('--hoi_model', default="data/models/anno-v2_ckpt_41980_20.pt", type=str)
    #parser.add_argument('--hoi_model', default="data/models/anno-v2_telic-intent_ckpt_41980_20_Mod_v2.pt", type=str)
    #parser.add_argument('--hoi_model', default="data/models/anno-ori-mod_ckpt_41940_20.pt", type=str)

    parser.add_argument('--pose_model', default="data/models/pose-model.pth", type=str)
    parsed_args = parser.parse_args()

    run(parsed_args)

