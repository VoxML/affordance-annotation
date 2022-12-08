import json
import os
import torch
import numpy as np
from tqdm import tqdm

import utils
from UPT.upt import build_detector
import UPT.util.transforms as T
from PIL import Image
from utils import resize_pad
from PoseContrast.model.resnet import resnet50
from PoseContrast.model.vp_estimator import BaselineEstimator
from scipy.spatial.transform import Rotation as R


class ImageProcessor:
    def __init__(self, args):
        self.device = args.device

        # Load UPT Model for Object and HOI Detection
        self.upt = build_detector(args.box_score_thresh).to(self.device)
        checkpoint = torch.load(args.hoi_model)
        self.upt.load_state_dict(checkpoint['model_state_dict'])
        self.upt.eval()

        # Load PoseContrast for Object Orientation
        self.net_feat = resnet50(num_classes=128).to(self.device)
        self.net_vp = BaselineEstimator(img_feature_dim=2048).to(self.device)
        checkpoint = torch.load(args.pose_model)
        self.net_feat.load_state_dict(checkpoint['net_feat'])
        self.net_vp.load_state_dict(checkpoint['net_vp'])
        self.net_feat.eval()
        self.net_vp.eval()

        # Transformations for Image Normalisation
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.upt_transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            self.normalize,
        ])

        # For Orientation Mapping
        self.left_vec = np.array([1., 0., 0.])
        self.front_vec = np.array([0., -1., 0.])
        self.up_vec = np.array([0., 0., 1.])

        # Switch y and z axes because different conventions
        self.remap_ori = np.array([0, 2, 1])

        # Map ObjetID to TextLabel
        self.obj_id_to_label = utils.id2label

    def process_image(self, img_path: str, with_feature=False):
        # print(img_path)
        result_dict = {}

        # Load and Transform Image
        image = Image.open(img_path).convert('RGB')
        img_w, img_h = image.size
        img, _ = self.upt_transform(image, None)
        img = [img.to(self.device)]

        # Predict Human, Objects and Telic/gibsonain Interactions
        # print(img)
        with torch.no_grad():
            output = self.upt(img)
        # print(output)
        # Check if any objects were detected
        if len(output) > 0:
            output = output[0]
        else:
            return result_dict  # Empty

        # Write Results to Dictionary
        result_dict["boxes_scores"] = output["bscores"].detach().cpu().numpy().tolist()
        # print("==============================")
        # print("boxes_scores", len(result_dict["boxes_scores"]))

        if with_feature:
            result_dict["unary_token"] = output["unary_token"].detach().cpu().numpy().tolist()
            result_dict["pairwise_tokens"] = output["pairwise_tokens"].detach().cpu().numpy().tolist()
        # print("unary_token", len(result_dict["unary_token"]))
        # print("pairwise_tokens", len(result_dict["pairwise_tokens"]))

        result_dict["pairing"] = output["pairing"].detach().cpu().numpy().tolist()
        result_dict["pairing_scores"] = output["scores"].detach().cpu().numpy().tolist()
        result_dict["pairing_label"] = output["labels"].detach().cpu().numpy().tolist()
        # print("pairing", len(result_dict["pairing"]))
        # print("pairing_scores", len(result_dict["pairing_scores"]))
        # print("pairing_label", len(result_dict["pairing_label"]))
        # Without that the obj label refers to the pairings which can be confusing.
        # Not a nice soluton, but works foir now .....
        # inx = np.array([i for i in range(len(output["objects"])) if i % 2 == 0])
        oboxes_label = output["objects"].detach().cpu().numpy().tolist()
        obj_count = len(output["bscores"])
        obj_label = [1 for _ in range(obj_count)]
        for obj_idx, obj_lab in zip(result_dict["pairing"][1], oboxes_label):
            obj_label[obj_idx] = obj_lab
        result_dict["boxes_label"] = obj_label

        box_label_names = [self.obj_id_to_label[str(x)] for x in result_dict["boxes_label"]]
        result_dict["boxes_label_names"] = box_label_names

        pred_h, pred_w = output["size"]
        remapped_boxes = []
        object_img_list = []
        # Remap ImageSize
        for bbox in output["boxes"]:
            round_w = img_w / pred_w.item()
            round_h = img_h / pred_h.item()
            remapped_boxes.append([bbox[0].item() * round_w, bbox[1].item() * round_h, bbox[2].item() * round_w,
                                   bbox[3].item() * round_h])
            pil_img = image.crop((bbox[0].item() * round_w, bbox[1].item() * round_h, bbox[2].item() * round_w,
                                  bbox[3].item() * round_h))
            img = resize_pad(pil_img, 224)
            img, _ = self.normalize(img, None)
            object_img_list.append(img)
        result_dict["boxes"] = remapped_boxes


        object_img_list = torch.stack(object_img_list).to(self.device)
        # Predict Orientation for every Human and Object
        with torch.no_grad():
            feat, _ = self.net_feat(object_img_list)
            out = self.net_vp(feat)
            vp_pred = self.net_vp.compute_vp_pred(out)

        # Convert EulerAngels to RotationMatrix
        vp_pred = vp_pred.float().clone()
        vp_pred[:, 1] = vp_pred[:, 1] - 90.
        vp_pred[:, 2] = vp_pred[:, 2] - 180.

        orientation_results = []
        for r_pred, obj_label in zip(vp_pred, box_label_names):
            [azi, ele, inp] = r_pred.detach().cpu().numpy()
            inp = -inp
            r_pred_rot = R.from_euler('zxy', [azi, ele, inp], degrees=True)

            front = r_pred_rot.apply(self.front_vec)[self.remap_ori].tolist()
            front[2] *= -1
            up = r_pred_rot.apply(self.up_vec)[self.remap_ori].tolist()
            up[2] *= -1
            left = r_pred_rot.apply(self.left_vec)[self.remap_ori].tolist()
            left[2] *= -1

            if obj_label == "knife":
                front, left = left, front

            orientation_results.append({"front": front,
                                        "up": up,
                                        "left": left,
                                        "rotation":
                                            {"azi": float(azi),
                                             "ele": float(ele),
                                             "inp": float(inp)}})

        result_dict["boxes_orientation"] = orientation_results

        return result_dict

    def process_images_in_folder(self, folder, with_feature=False, save=None):
        valid_images = [".jpg", ".jpeg", ".png"]

        if not os.path.isdir(folder):
            print(f"Inputfolder {folder} cound not be found".format(folder=folder))

        if save is not None:
            outfile = open(save, 'w')

        results = {}
        results_count = 0
        for file in tqdm(os.listdir(folder)):
            ext = os.path.splitext(file)[1]
            if ext.lower() in valid_images:
                img = os.path.join(folder, file)

                try:
                    result = self.process_image(img, with_feature=with_feature)
                    results_count += 1
                    if save is not None:
                        result["filename"] = file
                        outfile.write(json.dumps(result) + "\n")
                    else:
                        results[file] = result
                except Exception as e:
                    print("Error with image:", img)
                    print(e)
                #if results_count > 5:
                #    break
        if save is not None:
            outfile.flush()
            outfile.close()
        return results
