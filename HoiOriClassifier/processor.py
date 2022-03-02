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
from PoseContrast.model.model_utils import angles_to_matrix
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

    def process_image(self, img_path: str):
        #print(img_path)
        result_dict = {}

        # Load and Transform Image
        image = Image.open(img_path).convert('RGB')
        img_w, img_h = image.size
        img, _ = self.upt_transform(image, None)
        img = [img.to(self.device)]

        # Predict Human, Objects and Telic/gibsonain Interactions
        with torch.no_grad():
            output = self.upt(img)

        # Check if any objects were detected
        if len(output) > 0:
            output = output[0]
        else:
            return result_dict  # Empty

        # Write Results to Dictionary
        result_dict["boxes_scores"] = output["bscores"].detach().cpu().numpy().tolist()

        result_dict["pairing"] = output["pairing"].detach().cpu().numpy().tolist()
        result_dict["pairing_scores"] = output["scores"].detach().cpu().numpy().tolist()
        result_dict["pairing_label"] = output["labels"].detach().cpu().numpy().tolist()

        # Without that the obj label refers to the pairings which can be confusing.
        # Not a nice solution, but works for now .....
        #inx = np.array([i for i in range(len(output["objects"])) if i % 2 == 0])
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
            remapped_boxes.append([bbox[0].item() * round_w, bbox[1].item() * round_h, bbox[2].item() * round_w, bbox[3].item() * round_h])
            pil_img = image.crop((bbox[0].item() * round_w, bbox[1].item() * round_h, bbox[2].item() * round_w, bbox[3].item() * round_h))
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

        #vp_pred = vp_pred * np.pi / 180. # change degrees to radians
        #R_pred = angles_to_matrix(vp_pred)
        orientation_results = []
        for r_pred in vp_pred:
            [azi, ele, inp] = r_pred.detach().cpu().numpy()
            inp = -inp
            r_pred_rot = R.from_euler('zxy', [azi, ele, inp], degrees=True)
            #r_pred = r_pred.view(-1, 3)
            #r_pred = r_pred.detach().cpu().numpy()

            #front = r_pred.dot(self.front_vec)[self.remap_ori].tolist()
            front = r_pred_rot.apply(self.front_vec)[self.remap_ori].tolist()
            front[2] *= -1
            #up = r_pred.dot(self.up_vec)[self.remap_ori].tolist()
            up = r_pred_rot.apply(self.up_vec)[self.remap_ori].tolist()
            up[2] *= -1
            #left = r_pred.dot(self.left_vec)[self.remap_ori].tolist()
            left = r_pred_rot.apply(self.left_vec)[self.remap_ori].tolist()
            left[2] *= -1

            orientation_results.append({"front": front,
                                        "up": up,
                                        "left": left,
                                        "rotation":
                                            {"azi": azi,
                                             "ele": ele,
                                             "inp": inp}})

        result_dict["boxes_orientation"] = orientation_results
        return result_dict

    def process_images_in_folder(self, folder):
        valid_images = [".jpg", ".png"]

        if not os.path.isdir(folder):
            print(f"Inputfolder {folder} cound not be found".format(folder=folder))

        results = {}
        for file in tqdm(os.listdir(folder)):
            ext = os.path.splitext(file)[1]
            if ext.lower() in valid_images:
                img = os.path.join(folder, file)
                result = self.process_image(img)
                results[file] = result
        return results