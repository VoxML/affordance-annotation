"""
HICODet dataset under PyTorch framework

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import numpy as np
import pocket
from typing import Optional, List, Callable, Tuple

import torch
from pocket.data import ImageDataset, DataSubset


def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

class HICODetSubset(DataSubset):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def filename(self, idx: int) -> str:
        """Override: return the image file name in the subset"""
        return self._filenames[self._idx[self.pool[idx]]]

    def image_size(self, idx: int) -> Tuple[int, int]:
        """Override: return the size (width, height) of an image in the subset"""
        return self._image_sizes[self._idx[self.pool[idx]]]

    '''
    @property
    def anno_interaction(self) -> List[int]:
        """Override: Number of annotated box pairs for each interaction class"""
        num_anno = [0 for _ in range(self.num_interation_cls)]
        intra_idx = [self._idx[i] for i in self.pool]
        for idx in intra_idx:
            for hoi in self._anno[idx]['hoi']:
                num_anno[hoi] += 1
        return num_anno
    '''
    '''
    @property
    def anno_object(self) -> List[int]:
        """Override: Number of annotated box pairs for each object class"""
        num_anno = [0 for _ in range(self.num_object_cls)]
        anno_interaction = self.anno_interaction
        for corr in self._class_corr:
            num_anno[corr[1]] += anno_interaction[corr[0]]
        return num_anno
    '''
    '''
    @property
    def anno_action(self) -> List[int]:
        """Override: Number of annotated box pairs for each action class"""
        num_anno = [0 for _ in range(self.num_action_cls)]
        anno_interaction = self.anno_interaction
        for corr in self._class_corr:
            num_anno[corr[2]] += anno_interaction[corr[0]]
        return num_anno
    '''

class HICODet(ImageDataset):
    """
    Arguments:
        root(str): Root directory where images are downloaded to
        anno_file(str): Path to json annotation file
        transform(callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version
        target_transform(callable, optional): A function/transform that takes in the
            target and transforms it
        transforms (callable, optional): A function/transform that takes input sample 
            and its target as entry and returns a transformed version.
    """

    def __init__(self, root: str, anno_file: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None) -> None:
        super(HICODet, self).__init__(root, transform, target_transform, transforms)
        with open(anno_file, 'r') as f:
            anno = json.load(f)

        self.num_object_cls = 91
        self.num_action_cls = 2
        self.num_interation_cls = self.num_object_cls * self.num_action_cls
        self._anno_file = anno_file
        self.start = 0
        # Load annotations
        self._load_annotation_and_metadata(anno)

    def __len__(self) -> int:
        """Return the number of images"""
        return len(self._idx)

    def __getitem__(self, i: int) -> tuple:
        """
        Arguments:
            i(int): Index to an image
        
        Returns:
            tuple[image, target]: By default, the tuple consists of a PIL image and a
                dict with the following keys:
                    "boxes_h": list[list[4]]
                    "boxes_o": list[list[4]]
                    "hoi":: list[N]
                    "verb": list[N]
                    "object": list[N]
        """
        intra_idx = self._idx[i]
        anno = self._anno[intra_idx]
        fileid = [x for x in self._filenames[intra_idx] if x.isdigit()]
        fileid_int = int(''.join(fileid[4:]))
        anno["fileid"] = fileid_int
        return self._transforms(
            self.load_image(os.path.join(self._root, self._filenames[intra_idx])),
            anno,
        )

    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=' + repr(self._root)
        reprstr += ', anno_file='
        reprstr += repr(self._anno_file)
        reprstr += ')'
        # Ignore the optional arguments
        return reprstr

    def __str__(self) -> str:
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tImage directory: {}\n'.format(self._root)
        reprstr += '\tAnnotation file: {}\n'.format(self._root)
        return reprstr

    @property
    def annotations(self) -> List[dict]:
        return self._anno

    '''
    @property
    def class_corr(self) -> List[Tuple[int, int, int]]:
        """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]

        Returns:
            list[list[3]]
        """
        return self._class_corr.copy()
    '''


    @property   
    def object_n_verb_to_interaction(self) -> List[list]:
        lut = np.arange(0, self.num_object_cls * self.num_action_cls)
        lut = lut.reshape((-1, self.num_action_cls))
        return lut.tolist()

    '''
    @property
    def object_to_interaction(self) -> List[list]:
        """
        The interaction classes that involve each object type
        
        Returns:
            list[list]
        """
        obj_to_int = [[] for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            obj_to_int[corr[1]].append(corr[0])
        return obj_to_int
    '''

    @property
    def object_to_verb(self) -> List[list]:
        """
        The valid verbs for each object type

        Returns:
            list[list]
        """
        obj_to_verb = [list(range(self.num_action_cls)) for _ in range(self.num_object_cls)]
        #for corr in self._class_corr:
        #    obj_to_verb[corr[1]].append(corr[2])
        return obj_to_verb


    @property
    def anno_interaction(self) -> List[int]:
        """
        Number of annotated box pairs for each interaction class

        Returns:
            list[600]
        """
        return self._num_anno.copy()


    '''
    @property
    def anno_object(self) -> List[int]:
        """
        Returns:
            list
        Number of annotated box pairs for each object class
[90]
        """
        num_anno = [0 for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            num_anno[corr[1]] += self._num_anno[corr[0]]
        return num_anno
    '''

    '''
    @property
    def anno_action(self) -> List[int]:
        """
        Number of annotated box pairs for each action class

        Returns:
            list[117]
        """
        num_anno = [0 for _ in range(self.num_action_cls)]
        for corr in self._class_corr:
            num_anno[corr[2]] += self._num_anno[corr[0]]
        return num_anno
    '''

    @property
    def objects(self) -> List[str]:
        """
        Object names 

        Returns:
            list[str]
        """
        return self._objects.copy()

    @property
    def verbs(self) -> List[str]:
        """
        Verb (action) names

        Returns:
            list[str]
        """
        return self._verbs.copy()

    '''
    @property
    def interactions(self) -> List[str]:
        """
        Combination of verbs and objects

        Returns:
            list[str]
        """
        return [self._verbs[j] + ' ' + self.objects[i]
                for _, i, j in self._class_corr]
    '''

    def split(self, ratio: float) -> Tuple[HICODetSubset, HICODetSubset]:
        """
        Split the dataset according to given ratio

        Arguments:
            ratio(float): The percentage of training set between 0 and 1
        Returns:
            train(Dataset)
            val(Dataset)
        """
        perm = np.random.permutation(len(self._idx))
        n = int(len(perm) * ratio)
        return HICODetSubset(self, perm[:n]), HICODetSubset(self, perm[n:])

    def filename(self, idx: int) -> str:
        """Return the image file name given the index"""
        return self._filenames[self._idx[idx]]

    def image_size(self, idx: int) -> Tuple[int, int]:
        """Return the size (width, height) of an image"""
        return self._image_sizes[self._idx[idx]]

    def _load_annotation_and_metadata(self, f: dict) -> None:
        """
        Arguments:
            f(dict): Dictionary loaded from {anno_file}.json
        """
        #self._num_anno = num_anno

        self._anno = f['annotation']
        self._filenames = f['filenames']
        self._image_sizes = f['size']
        self._hico_object = f['objects']
        #self._class_corr = f['correspondence']
        self._empty_idx = f['empty']
        self._verbs = f['verbs']

        self.hoi_annotation = {}  # {obj#verb: T/G/-} (string)


        with open(os.path.join(os.path.dirname(self._anno_file), "textual_annotations.csv")) as file:
            for line in file:
                line = line.strip()
                splitline = line.split(";")

                if splitline[0].isdigit():
                    affordance = ""
                    if splitline[4] != "":
                        affordance = splitline[4]
                    else:
                        affordance = splitline[3]

                    #affordance = splitline[4] if splitline[4] != "" else splitline[3]
                    affordance = affordance.strip()

                    if affordance == "T":
                        self.hoi_annotation[(splitline[1].strip(), splitline[2].strip())] = 1
                    elif affordance == "G":
                        self.hoi_annotation[(splitline[1].strip(), splitline[2].strip())] = 0

        
        self.label2id = {
            "N/A": 0,
            "airplane": 5,
            "apple": 53,
            "backpack": 27,
            "banana": 52,
            "baseball bat": 39,
            "baseball glove": 40,
            "bear": 23,
            "bed": 65,
            "bench": 15,
            "bicycle": 2,
            "bird": 16,
            "blender": 83,
            "boat": 9,
            "book": 84,
            "bottle": 44,
            "bowl": 51,
            "broccoli": 56,
            "bus": 6,
            "cake": 61,
            "car": 3,
            "carrot": 57,
            "cat": 17,
            "cell phone": 77,
            "chair": 62,
            "clock": 85,
            "couch": 63,
            "cow": 21,
            "cup": 47,
            "desk": 69,
            "dining table": 67,
            "dog": 18,
            "donut": 60,
            "door": 71,
            "elephant": 22,
            "eye glasses": 30,
            "fire hydrant": 11,
            "fork": 48,
            "frisbee": 34,
            "giraffe": 25,
            "hair drier": 89,
            "handbag": 31,
            "hat": 26,
            "horse": 19,
            "hot dog": 58,
            "keyboard": 76,
            "kite": 38,
            "knife": 49,
            "laptop": 73,
            "microwave": 78,
            "mirror": 66,
            "motorcycle": 4,
            "mouse": 74,
            "orange": 55,
            "oven": 79,
            "parking meter": 14,
            "person": 1,
            "pizza": 59,
            "plate": 45,
            "potted plant": 64,
            "refrigerator": 82,
            "remote": 75,
            "sandwich": 54,
            "scissors": 87,
            "sheep": 20,
            "shoe": 29,
            "sink": 81,
            "skateboard": 41,
            "skis": 35,
            "snowboard": 36,
            "spoon": 50,
            "sports ball": 37,
            "stop sign": 13,
            "street sign": 12,
            "suitcase": 33,
            "surfboard": 42,
            "teddy bear": 88,
            "tennis racket": 43,
            "tie": 32,
            "toaster": 80,
            "toilet": 70,
            "toothbrush": 90,
            "traffic light": 10,
            "train": 7,
            "truck": 8,
            "tv": 72,
            "umbrella": 28,
            "vase": 86,
            "window": 68,
            "wine glass": 46,
            "zebra": 24
        }

        self.id2label = {
            "0": "N/A",
            "1": "person",
            "10": "traffic light",
            "11": "fire hydrant",
            "12": "street sign",
            "13": "stop sign",
            "14": "parking meter",
            "15": "bench",
            "16": "bird",
            "17": "cat",
            "18": "dog",
            "19": "horse",
            "2": "bicycle",
            "20": "sheep",
            "21": "cow",
            "22": "elephant",
            "23": "bear",
            "24": "zebra",
            "25": "giraffe",
            "26": "hat",
            "27": "backpack",
            "28": "umbrella",
            "29": "shoe",
            "3": "car",
            "30": "eye glasses",
            "31": "handbag",
            "32": "tie",
            "33": "suitcase",
            "34": "frisbee",
            "35": "skis",
            "36": "snowboard",
            "37": "sports ball",
            "38": "kite",
            "39": "baseball bat",
            "4": "motorcycle",
            "40": "baseball glove",
            "41": "skateboard",
            "42": "surfboard",
            "43": "tennis racket",
            "44": "bottle",
            "45": "plate",
            "46": "wine glass",
            "47": "cup",
            "48": "fork",
            "49": "knife",
            "5": "airplane",
            "50": "spoon",
            "51": "bowl",
            "52": "banana",
            "53": "apple",
            "54": "sandwich",
            "55": "orange",
            "56": "broccoli",
            "57": "carrot",
            "58": "hot dog",
            "59": "pizza",
            "6": "bus",
            "60": "donut",
            "61": "cake",
            "62": "chair",
            "63": "couch",
            "64": "potted plant",
            "65": "bed",
            "66": "mirror",
            "67": "dining table",
            "68": "window",
            "69": "desk",
            "7": "train",
            "70": "toilet",
            "71": "door",
            "72": "tv",
            "73": "laptop",
            "74": "mouse",
            "75": "remote",
            "76": "keyboard",
            "77": "cell phone",
            "78": "microwave",
            "79": "oven",
            "8": "truck",
            "80": "toaster",
            "81": "sink",
            "82": "refrigerator",
            "83": "blender",
            "84": "book",
            "85": "clock",
            "86": "vase",
            "87": "scissors",
            "88": "teddy bear",
            "89": "hair drier",
            "9": "boat",
            "90": "toothbrush"
        }

        self._objects = list(self.label2id.keys())


        num_anno = [0 for _ in range(self.num_interation_cls)]
        for anno_idx, anno in enumerate(self._anno):
            merged_boxes_h = []
            merged_boxes_o = []
            merged_obj = []
            merged_verb = []
            merged_hoi_list = []
            for hbox, obox, objidx, verbidx in zip(anno["boxes_h"], anno["boxes_o"], anno["object"], anno["verb"]):
                found = False

                objstr = self._hico_object[objidx]
                newobjid = self.label2id[objstr.replace("_", " ")]

                verbstr = self._verbs[verbidx]

                #if objstr != "bicycle":
                #    continue

                #if objstr != "car":
                #    continue

                #if verbstr != "wield":
                #    continue

                #if verbstr != "drive":
                #    continue

                #if objstr != "car" or verbstr != "drive":
                #    continue

                #if objstr != "book" or verbstr != "read":
                #    continue

                if (objstr.replace(" ", "_"), verbstr) not in self.hoi_annotation:
                    continue
                    
                newverbid = self.hoi_annotation[(objstr.replace(" ", "_"), verbstr)]

                for merged_idx, (merged_hbox, merged_obox, merged_o, merged_v) in enumerate(
                        zip(merged_boxes_h, merged_boxes_o, merged_obj, merged_verb)):
                    hbox_overlap = get_iou({"x1": hbox[0], "x2": hbox[2], "y1": hbox[1], "y2": hbox[3]},
                                           {"x1": merged_hbox[0], "x2": merged_hbox[2], "y1": merged_hbox[1],
                                            "y2": merged_hbox[3]})
                    obox_overlap = get_iou({"x1": obox[0], "x2": obox[2], "y1": obox[1], "y2": obox[3]},
                                           {"x1": merged_obox[0], "x2": merged_obox[2], "y1": merged_obox[1],
                                            "y2": merged_obox[3]})

                    if hbox_overlap > 0.5 and obox_overlap > 0.5 and newobjid == merged_o:
                        if merged_v < newverbid:
                            merged_verb[merged_idx] = newverbid
                            merged_hoi_list[merged_idx] = newobjid * 2 + newverbid
                            num_anno[newobjid * 2 + merged_v] -= 1
                            num_anno[newobjid * 2 + newverbid] += 1
                        found = True
                        break
                if not found:
                    merged_boxes_h.append(hbox)
                    merged_boxes_o.append(obox)
                    merged_obj.append(newobjid)
                    merged_verb.append(newverbid)
                    merged_hoi_list.append(newobjid * 2 + newverbid)
                    num_anno[newobjid * 2 + newverbid] += 1

            anno["boxes_h"] = merged_boxes_h
            anno["boxes_o"] = merged_boxes_o
            anno["object"] = merged_obj
            anno["verb"] = merged_verb
            anno["hoi"] = merged_hoi_list
            if len(merged_verb) == 0:
                f['empty'].append(anno_idx)

        idx = list(range(len(f['filenames'])))
        assert len(self._anno) == len(idx)

        for empty_idx in list(set(f['empty'])):
            idx.remove(empty_idx)

        self._idx = idx
        self._num_anno = num_anno


if __name__ == "__main__":
    os.chdir("..")
    #dataset = HICODet(
    #    root=os.path.join("../HicoDetDataset", 'hico_20160224_det/images', "test2015"),
    #    anno_file=os.path.join("../HicoDetDataset", 'instances_{}.json'.format("test2015")),
    #    target_transform=pocket.ops.ToTensor(input_format='dict')
    #)
    dataset = HICODet(
        root=os.path.join("D:/Corpora/HICO-DET", 'hico_20160224_det/images', "train2015"),
        anno_file=os.path.join("D:/Corpora/HICO-DET", 'instances_{}.json'.format("train2015")),
        target_transform=pocket.ops.ToTensor(input_format='dict')
    )
    #print(dataset.filename(4))
    print(len(dataset))
    #print(dataset.anno_interaction)
    #print(dataset.interactions)
    #print(dataset[0])
    #print("--------------")
    #print(dataset.anno_action)
    #print(dataset.anno_object)
    #print(dataset.anno_interaction)
    #print("--------------")
    #print(dataset.object_to_verb)
    #print(dataset.object_to_interaction)
    #print(dataset.object_n_verb_to_interaction)
    #print("--------------")
    #print(dataset.class_corr)

