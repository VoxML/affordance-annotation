import random
import copy
import numpy as np
import torchvision.transforms as transforms


def get_rng_colors(count):
    colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(count)]
    return colors


def ori_dict_to_vec(ori_dict):
    vector = np.zeros(3)
    keys = ori_dict.keys()
    if "n/a" in keys or len(keys) == 0:
        return vector
    elif "+x" in keys:
        vector[0] = 1
    elif "-x" in keys:
        vector[0] = -1
    elif "+y" in keys:
        vector[2] = 1
    elif "-y" in keys:
        vector[2] = -1
    elif "+z" in keys:
        vector[1] = 1
    elif "-z" in keys:
        vector[1] = -1
    else:
        print(ori_dict)
        print("!!!!!!!!!!!!!!!!!")
    return vector


def resize_pad(im, dim):
    w, h = im.size
    im = transforms.functional.resize(im, int(dim * min(w, h) / max(w, h)))
    left = int(np.ceil((dim - im.size[0]) / 2))
    right = int(np.floor((dim - im.size[0]) / 2))
    top = int(np.ceil((dim - im.size[1]) / 2))
    bottom = int(np.floor((dim - im.size[1]) / 2))
    im = transforms.functional.pad(im, (left, top, right, bottom))
    return im


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


def merge_bboxes(annotation, object_names, verb_names, threshold=0.5):
    """
    Merges BBoxes between multiple Hois
    :param annotation: json
    :param threshold: float
    :return: json; Merged Annotation with global bboxes
    """
    #annotation = copy.deepcopy(annotation)
    human_bboxes = []  # List with merged human_bboxes.
    object_bboxes = []  # List with merged object_bboxes.
    object_labels = []
    connections = []
    connection_verbs = []


    for h_bbox, o_bbox, obj, verb in zip(annotation["boxes_h"], annotation["boxes_o"], annotation["object"], annotation["verb"]):
        human_id = -1
        for ref_idx, ref_box in enumerate(human_bboxes):
            iou = get_iou({"x1": h_bbox[0], "x2": h_bbox[2], "y1": h_bbox[1], "y2": h_bbox[3]}, {"x1": ref_box[0], "x2": ref_box[2], "y1": ref_box[1], "y2": ref_box[3]})
            if iou > threshold:  # If intersection_over_union is over Threshold, Merge
                human_id = ref_idx
                break
        if human_id < 0:
            human_bboxes.append(h_bbox)
            human_id = len(human_bboxes) - 1

        obj_id = -1
        for ref_idx, (ref_box, ref_box_lab) in enumerate(zip(object_bboxes, object_labels)):
            iou = get_iou({"x1": o_bbox[0], "x2": o_bbox[2], "y1": o_bbox[1], "y2": o_bbox[3]}, {"x1": ref_box[0], "x2": ref_box[2], "y1": ref_box[1], "y2": ref_box[3]})
            if iou > threshold and ref_box_lab == obj:  # If intersection_over_union is over Threshold, Merge
                obj_id = ref_idx
                break
        if obj_id < 0:
            object_bboxes.append(o_bbox)
            object_labels.append(obj)
            obj_id = len(object_bboxes) - 1

        connections.append((human_id, obj_id))
        connection_verbs.append(verb)

    new_format_annotation = {}
    new_format_annotation["human_bboxes"] = human_bboxes
    new_format_annotation["object_bboxes"] = object_bboxes
    new_format_annotation["object_labels"] = [object_names[x] for x in object_labels]
    new_format_annotation["connections"] = connections
    new_format_annotation["connection_verbs"] = [verb_names[x] for x in connection_verbs]

    return new_format_annotation


colors = ["#000000", "#FF3300", "#4ade80", "#facc15", "#60a5fa", "#fb923c", "#c084fc", "#22d3ee", "#a3e635", "#663300",
          "#00FFFF", "#3300FF", "#66CC99", "#FFFF33", "#CC6600", "#999933", "#006666", "#CC00FF", "#FF99CC", "#f87171",
          "#999999"] * 50


id2label = {
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