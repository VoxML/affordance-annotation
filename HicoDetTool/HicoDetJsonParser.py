import json
import configparser
from utils.utils import get_iou


def merge_bboxes(annotation, threshold=0.7):
    """
    Merges BBoxes between multiple Hois

    Parameters
    ----------
    annotation : json
        The annotation for one image

    Returns
    -------
    annotation
        Annotation with merged global bboxes
    """
    human_bboxes = []  # List with merged human_bboxes.
    object_bboxes = []  # List with merged object_bboxes.

    for hois in annotation["hois"]:
        for h_idx, h_bbox in enumerate(hois["human_bboxes"]):
            found = False  # To check if there was a mergable bbox found.
            for ref_idx, ref_box in enumerate(human_bboxes):
                iou = get_iou({"x1": h_bbox[0], "x2": h_bbox[2], "y1": h_bbox[1], "y2": h_bbox[3]}, {"x1": ref_box[0], "x2": ref_box[2], "y1": ref_box[1], "y2": ref_box[3]})
                if iou > threshold:  # If intersection_over_union is over Threshold, Merge
                    for connection in hois["connections"]:
                        if connection[0] == h_idx:  # Change connections to global ids
                            connection[0] = ref_idx
                            found = True
                            break
            if not found:  # If no fitting bbox was found in the global list. Add this one.
                human_bboxes.append(h_bbox)
                for connection in hois["connections"]:  # Change connections to global ids
                    if connection[0] == h_idx:
                        connection[0] = len(human_bboxes) - 1

        # Do the same with the object_bboxes
        for b_idx, b_bbox in enumerate(hois["object_bboxes"]):
            found = False
            for ref_idx, ref_box in enumerate(object_bboxes):
                iou = get_iou({"x1": b_bbox[0], "x2": b_bbox[2], "y1": b_bbox[1], "y2": b_bbox[3]}, {"x1": ref_box[0], "x2": ref_box[2], "y1": ref_box[1], "y2": ref_box[3]})
                if iou > threshold:
                    for connection in hois["connections"]:
                        if connection[1] == b_idx:
                            connection[1] = ref_idx
                            found = True
                            break
            if not found:
                object_bboxes.append(b_bbox)
                for connection in hois["connections"]:
                    if connection[1] == b_idx:
                        connection[1] = len(human_bboxes) - 1

        # Delete the old bboxes. connection ids now refere to the global bboxes.
        hois.pop("human_bboxes")
        hois.pop("object_bboxes")

    # Add the global bboxes to the annotation
    annotation["human_bboxes"] = human_bboxes
    annotation["object_bboxes"] = object_bboxes

    return annotation




def print_annotation(annotation):
    print(annotation["global_id"])
    for hois in annotation["hois"]:
        print("---")
        print(hois["id"])
        print(hois["human_bboxes"])
        print(hois["object_bboxes"])


if __name__ == "__main__":
    configp = configparser.ConfigParser()
    configp.read('config.ini')
    with open(configp["HICODET"]["anno_list_json"]) as json_file:
        data = json.load(json_file)
    # print_annotation(data[0])
    # print_annotation(data[0])
    print(data[33])
    merged_anno = merge_bboxes(data[37], configp.getfloat("HICODET", "bbox_merge_threshold"))
    print(merged_anno)
        
