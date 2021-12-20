import copy

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
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

def merge_bboxes(annotation, threshold=0.5):
    """
    Merges BBoxes between multiple Hois
    :param annotation: json
    :param threshold: float
    :return: json; Merged Annotation with global bboxes
    """
    annotation = copy.deepcopy(annotation)
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
                        connection[1] = len(object_bboxes) - 1

        # Delete the old bboxes. connection ids now refere to the global bboxes.
        hois.pop("human_bboxes")
        hois.pop("object_bboxes")

    # Add the global bboxes to the annotation
    annotation["human_bboxes"] = human_bboxes
    annotation["object_bboxes"] = object_bboxes

    return annotation


def load_hoi_annotation(path):
    hoi_annotation = {}  # {obj#verb: T/G/-}
    with open(path) as file:
        for line in file:
            splitline = line.split()
            if len(splitline) > 3:
                hoi_annotation[(splitline[1], splitline[2])] = splitline[3]
            else:
                hoi_annotation[(splitline[1], splitline[2])] = "-"
    return hoi_annotation
