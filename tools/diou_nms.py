import numpy as np

def diou_nms(boxes, scores, iou_threshold):
    """
    Perform non-maximum suppression using DIoU metric.

    Args:
    - boxes: numpy array of shape (N, 4) holding N boxes, each box is represented as [x1, y1, x2, y2]
    - scores: numpy array of shape (N,) holding the scores for each box
    - iou_threshold: float, threshold for DIoU

    Returns:
    - list of indices of the boxes to keep
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # Calculate IoU
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Calculate the center distance
        center_x1 = (x1[i] + x2[i]) / 2
        center_y1 = (y1[i] + y2[i]) / 2
        center_x2 = (x1[order[1:]] + x2[order[1:]]) / 2
        center_y2 = (y1[order[1:]] + y2[order[1:]]) / 2
        inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2

        # Calculate the diagonal length of the smallest enclosing box
        outer_x1 = np.minimum(x1[i], x1[order[1:]])
        outer_y1 = np.minimum(y1[i], y1[order[1:]])
        outer_x2 = np.maximum(x2[i], x2[order[1:]])
        outer_y2 = np.maximum(y2[i], y2[order[1:]])
        outer_diag = (outer_x2 - outer_x1)**2 + (outer_y2 - outer_y1)**2

        # Calculate DIoU
        diou = iou - inter_diag / outer_diag

        inds = np.where(diou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

# Example usage
if __name__ == "__main__":
    # Example boxes and scores
    boxes = np.array([
        [100, 100, 200, 200],
        [110, 110, 210, 210],
        [120, 120, 220, 220],
        [130, 130, 230, 230],
        [200, 200, 300, 300]
    ])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

    # Perform DIoU NMS
    keep_indices = diou_nms(boxes, scores, iou_threshold=0.5)

    print("Indices of kept boxes:", keep_indices)
    print("Kept boxes:")
    print(boxes[keep_indices])
    print("Kept scores:")
    print(scores[keep_indices])