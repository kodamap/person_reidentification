import numpy as np
import cv2
import traceback
from scipy.spatial import distance


def resize_frame(frame, height):

    try:
        scale = height / frame.shape[1]
    except ZeroDivisionError as e:
        traceback.print_exc()
        return frame
    try:
        frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)
    except cv2.error as e:
        traceback.print_exc()

    return frame


def cos_similarity(X, Y):
    m = X.shape[0]
    Y = Y.T  # (m, 256) x (256, n) = (m, n)
    return np.dot(X, Y) / (
        np.linalg.norm(X.T, axis=0).reshape(m, 1) * np.linalg.norm(Y, axis=0)
    )


def get_euclidean_distance(x, Y):
    # input x: (1, 2)  center coordinate of a person frame
    #      Y: (m, 2)  center coordinate of the other person frames
    # output : (m,)
    return np.linalg.norm(x - Y, axis=1)


def get_iou2(box, boxes):
    # Input   box  : ndarray (1, 4)
    #         boxes: ndarray (m, 4)
    # Output: Iou  : ndarray (m,)
    ximin = np.maximum(box[0], boxes[:, 0])
    yimin = np.maximum(box[1], boxes[:, 1])
    ximax = np.minimum(box[2], boxes[:, 2])
    yimax = np.minimum(box[3], boxes[:, 3])
    inter_width = ximax - ximin
    inter_height = yimax - yimin
    inter_area = np.maximum(inter_width, 0) * np.maximum(inter_height, 0)
    box_ = (box[2] - box[0]) * (box[3] - box[1])
    boxes_ = (boxes[:, 2] - boxes[:, 0]) * (boxes[0:, 3] - boxes[0:, 1])
    union_area = box_ + boxes_ - inter_area
    return inter_area / union_area


def get_iou(box1: tuple, box2: tuple) -> float:
    # box: (xmin, ymin, xmax, ymax)
    # (xmin, ymin) : top left corner of the bounding box
    # (xmin, ymin) : bottom right corner of the bounding box
    ximin = max(box1[0], box2[0])
    yimin = max(box1[1], box2[1])
    ximax = min(box1[2], box2[2])
    yimax = min(box1[3], box2[3])
    # Intersection area
    inter_width = ximax - ximin
    inter_height = yimax - yimin
    inter_area = max(inter_width, 0) * max(inter_height, 0)
    # Union area
    box1_ = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_ = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_ + box2_ - inter_area
    return inter_area / union_area


def get_mahalanobis_distance(center, track_points):
    cov = np.cov(track_points.T)
    return distance.mahalanobis(track_points[-1], center, np.linalg.pinv(cov))


def affine_translation(box: tuple, top_left: tuple = (0, 0)) -> tuple:
    translation_matrix = np.eye(3)
    x = box[0] - top_left[0]
    y = box[1] - top_left[1]

    translation_matrix[0][2] = -1 * x
    translation_matrix[1][2] = -1 * y
    xmin, ymin, xmax, ymax = box
    min = np.array([xmin, ymin, 1]).reshape(3, 1)
    max = np.array([xmax, ymax, 1]).reshape(3, 1)
    min = translation_matrix @ min
    max = translation_matrix @ max
    return min[0][0], min[1][0], max[0][0], max[1][0]


def get_standard_deviation(x: list) -> tuple:
    mean = np.mean(x)
    # std = np.sqrt(np.sum((x - mean) ** 2) / len(x))
    std = np.std(x)
    return mean, std


def get_box_coordinates(prev_box, center) -> tuple:
    box_w, box_h = prev_box[2] - prev_box[0], prev_box[3] - prev_box[1]
    xmin = center[0] - (box_w / 2)
    ymin = center[1] - (box_h / 2)
    xmax = xmin + box_w
    ymax = ymin + box_h
    return xmin, ymin, xmax, ymax
