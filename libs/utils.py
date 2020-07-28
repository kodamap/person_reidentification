import numpy as np
import cv2
import traceback


def get_person_frames(persons, frame):

    frame_h, frame_w = frame.shape[:2]
    person_frames = []
    boxes = []

    for person_id, person in enumerate(persons[0][0]):
        box = person[3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
        (xmin, ymin, xmax, ymax) = box.astype("int")
        boxes.append((xmin, ymin, xmax, ymax))
        person_frame = frame[ymin:ymax, xmin:xmax]
        person_h, person_w = person_frame.shape[:2]
        # Resizing person_frame will be failed when witdh or height of the person_fame is 0
        # ex. (243, 0, 3)
        if person_h != 0 and person_w != 0:
            person_frames.append(person_frame)
    return person_frames, boxes


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


def get_distance(x, Y):
    # input x: (1, 2)  center coordinate of a person frame
    #       Y: (m, 2)  center coordinate of the other person frames
    # output : (m,)
    return np.abs(np.linalg.norm(x - Y, axis=1))


def get_iou(box1, box2):
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

