from logging import getLogger, basicConfig, DEBUG, INFO
import os
import sys
import cv2
import numpy as np
from timeit import default_timer as timer
from libs.utils import cos_similarity, get_iou, get_distance
import pickle as pkl
import configparser

logger = getLogger(__name__)
basicConfig(
    level=INFO, format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s"
)

config = configparser.ConfigParser()
config.read("config.ini")

# Tracking parameters
reid_limit = eval(config.get("TRACKER", "reid_limit"))
sim_thld = eval(config.get("TRACKER", "sim_thld"))
min_sim_thld = eval(config.get("TRACKER", "min_sim_thld"))
reg_sim_thld = eval(config.get("TRACKER", "reg_sim_thld"))
iou_thld = eval(config.get("TRACKER", "iou_thld"))
save_points = eval(config.get("TRACKER", "save_points"))
max_grid = eval(config.get("TRACKER", "max_grid"))

# basic colors
green = eval(config.get("COLORS", "green"))
skyblue = eval(config.get("COLORS", "skyblue"))


class Tracker:
    def __init__(self, detector, frame, axis, grid):

        # initialize tracker parameters
        self.person_id_detector = detector
        self.axis = axis
        self.person_vecs = None
        self.track_points = []
        self.tracker_prev_time = timer()
        self.tracker_accum_time = 0
        self.counter_stats = ""
        self.counter_in = 0
        self.counter_out = 0
        self.counter_right = 0
        self.counter_left = 0
        self.counter_bottom = 0
        self.counter_upper = 0
        self.colors = pkl.load(open("pallete", "rb"))
        self.counter_overlap = 0
        self.params = f"@Track sim:{sim_thld} min_sim:{min_sim_thld} reg_sim:{reg_sim_thld} iou:{iou_thld} axis:{axis} grid:{grid}"
        # set tracker boundary
        self.grid, self.is_count = self._set_grid(grid)
        self.counter_range = self._set_boundary(frame, self.grid)

    def _set_grid(self, grid):
        # to couter person minimum grid need to be grater than or equal to  3
        if 3 <= grid <= max_grid:
            return grid, True
        elif grid > max_grid:
            return max_grid, True
        else:
            return max_grid, False

    def _set_boundary(self, frame, grid):
        boundary0 = frame.shape[(np.abs(self.axis - 1))] // grid
        boundary1 = frame.shape[(np.abs(self.axis - 1))] - boundary0
        # axis:0: vertical boundary line: x-axis
        # axis:1  horizontal boundary line: y-axis
        if self.axis == 0:
            # 0: left side boundary 1:right side boundary
            start0 = (boundary0, 0)
            end0 = (boundary0, frame.shape[0])
            start1 = (boundary1, 0)
            end1 = (boundary1, frame.shape[0])
        else:
            # 0: upward boundary 1: right side boundary
            start0 = (0, boundary0)
            end0 = (frame.shape[1], boundary0)
            start1 = (0, boundary1)
            end1 = (frame.shape[1], boundary1)

        counter_range = ((start0, end0), (start1, end1))

        return counter_range

    def get_feature_vecs(self, person_frames):
        feature_vecs = np.zeros((len(person_frames), 256))
        for person_id, person_frame in enumerate(person_frames):
            self.person_id_detector.infer(person_frame)
            feature_vec = self.person_id_detector.get_results()
            feature_vecs[person_id] = feature_vec
        return feature_vecs

    def register_person_vec(self, target_vec, prev_feature_vecs, box):
        """Restrict to register too many person vectors, compare the cos similarity 
        between target vector and previous feature_vectors and register the feature 
        vectors that is higher than reg_sim_threshold to get better ones.
        """
        similarity = cos_similarity(target_vec.reshape(1, 256), prev_feature_vecs)
        similarity = similarity.squeeze(axis=0)

        # if similarity is empty
        # ValueError: attempt to get argmax of an empty sequence
        if not similarity.any():
            return

        max_id = np.nanargmax(similarity)
        if similarity[max_id] > reg_sim_thld:
            self.person_vecs = np.vstack((self.person_vecs, target_vec))
            center = self.get_center_of_person(box)
            self.track_points.append([center])
            self.person_boxes.append(box)
            logger.info(
                f"registered: sim {similarity[max_id]} person_id:{len(self.person_boxes)-1}"
            )
        else:
            logger.debug(f"not registered: sim {similarity[max_id]}")

    def save_track_points(self, person_id, center, save_points=50):
        person_track_points = self.track_points[person_id]
        if len(person_track_points) > save_points:
            person_track_points.pop(0)
        person_track_points.append(center)
        self.track_points[person_id] = person_track_points

    def update_tracking(self, person_id, feature_vec, box):
        self.person_vecs[person_id] = feature_vec
        self.person_boxes[person_id] = box
        center = self.get_center_of_person(box)
        self.save_track_points(person_id, center, save_points=save_points)

    def disable_tracking(self, track_points, person_id):
        self.track_points[person_id] = [track_points[-1]]
        ##self.track_points[person_id] = [(0, 0)]
        ##self.person_vecs[person_id] = np.zeros((1, 256))
        self.person_vecs[person_id] = np.ones((1, 256))

    def is_out_of_couter_area(self, center):
        counter_range0 = self.counter_range[0][0][self.axis]
        counter_range1 = self.counter_range[1][0][self.axis]
        return center[self.axis] < counter_range0 or counter_range1 < center[self.axis]

    def person_counter(self, frame, person_id):
        """axis:
            0 x-axis: count person horizontally
            1 y-axis: count person vertically
        """
        track_points = self.track_points[person_id]
        if track_points is None or len(track_points) < 3:
            return

        start = track_points[0][self.axis]
        end = track_points[-1][self.axis]

        # axis:0 to the right direction, axis:1 to the downward direction
        counter_range = self.counter_range[1][0][self.axis]
        if start <= counter_range <= end:
            self.counter_right += 1
            self.counter_bottom += 1
            self.disable_tracking(track_points, person_id)
            logger.info(
                f"right-bottom: {person_id} ({start} <= {counter_range} <= {end})"
            )

        # axis:0 to the left direction axis:1 to the upward direction
        counter_range = self.counter_range[0][0][self.axis]
        if start >= counter_range >= end:
            self.counter_left += 1
            self.counter_upper += 1
            self.disable_tracking(track_points, person_id)
            logger.info(
                f"left-upper {person_id} ({start} >= {counter_range} >=  {end})"
            )

        if self.axis == 0:
            self.counter_stats = f"left:{self.counter_left} right:{self.counter_right}"
        elif self.axis == 0:
            self.counter_stats = (
                f"upper:{self.counter_upper} bottom:{self.counter_bottom}"
            )
        else:
            self.counter_stats = f"invalid axis:{self.axis}"

    def draw_box(self, frame, box, conf, person_id, center, color):
        xmin, ymin, xmax, ymax = box
        size = cv2.getTextSize(conf, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        xtext = xmin + size[0][0] + 20
        cv2.rectangle(
            frame, (xmin, ymin - 22), (xtext, ymin), color, -1,
        )
        cv2.rectangle(
            frame, (xmin, ymin), (xmax, ymax), color, 1,
        )
        cv2.rectangle(
            frame, (xmin, ymin - 22), (xtext, ymin), color,
        )
        cv2.putText(
            frame,
            f"{person_id} {conf}",
            (xmin + 3, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )
        return frame

    def draw_track_points(self, frame, track_points, color):
        pts = np.array(track_points)[:-1]
        cv2.polylines(
            frame, [pts], isClosed=False, color=color, thickness=2,
        )
        return frame

    def draw_couter_stats(self, frame):
        if not self.is_count:
            return frame
        start0, end0 = self.counter_range[0]
        start1, end1 = self.counter_range[1]
        cv2.line(frame, start0, end0, skyblue, 1)
        cv2.line(frame, start1, end1, skyblue, 1)
        cv2.putText(
            frame,
            self.counter_stats,
            (20, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        return frame

    def draw_params(self, frame):
        cv2.putText(
            frame,
            self.params,
            (10, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 10, 10),
            1,
        )
        return frame

    def is_overlap(self, det_id, boxes):
        """ Note:
        Treat as overlap state 'True' when two persons are exactly overlapped
        by chance(thus display as one person) to prevent unstable feature vectors from updating.
        This code allows overlap state by 5 times to be 'True' which means no update
        and no register" should be done.
        
        Todo: This is not the best way, need to be fixed with other measure.
        """
        if len(boxes) == 1:
            logger.debug(
                f"no boxes overlapped found {det_id} overlap count:{self.counter_overlap}"
            )
            if self.counter_overlap > 5:
                self.counter_overlap = 0
                return False
            else:
                self.counter_overlap += 1
                return True
        for box_id, box_ in enumerate(boxes):
            if box_id == det_id:
                continue
            iou = get_iou(boxes[det_id], box_)
            if iou > iou_thld:
                self.counter_overlap = 0
                return True
        self.counter_overlap = 0
        return False

    def person_is_matched(self, frame, person_id, box, similarity):
        center = self.get_center_of_person(box)
        conf = f"{round(similarity * 100, 1)}%"
        # 1. draw box
        frame = self.draw_box(
            frame, box, conf, person_id, center, self.colors[person_id]
        )
        # 2. draw tracking points
        if len(self.track_points[person_id]) > 2:
            frame = self.draw_track_points(
                frame, self.track_points[person_id], self.colors[person_id]
            )
        # 3. count person depends on the direction
        self.person_counter(frame, person_id)
        return frame

    def fisrt_detection(self, feature_vecs, boxes):
        self.person_vecs = feature_vecs
        self.person_boxes = boxes
        self.prev_feature_vecs = feature_vecs
        for i, box in enumerate(boxes):
            center = self.get_center_of_person(box)
            self.track_points.append([center])

    def closest_distance(self, center, track_points):
        distances = get_distance(center.reshape(1, 2), track_points)
        closest_id = np.nanargmin(distances)
        ##Get the second closest id if closest_id is out of tracking
        ##if np.count_nonzero(self.person_vecs[closest_id]) == 0:
        ##    for i in range(len(distances) - 1):
        ##       closest_id = np.argsort(distances)[i + 1]
        ##      if np.count_nonzero(self.person_vecs[closest_id]) != 0:
        ##            break
        closest_distance = distances[closest_id]
        return closest_id, closest_distance

    def get_center_of_person(self, box):
        # (cy, cy) : the center coordinate of a person
        xmin, ymin, xmax, ymax = box
        cx = int((xmax - xmin) / 2 + xmin)
        cy = int((ymax - ymin) / 2 + ymin)
        return (cx, cy)

    def person_reidentification(self, frame, persons, person_frames, boxes):

        if not person_frames:
            frame = self.draw_params(frame)
            frame = self.draw_couter_stats(frame)
            return frame

        feature_vecs = self.get_feature_vecs(person_frames[:reid_limit])

        # at the first loop
        if self.person_vecs is None:
            self.fisrt_detection(feature_vecs, boxes)

        similarities = cos_similarity(feature_vecs, self.person_vecs)
        similarities[np.isnan(similarities)] = 0
        person_ids = np.nanargmax(similarities, axis=1)

        logger.debug(
            f"--- person_reid start --- sim:{similarities} person_ids:{person_ids}"
        )

        for det_id, person_id in enumerate(person_ids):

            center = np.array(self.get_center_of_person(boxes[det_id]))

            # get the closest location of the person from track points
            track_points = np.array(
                [track_point[-1] for track_point in self.track_points]
            )
            closest_id, closest_distance = self.closest_distance(center, track_points)
            logger.debug(
                f"clst_id:{closest_id} clst_dist:{closest_distance} {closest_id} {person_id} {closest_id == person_id}"
            )

            # get cosine similarity and check if the person frames are overlapped
            similarity = similarities[det_id][person_id]
            is_overlap = self.is_overlap(det_id, boxes)

            # monitoring high similarity
            if similarity > 0.85:
                logger.debug(
                    f"det_id:{det_id} person_id:{person_id} sim:{similarity} overlap:{is_overlap}"
                )

            # 1. most likely the same person
            if similarity > sim_thld:
                logger.debug(f"update1 {det_id} to {person_id} {similarity}")
                self.update_tracking(
                    person_id, feature_vecs[det_id].reshape(1, 256), boxes[det_id]
                )
                frame = self.person_is_matched(
                    frame, person_id, boxes[det_id], similarity
                )
            # 2. nothing to do when person is out of counter area
            elif not is_overlap:
                # 3. apply minumum similarity threshold when a person is in the closest distance
                #      (Euclidean distance) with their lastet saved track points
                if similarity >= min_sim_thld and closest_id == person_id:
                    logger.debug(f"update2 {det_id} to {person_id} {similarity}")
                    self.update_tracking(
                        person_id, feature_vecs[det_id].reshape(1, 256), boxes[det_id]
                    )
                # 4. nothing to do when person is out of counter area
                elif self.is_out_of_couter_area(center):
                    continue
                # 5. finally register a new person
                else:
                    logger.debug(
                        f"register det_id:{det_id} person_id:{person_id} sim:{similarity}"
                    )
                    self.register_person_vec(
                        feature_vecs[det_id], self.prev_feature_vecs, boxes[det_id]
                    )

        self.prev_feature_vecs = feature_vecs
        frame = self.draw_couter_stats(frame)
        frame = self.draw_params(frame)

        return frame

