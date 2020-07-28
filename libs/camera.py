""" ref:
https://github.com/ECI-Robotics/opencv_remote_streaming_processing/
"""

import cv2
import numpy as np
import math
import os
import sys
from logging import getLogger, basicConfig, DEBUG, INFO, ERROR
from timeit import default_timer as timer

logger = getLogger(__name__)

basicConfig(
    level=INFO, format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s"
)


class VideoCamera:
    def __init__(self, input, resize_width, v4l):
        self.resize_width = resize_width
        if input == "cam":
            self.input_stream = 0
            if v4l:
                # for Picamera, added VideoCaptureAPIs(cv2.CAP_V4L)
                self.cap = cv2.VideoCapture(self.input_stream, cv2.CAP_V4L)
            else:
                self.cap = cv2.VideoCapture(self.input_stream)
                ##self.cap = cv2.VideoCapture("http://172.25.250.212:8081/")
        else:
            self.input_stream = input
            assert os.path.isfile(input), "Specified input file doesn't exist"
            self.cap = cv2.VideoCapture(self.input_stream)

        ret, self.frame = self.cap.read()

        if ret:
            cap_prop = self._get_cap_prop()
            logger.info(
                "cap_pop:{}, resize_width:{}".format(cap_prop, self.resize_width)
            )
        else:
            logger.error(
                "Please try to start with command line parameters using --v4l if you use RaspCamera"
            )
            os._exit(1)

        if self.frame.shape[0] > self.resize_width:
            scale = self.resize_width / self.frame.shape[1]
            self.frame = cv2.resize(self.frame, dsize=None, fx=scale, fy=scale)

    def __del__(self):
        self.cap.release()

    def _get_cap_prop(self):
        return (
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            self.cap.get(cv2.CAP_PROP_FPS),
        )

    def get_frame(self, flip_code):

        ret, frame = self.cap.read()
        if frame.shape[0] > self.resize_width:
            scale = self.resize_width / frame.shape[1]
            frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)

        if ret:
            if self.input_stream == 0 and flip_code is not None:
                frame = cv2.flip(frame, int(flip_code))

            return frame
