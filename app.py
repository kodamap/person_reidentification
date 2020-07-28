###############################################################################
#
# The MIT License (MIT)
#
# Copyright (c) 2014 Miguel Grinberg
#
# Released under the MIT license
# https://github.com/miguelgrinberg/flask-video-streaming/blob/master/LICENSE
#
###############################################################################

from flask import Flask, Response, render_template, request, jsonify, url_for
from libs.camera import VideoCamera
from logging import getLogger, basicConfig, DEBUG, INFO
import os
import sys
import cv2
import numpy as np
import json
import base64
from libs.interactive_detection import Detections, Tracker
from libs.argparser import build_argparser
from datetime import datetime
from openvino.inference_engine import get_version
import configparser

app = Flask(__name__)
logger = getLogger(__name__)

basicConfig(
    level=INFO, format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s"
)

config = configparser.ConfigParser()
config.read("config.ini")

# detection control flag
is_async = eval(config.get("DEFAULT", "is_async"))
is_det = eval(config.get("DEFAULT", "is_det"))
is_reid = eval(config.get("DEFAULT", "is_reid"))

# 0:x-axis 1:y-axis -1:both axis
flip_code = eval(config.get("DEFAULT", "flip_code"))

resize_width = int(config.get("CAMERA", "resize_width"))


def gen(camera):
    while True:
        frame = camera.get_frame(flip_code)
        frame = detections.person_detection(frame, is_async, is_det, is_reid)
        ret, jpeg = cv2.imencode(".jpg", frame)
        frame = jpeg.tostring()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


@app.route("/")
def index():
    return render_template(
        "index.html", is_async=is_async, flip_code=flip_code, enumerate=enumerate,
    )


@app.route("/video_feed")
def video_feed():
    return Response(gen(camera), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/detection", methods=["POST"])
def detection():
    global is_async
    global is_det
    global is_reid

    command = request.json["command"]
    if command == "async":
        is_async = True
    elif command == "sync":
        is_async = False

    if command == "person-det":
        is_det = not is_det
        is_reid = False
    if command == "person-reid":
        is_det = False
        is_reid = not is_reid

    result = {
        "command": command,
        "is_async": is_async,
        "flip_code": flip_code,
        "is_det": is_det,
        "is_reid": is_reid,
    }
    logger.info(
        "command:{} is_async:{} flip_code:{} is_det:{} is_reid:{}".format(
            command, is_async, flip_code, is_det, is_reid
        )
    )

    return jsonify(ResultSet=json.dumps(result))


@app.route("/flip", methods=["POST"])
def flip_frame():
    global flip_code

    command = request.json["command"]

    if command == "flip" and flip_code is None:
        flip_code = 0
    elif command == "flip" and flip_code == 0:
        flip_code = 1
    elif command == "flip" and flip_code == 1:
        flip_code = -1
    elif command == "flip" and flip_code == -1:
        flip_code = None

    result = {"command": command, "is_async": is_async, "flip_code": flip_code}
    return jsonify(ResultSet=json.dumps(result))


if __name__ == "__main__":

    # arg parse
    args = build_argparser().parse_args()
    devices = [args.device, args.device_reidentification]

    if 0 < args.grid < 3:
        print("\nargument grid must be grater than 3")
        sys.exit(1)
    camera = VideoCamera(args.input, resize_width, args.v4l)
    logger.info(
        f"input:{args.input} v4l:{args.v4l} frame shape: {camera.frame.shape} axis:{args.axis} grid:{args.grid}"
    )
    detections = Detections(camera.frame, devices, args.axis, args.grid)

    app.run(host="0.0.0.0", threaded=True)

