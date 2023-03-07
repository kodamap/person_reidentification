from argparse import ArgumentParser


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Path to video file or image. 'cam' for capturing video stream from camera",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-d",
        "--device",
        help="Specify the target device for Person Detection to infer on; CPU, GPU, FPGA or MYRIAD is acceptable.",
        default="CPU",
        choices=["CPU", "GPU", "FPGA", "MYRIAD", "AUTO"],
        type=str,
    )
    parser.add_argument(
        "-d_reid",
        "--device_reidentification",
        help="Specify the target device for person re-identificaiton to infer on; CPU, GPU, FPGA or MYRIAD \
            is acceptable.",
        default="CPU",
        choices=["CPU", "GPU", "FPGA", "MYRIAD"],
        type=str,
    )
    parser.add_argument(
        "--v4l", help="cv2.VideoCapture with cv2.CAP_V4L", action="store_true"
    )
    parser.add_argument(
        "-g",
        "--grid",
        default=0,
        help="Specify how many grid to divide frame. This is used to define boundary area when the tracker counts person. 0 ~ 2 means not counting person. (range: 3 < max_grid)",
        type=int,
    )
    parser.add_argument(
        "-v", "--verbose", help="set logging level Debug", action="store_true"
    )
    return parser
