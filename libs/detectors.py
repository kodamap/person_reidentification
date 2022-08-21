import cv2
import os
import sys
from logging import getLogger

from openvino.inference_engine import IECore

# OpenVINO 2021: The IEPlugin class is deprecated
try:
    from openvino.inference_engine import IEPlugin
except ImportError:
    pass
from openvino.inference_engine import get_version

import numpy as np

logger = getLogger(__name__)

is_myriad_plugin_initialized = False
myriad_plugin = None


class BaseDetection(object):
    def __init__(self, device, model_xml, detection_of):
        """MYRIAD device's plugin should be initialized only once, 
        MYRIAD plugin would be failed when creating exec_net 
        RuntimeError: Can not init Myriad device: NC_ERROR
        """

        self.ie = IECore()
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        net = self.ie.read_network(model=model_xml, weights=model_bin)
        # Load IR model to the plugin
        logger.info("Reading IR for {}...".format(detection_of))
        self._load_ir_to_plugin(device, net, detection_of)

    def _load_ir_to_plugin(self, device, net, detection_of):

        global is_myriad_plugin_initialized
        global myriad_plugin

        if detection_of == "Person Detection":
            logger.info("Checking Person Detection network inputs")
            assert (
                len(net.input_info.keys()) == 1
            ), "Person Detection network should have only one input"
            assert (
                len(net.outputs) == 1
            ), "Person Detection network should have only one output"

        self.input_blob = next(iter(net.input_info))
        self.out_blob = next(iter(net.outputs))

        # Loading model to the plugin
        logger.info(
            f"Loading {device} model to the {detection_of} plugin... version:{get_version()}"
        )
        # Example: version: 2021.4.2-3974-e2a469a3450-releases/2021/4
        # The IEPlugin class is deprecated
        if str(get_version().split("-")[0]) > "2021":
            self.exec_net = self.ie.load_network(
                network=net, device_name=device, num_requests=2
            )
        else:
            if device == "MYRIAD" and not is_myriad_plugin_initialized:
                # To prevent MYRIAD Plugin from initializing failed, use IEPlugin Class which is deprecated
                # "RuntimeError: Can not init Myriad device: NC_ERROR"
                self.plugin = IEPlugin(device=device, plugin_dirs=None)
                self.exec_net = self.plugin.load(network=net, num_requests=2)
                is_myriad_plugin_initialized = True
                myriad_plugin = self.plugin
            elif device == "MYRIAD" and is_myriad_plugin_initialized:
                logger.info(f"device plugin for {device} already initialized")
                self.plugin = myriad_plugin
                self.exec_net = self.plugin.load(network=net, num_requests=2)
            else:
                self.exec_net = self.ie.load_network(
                    network=net, device_name=device, num_requests=2
                )

        self.input_dims = net.input_info[self.input_blob].input_data.shape
        self.output_dims = net.outputs[self.out_blob].shape
        logger.info(
            f"{detection_of} input dims:{self.input_dims} output dims:{self.output_dims}"
        )


class PersonDetection(BaseDetection):
    def __init__(self, device, model_xml):
        detection_of = "Person Detection"
        super().__init__(device, model_xml, detection_of)

        self.cur_request_id = 0
        self.next_request_id = 1

    def infer(self, frame, next_frame, is_async):
        n, c, h, w = self.input_dims

        if is_async:
            # logger.debug(
            #    "*** start_async *** cur_req_id:{} next_req_id:{} async:{}".format(
            #        self.cur_request_id, self.next_request_id, is_async
            #    )
            # )
            in_frame = cv2.resize(next_frame, (w, h))
            # Change data layout from HWC to CHW
            in_frame = in_frame.transpose((2, 0, 1))
            in_frame = in_frame.reshape((n, c, h, w))
            self.exec_net.start_async(
                request_id=self.next_request_id, inputs={self.input_blob: in_frame}
            )
        else:
            # logger.debug(
            #    "*** start_sync *** cur_req_id:{} next_req_id:{} async:{}".format(
            #        self.cur_request_id, self.next_request_id, is_async
            #    )
            # )
            self.exec_net.requests[self.cur_request_id].wait(-1)
            in_frame = cv2.resize(frame, (w, h))
            # Change data layout from HWC to CHW
            in_frame = in_frame.transpose((2, 0, 1))
            in_frame = in_frame.reshape((n, c, h, w))
            self.exec_net.start_async(
                request_id=self.cur_request_id, inputs={self.input_blob: in_frame}
            )

    def get_results(self, is_async, prob_threshold_person):
        """
        The net outputs a blob with shape: [1, 1, 200, 7]
        The description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        """
        persons = None
        if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
            # res's shape: [1, 1, 200, 7]
            res = (
                self.exec_net.requests[self.cur_request_id]
                .output_blobs[self.out_blob]
                .buffer
            )
            # Get rows whose confidence is larger than prob_threshold.
            # detected persons are also used by age/gender, emotion, landmark, head pose detection.
            persons = res[0][:, np.where(res[0][0][:, 2] > prob_threshold_person)]

        if is_async:
            self.cur_request_id, self.next_request_id = (
                self.next_request_id,
                self.cur_request_id,
            )

        return persons


class PersonReIdentification(BaseDetection):
    def __init__(self, device, model_xml):
        detection_of = "Person re-identifications"
        super().__init__(device, model_xml, detection_of)

    def infer(self, person_frame):
        n, c, h, w = self.input_dims
        in_frame = cv2.resize(person_frame, (w, h))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((n, c, h, w))
        self.exec_net.infer(inputs={self.input_blob: in_frame})

    def get_results(self):
        """
        person-reidentification-retail-0031:
        person-reidentification-retail-0248:
        Output:
            The net outputs a blob with the [1, 256, 1, 1] shape named descriptor, 
            which can be compared with other descriptors using the cosine distance.
        """
        res = self.exec_net.requests[0].output_blobs[self.out_blob].buffer
        feature_vec = res.reshape(1, 256)
        return feature_vec

