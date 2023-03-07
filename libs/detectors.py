import cv2
import numpy as np
from logging import getLogger

from openvino.runtime import Core
import openvino.runtime as ov
from openvino.runtime import get_version

logger = getLogger(__name__)


class BaseDetection(object):
    def __init__(self, device, model_xml, detection_of):

        ie = Core()
        # read the network and corresponding weights from file
        model = ie.read_model(model=model_xml)

        # compile the model for the CPU (you can choose manually CPU, GPU, MYRIAD etc.)
        # or let the engine choose the best available device (AUTO)
        self.compiled_model = ie.compile_model(model=model, device_name=device)

        # get input node
        self.input_layer_ir = model.input(0)
        n, c, h, w = self.input_layer_ir.shape
        self.shape = (w, h)
        logger.info(
            f"Loading {device} model to the {detection_of} ... version:{get_version()}"
        )

    def preprocess(self, frame):
        """
         Define the preprocess function for input data

        :param: image: the orignal input frame
        :returns:
                resized_image: the image processed
        """
        resized_frame = cv2.resize(frame, self.shape)
        resized_frame = cv2.cvtColor(
            np.array(resized_frame), cv2.COLOR_BGR2RGB)
        resized_frame = resized_frame.transpose((2, 0, 1))
        resized_frame = np.expand_dims(
            resized_frame, axis=0).astype(np.float32)
        return resized_frame


class PersonDetection(BaseDetection):
    def __init__(self, device, model_xml):
        detection_of = "Person Detection"
        super().__init__(device, model_xml, detection_of)

        # Create 2 infer requests
        self.curr_request = self.compiled_model.create_infer_request()
        self.next_request = self.compiled_model.create_infer_request()

    def infer(self, frame, next_frame, is_async):
        """
        Ref: async api
        https://github.com/openvinotoolkit/openvino_notebooks/blob/2022.1/notebooks/115-async-api/115-async-api.ipynb
        """

        if is_async:
            resized_frame = self.preprocess(next_frame)
            self.next_request.set_tensor(
                self.input_layer_ir, ov.Tensor(resized_frame))
            # Start the "next" inference request
            self.next_request.start_async()

        else:
            self.curr_request.wait_for(-1)
            resized_frame = self.preprocess(frame)
            self.curr_request.set_tensor(
                self.input_layer_ir, ov.Tensor(resized_frame))
            # Start the current inference request
            self.curr_request.start_async()

    def get_results(self, is_async, prob_threshold_person):
        """
        The net outputs a blob with shape: [1, 1, 200, 7]
        The description has the format: [image_id, label, conf, x_min, y_min, x_max, y_max]
        """
        persons = None
        if self.curr_request.wait_for(-1) == 1:
            res = self.curr_request.get_output_tensor(0).data
            # res's shape: [1, 1, 200, 7]
            persons = res[0][:, np.where(
                res[0][0][:, 2] > prob_threshold_person)]

        if is_async:
            self.curr_request, self.next_request = self.next_request, self.curr_request

        return persons


class PersonReIdentification(BaseDetection):
    def __init__(self, device, model_xml):
        detection_of = "Person re-identifications"
        super().__init__(device, model_xml, detection_of)
        self.infer_request = self.compiled_model.create_infer_request()

    def infer(self, person_frame):
        resized_frame = self.preprocess(person_frame)
        self.infer_request.set_tensor(
            self.input_layer_ir, ov.Tensor(resized_frame))
        self.infer_request.infer()

    def get_results(self):
        """
        person-reidentification-retail-0031:
        person-reidentification-retail-0248:
        Output:
            The net outputs a blob with the [1, 256, 1, 1] shape named descriptor, 
            which can be compared with other descriptors using the cosine distance.
        """
        res = self.infer_request.get_output_tensor(0).data
        feature_vec = res.reshape(1, 256)
        return feature_vec
