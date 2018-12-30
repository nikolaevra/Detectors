import os

from PIL import Image

from detectors.yolo.module_utils.timer import time
from detectors.yolo.yolo_wrapper import YoloDetectorWrapper
from py_perception.iou_tracker import Tracker


class TaskRunner:
    def __init__(self, config, base_dir):
        self.base_dir = base_dir
        self.ckpt_path = os.path.join(base_dir, config['MODEL_CKPT_PATH'])
        self.cls_path = os.path.join(base_dir, config['CLS_PATH'])
        self.images_file_path = os.path.join(base_dir, config['DATA_DIR'])

        self.detector = YoloDetectorWrapper(
            tiny=False,
            ckpt_path=self.ckpt_path,
            cls_path=self.cls_path,
            frozen_model='',
            gpu=1
        )
        self.tracker = Tracker(config['SIGMA_L'], config['SIGMA_H'], config['SIGMA_IOU'], config['T_MIN'])

    def run_perception_task(self):
        with open(self.images_file_path, 'r') as f:
            images = f.read()

        for image_file in images.split('\n'):
            start = time.time()

            image_path = os.path.join(self.base_dir, image_file)
            img = Image.open(image_path)

            load_image = time.time()

            detections = self.detector.detect(img)
            grouped_detections = [{'score': bboxs[0][1], 'bbox': bboxs[0][0], 'class': cls} for cls, bboxs in
                                  detections.items()]

            image_inference = time.time()

            tracks = self.tracker.track_iou(grouped_detections)

            track_update = time.time()

            image_time = image_inference - load_image
            tracker_time = track_update - image_inference
            total_time = track_update - load_image

            print(
                "Load Img: {:.2f}s | Inference: {:.2f} | Track Update: {:.2f} | Total Time: {:.2f} | FPS: {:.2f}".format(
                    load_image - start,
                    image_time,
                    tracker_time,
                    total_time,
                    1.0 / total_time
                )
            )
