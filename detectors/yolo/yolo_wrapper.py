# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import yolo_v3
import yolo_v3_tiny
from module_utils.timer import time
from utils import (load_coco_names, draw_boxes, detections_boxes, non_max_suppression, get_boxes_and_inputs,
                   get_boxes_and_inputs_pb, load_graph, letter_box_image)


class YoloDetectorWrapper:
    def __init__(self, tiny=False, cls_path='coco.names', img_size=(416, 416), data_format='NHWC', frozen_model='',
                 ckpt_path='saved_model/model.ckpt', conf_threshold=0.5, iou_threshold=0.4, gpu_memory_fraction=0.2,
                 gpu=0):
        """ Wrapper class for the YOLO v3 detector.

        :param tiny: if you want to use tiny yolo
        :param cls_path: file storing detection classes
        :param img_size: tuple storing image size
        :param data_format: Data format: NCHW (gpu only) / NHWC
        :param ckpt_path: path to model checkpoint file
        """
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction) if gpu > 0 else None
        self.config = tf.ConfigProto(
            gpu_options=self.gpu_options,
            device_count={'CPU': 1, 'GPU': gpu},
            log_device_placement=True
        )

        self.frozen_model = frozen_model
        self.gpu_memory_fraction = gpu_memory_fraction
        self.tiny = tiny
        self.size = img_size
        self.data_format = data_format
        self.ckpt_file = ckpt_path

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # main_device = "/gpu:0" if gpu > 0 else "/cpu:0"

        # with tf.device(main_device):
        if self.tiny:
            self.model = yolo_v3_tiny.yolo_v3_tiny
        else:
            self.model = yolo_v3.yolo_v3

        self.classes = load_coco_names(cls_path)
        self.boxes, self.inputs = get_boxes_and_inputs(self.model, len(self.classes), self.size, self.data_format)
        self.saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))
        self.sess = tf.Session()

        t0 = time.time()
        self.saver.restore(self.sess, self.ckpt_file)
        print('Model restored in {:.2f}s'.format(time.time()-t0))

    def detect(self, img):
        img_resized = letter_box_image(img, self.size[0], self.size[1], 128)
        img_resized = img_resized.astype(np.float32)

        detected_boxes = self.sess.run(self.boxes, feed_dict={self.inputs: [img_resized]})

        filtered_boxes = non_max_suppression(detected_boxes,
                                             confidence_threshold=self.conf_threshold,
                                             iou_threshold=self.iou_threshold)
        return filtered_boxes

    def save_image(self, img, filtered_boxes, output_img='out'):
        draw_boxes(filtered_boxes, img, self.classes, self.size, True)
        img.save(output_img + '.png')

    def export(self, path, filename='graph.pb'):
        print('Exporting Graph.')
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(path + '/train', self.sess.graph)
        # train_writer.add_summary(tf.Variable(3), 0)
