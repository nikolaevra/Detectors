# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image

import yolo_v3
import yolo_v3_tiny
from timer import Timer
from utils import load_coco_names, draw_boxes, detections_boxes, non_max_suppression


class YoloDetectorWrapper:
    def __init__(self, tiny=False, cls_file='coco.names', img_size=(416, 416), data_format='NHWC', ckpt_file='./saved_model/model.ckpt', conf_threshold=0.5, iou_threshold=0.4):
        """ Wrapper class for the YOLO v3 detector.

        :param tiny: if you want to use tiny yolo
        :param cls_file: file storing detection classes
        :param img_size: tuple storing image size
        :param data_format: Data format: NCHW (gpu only) / NHWC
        :param ckpt_file: path to model checkpoint file
        """
        self.tiny = tiny
        self.size = img_size
        self.data_format = data_format

        # Temporary solution
        self.ckpt_file = ckpt_file if not tiny else '.' + ckpt_file.split('.')[1] + '-tiny.ckpt'

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if self.tiny:
            self.model = yolo_v3_tiny.yolo_v3_tiny
        else:
            self.model = yolo_v3.yolo_v3

        self.classes = load_coco_names(cls_file)

        self.inputs = tf.placeholder(tf.float32, [1, self.size[0], self.size[1], 3])

        with tf.variable_scope('detector'):
            self.detections = self.model(self.inputs, len(self.classes), data_format=self.data_format)

        self.saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

        self.boxes = detections_boxes(self.detections)

        self.sess = tf.Session()
        self.saver.restore(self.sess, self.ckpt_file)
        print('Model restored.')

    def detect(self, image, output_img='out'):
        img = Image.open(image)
        img_resized = img.resize(size=(self.size[0], self.size[1]))

        timer = Timer()
        timer.tic()

        detected_boxes = self.sess.run(
            self.boxes,
            feed_dict={self.inputs: [np.array(img_resized, dtype=np.float32)]}
        )

        filtered_boxes = non_max_suppression(
            detected_boxes,
            confidence_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold
        )

        timer.toc()
        total_t = timer.total_time

        print('Detection took {:.3f}s'.format(total_t))

        draw_boxes(filtered_boxes, img, self.classes, (self.size[0], self.size[1]))

        img.save((output_img if self.tiny else output_img + '-tiny') + '.jpg')
