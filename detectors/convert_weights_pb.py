# -*- coding: utf-8 -*-

import tensorflow as tf
import yolo.yolo_v3
import yolo.yolo_v3_tiny

from yolo.utils import load_weights, load_coco_names, detections_boxes, freeze_graph

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'class_names', 'yolo/coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolo/darknet_weights/yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NHWC', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'output_graph', '../graphs/yolov3_tiny_NHWC.pb', 'Frozen tensorflow protobuf model output path')

tf.app.flags.DEFINE_bool(
    'tiny', True, 'Use tiny version of YOLOv3')
tf.app.flags.DEFINE_integer('width', 416, 'Image size')
tf.app.flags.DEFINE_integer('height', 416, 'Image size')


def main(argv=None):
    if FLAGS.tiny:
        model = yolo.yolo_v3_tiny.yolo_v3_tiny
    else:
        model = yolo.yolo_v3.yolo_v3

    classes = load_coco_names(FLAGS.class_names)

    # placeholder for detector inputs
    inputs = tf.placeholder(tf.float32, [1, FLAGS.height, FLAGS.width, 3], "inputs")

    with tf.variable_scope('detector'):
        detections = model(inputs, len(classes), data_format=FLAGS.data_format)
        load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)

    # Sets the output nodes in the current session
    boxes = detections_boxes(detections)

    with tf.Session() as sess:
        sess.run(load_ops)
        freeze_graph(sess, FLAGS.output_graph)


if __name__ == '__main__':
    tf.app.run()
