# -*- coding: utf-8 -*-

import tensorflow as tf

import yolo.yolo_v3
import yolo.yolo_v3_tiny

from yolo.utils import load_coco_names, load_weights

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'class_names', '../config/obj.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolo/darknet_weights/yolov3-tiny_obj_last.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NHWC', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_bool(
    'tiny', True, 'Use tiny version of YOLOv3')
tf.app.flags.DEFINE_string(
    'ckpt_file', '../models/tiny/model.ckpt', 'Checkpoint file')
tf.app.flags.DEFINE_integer('width', 416, 'Image size')
tf.app.flags.DEFINE_integer('height', 416, 'Image size')


def main(argv=None):
    if FLAGS.tiny:
        model = yolo.yolo_v3_tiny.yolo_v3_tiny
    else:
        model = yolo.yolo_v3.yolo_v3

    classes = load_coco_names(FLAGS.class_names)

    # placeholder for detector inputs
    # any size > 320 will work here
    inputs = tf.placeholder(tf.float32, [1, FLAGS.height, FLAGS.width, 3])

    with tf.variable_scope('detector'):
        detections = model(inputs, len(classes),
                           data_format=FLAGS.data_format)
        load_ops = load_weights(tf.global_variables(
            scope='detector'), FLAGS.weights_file)

    saver = tf.train.Saver(tf.global_variables(scope='detector'))

    with tf.Session() as sess:
        sess.run(load_ops)

        save_path = saver.save(sess, save_path=FLAGS.ckpt_file)
        print('Model saved in path: {}'.format(save_path))


if __name__ == '__main__':
    tf.app.run()