# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from layer_utils.generate_anchors import generate_anchors


def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8, 16, 32),
                         anchor_ratios=(0.5, 1, 2)):
    """ A wrapper function to generate anchors given different scales. Also return the number of
    anchors in variable 'length'
    """
    anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
    A = anchors.shape[0]
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack(
        (shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0]
    # width changes faster, so here it is H, W, C
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    length = np.int32(anchors.shape[0])

    return anchors, length


def generate_anchors_pre_tf(height, width, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    """ Generate anchor boxes

    :param height: number of vertical anchor boxes that fit into image dimensions
    :param width: number of vertical anchor boxes that fit into image dimensions
    :param feat_stride: the size of the feature stide
    :param anchor_scales: scales of anchors
    :param anchor_ratios: aspect ratios of anchor boxes
    :return:
    """
    # Generate all of the horizontal and vertical positions for anchors.
    shift_x = tf.range(width) * feat_stride
    shift_y = tf.range(height) * feat_stride

    # Create a meshgrid of the positions of all anchors
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)

    # Flatten meshgrids into 1D arrays
    sx = tf.reshape(shift_x, shape=(-1,))
    sy = tf.reshape(shift_y, shape=(-1,))

    # Stack flat meshgrids together twice to be used as increments for anchors shaped like
    # [x_min, y_min, x_max, y_max]
    shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))

    # Find total number of shifts for anchors
    K = tf.multiply(width, height)

    # Permutate dimensions
    shifts = tf.transpose(tf.reshape(shifts, shape=[1, K, 4]), perm=(1, 0, 2))

    # Generates all of the anchors of all aspect ratios. Generates (num_ratios x num_scales)
    # matrix of vectors containing [x_min, y_min, x_max, y_max]
    anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))

    # Get the number of anchors
    A = anchors.shape[0]

    # shape of object storing anchors. In default case dim = (1, 9, 4)
    anchor_constant = tf.constant(anchors.reshape((1, A, 4)), dtype=tf.int32)

    # Number of all anchors for all shifts
    length = K * A

    # Initialize positions of all anchors, by incrememnting the base anchor with all points on the
    # meshgrid
    anchors_tf = tf.reshape(tf.add(anchor_constant, shifts), shape=(length, 4))

    # return matrix of all anchors as well as number of all anchors
    return tf.cast(anchors_tf, dtype=tf.float32), length
