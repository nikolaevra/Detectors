# coding=utf-8

import tensorflow as tf
from core import common, utils

slim = tf.contrib.slim


class yolov3_tiny(object):

    def __init__(self, num_classes=80, batch_norm_decay=0.9, leaky_relu=0.1,
                 anchors_path='./data/yolo_tiny_anchors.txt', data_format='NHWC'):

        # self._ANCHORS = [(10, 14),  (23, 27),  (37, 58),
        #             (81, 82),  (135, 169),  (344, 319)]
        self._ANCHORS = utils.get_anchors(anchors_path)
        self._BATCH_NORM_DECAY = batch_norm_decay
        self._LEAKY_RELU = leaky_relu
        self._NUM_CLASSES = num_classes
        self.data_format = data_format
        self.feature_maps = []  # [[None, 26, 26, 255], [None, 52, 52, 255]]

    def init_feature_maps(self, inputs, is_training=False, reuse=False):
        """
        Creates YOLO v3 model.

        :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
               Dimension batch_size may be undefined. The channel order is RGB.
        :param is_training: whether is training or not.
        :param reuse: whether or not the network and its variables should be reused.
        :return:
        """
        # it will be needed later on
        self.img_size = tf.shape(inputs)[1:3]

        # set batch norm params
        batch_norm_params = {
            'decay': self._BATCH_NORM_DECAY,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        # Set activation_fn and parameters for conv2d, batch_norm.
        with slim.arg_scope([slim.conv2d, slim.batch_norm, common.fixed_padding], reuse=reuse):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x,
                                                                         alpha=self._LEAKY_RELU)):
                with tf.variable_scope('darknet-tiny'):
                    for i in range(6):
                        inputs = common.conv2d_fixed_padding(
                            inputs, 16 * pow(2, i), 3)

                        if i == 4:
                            route_1 = inputs

                        if i == 5:
                            inputs = slim.max_pool2d(
                                inputs, [2, 2], stride=1, padding="SAME", scope='pool2')
                        else:
                            inputs = slim.max_pool2d(
                                inputs, [2, 2], scope='pool2')

                with tf.variable_scope('yolo-v3-tiny'):
                    inputs = common.conv2d_fixed_padding(inputs, 1024, 3)
                    inputs = common.conv2d_fixed_padding(inputs, 256, 1)
                    route_2 = inputs

                    inputs = common.conv2d_fixed_padding(inputs, 512, 3)

                    feature_map_1 = common.detection_layer(inputs, self._ANCHORS[3:6], self._NUM_CLASSES)
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inputs = common.conv2d_fixed_padding(route_2, 128, 1)
                    upsample_size = route_1.get_shape().as_list()
                    inputs = common.upsample(inputs, upsample_size, self.data_format)

                    inputs = tf.concat([inputs, route_1], axis=1 if self.data_format == 'NCHW' else 3)

                    inputs = common.conv2d_fixed_padding(inputs, 256, 3)

                    feature_map_2 = common.detection_layer(inputs, self._ANCHORS[0:3], self._NUM_CLASSES)
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

            return feature_map_1, feature_map_2

    def get_feature_map_anchors(self, feature_maps):
        feature_map_1, feature_map_2 = feature_maps
        return [(feature_map_1, self._ANCHORS[3:6]),
                (feature_map_2, self._ANCHORS[0:3]), ]

    def predict(self, feature_maps):
        """
        Note: given by feature_maps, compute the receptive field
              and get boxes, confs and class_probs
        input_argument: feature_maps -> [None, 26, 26, 255],
                                        [None, 52, 52, 255],
        """
        feature_map_anchors = self.get_feature_map_anchors(feature_maps)

        return common.get_predictions(feature_map_anchors, self.img_size, self._NUM_CLASSES, self.data_format)

    def compute_loss(self, y_pred, y_true):
        """
        Note: compute the loss
        Arguments: y_pred, list -> [feature_map_1, feature_map_2]
                                        the shape of [None, 13, 13, 2*85]. etc
        """
        loss_coord, loss_sizes, loss_confs, loss_class = 0., 0., 0., 0.
        _ANCHORS = [self._ANCHORS[3:6], self._ANCHORS[0:3]]

        for i in range(len(y_pred)):
            result = common.loss_layer(y_pred[i], y_true[i], _ANCHORS[i], self.img_size, self._NUM_CLASSES)
            loss_coord += result[0]
            loss_sizes += result[1]
            loss_confs += result[2]
            loss_class += result[3]

        total_loss = loss_coord + loss_sizes + loss_confs + loss_class
        return [total_loss, loss_coord, loss_sizes, loss_confs, loss_class]
