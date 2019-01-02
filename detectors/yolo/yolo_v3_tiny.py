# -*- coding: utf-8 -*-

import tensorflow as tf
from yolo_v3 import _conv2d_fixed_padding, _fixed_padding, _get_size, \
    _upsample

slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

_ANCHORS = [(10, 14), (23, 27), (37, 58),
            (81, 82), (135, 169), (344, 319)]


def predict(self, feature_maps):
    """
    Note: given by feature_maps, compute the receptive field
          and get boxes, confs and class_probs
    input_argument: feature_maps -> [None, 13, 13, 255],
                                    [None, 26, 26, 255],
                                    [None, 52, 52, 255],
    """
    feature_map_1, feature_map_2, feature_map_3 = feature_maps
    feature_map_anchors = [(feature_map_1, self._ANCHORS[6:9]),
                           (feature_map_2, self._ANCHORS[3:6]),
                           (feature_map_3, self._ANCHORS[0:3]), ]

    results = [self._reorg_layer(feature_map, anchors) for (feature_map, anchors) in
               feature_map_anchors]
    boxes_list, confs_list, probs_list = [], [], []

    for result in results:
        boxes, conf_logits, prob_logits = self._reshape(*result)

        confs = tf.sigmoid(conf_logits)
        probs = tf.sigmoid(prob_logits)

        boxes_list.append(boxes)
        confs_list.append(confs)
        probs_list.append(probs)

    boxes = tf.concat(boxes_list, axis=1)
    confs = tf.concat(confs_list, axis=1)
    probs = tf.concat(probs_list, axis=1)

    center_x, center_y, height, width = tf.split(boxes, [1, 1, 1, 1], axis=-1)
    x0 = center_x - height / 2
    y0 = center_y - width / 2
    x1 = center_x + height / 2
    y1 = center_y + width / 2

    boxes = tf.concat([x0, y0, x1, y1], axis=-1)
    return boxes, confs, probs


def compute_loss(self, y_pred, y_true):
    """
    Note: compute the loss
    Arguments: y_pred, list -> [feature_map_1, feature_map_2, feature_map_3]
                                    the shape of [None, 13, 13, 3*85]. etc
    """
    loss_coord, loss_sizes, loss_confs, loss_class = 0., 0., 0., 0.
    _ANCHORS = [self._ANCHORS[6:9], self._ANCHORS[3:6], self._ANCHORS[0:3]]

    for i in range(len(y_pred)):
        result = self.loss_layer(y_pred[i], y_true[i], _ANCHORS[i])
        loss_coord += result[0]
        loss_sizes += result[1]
        loss_confs += result[2]
        loss_class += result[3]

    total_loss = loss_coord + loss_sizes + loss_confs + loss_class
    return [total_loss, loss_coord, loss_sizes, loss_confs, loss_class]


def loss_layer(self, feature_map_i, y_true, anchors):
    NO_OBJECT_SCALE = 1.0
    OBJECT_SCALE = 5.0
    COORD_SCALE = 1.0
    CLASS_SCALE = 1.0

    grid_size = tf.shape(feature_map_i)[1:3]
    stride = tf.cast(self.img_size // grid_size, dtype=tf.float32)

    pred_result = self._reorg_layer(feature_map_i, anchors)
    xy_offset, pred_box, pred_box_conf_logits, pred_box_class_logits = pred_result

    true_box_xy = y_true[..., :2]  # absolute coordinate
    true_box_wh = y_true[..., 2:4]  # absolute size

    pred_box_xy = pred_box[..., :2]  # absolute coordinate
    pred_box_wh = pred_box[..., 2:4]  # absolute size

    # caculate iou between true boxes and pred boxes
    intersect_xy1 = tf.maximum(true_box_xy - true_box_wh / 2.0,
                               pred_box_xy - pred_box_xy / 2.0)
    intersect_xy2 = tf.minimum(true_box_xy + true_box_wh / 2.0,
                               pred_box_xy + pred_box_wh / 2.0)
    intersect_wh = tf.maximum(intersect_xy2 - intersect_xy1, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_area = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_area = true_area + pred_area - intersect_area
    iou_scores = tf.truediv(intersect_area, union_area)
    iou_scores = tf.expand_dims(iou_scores, axis=-1)

    true_box_conf = y_true[..., 4:5]
    pred_box_conf = tf.sigmoid(pred_box_conf_logits)
    ### adjust x and y => relative position to the containing cell
    true_box_xy = true_box_xy / stride - xy_offset
    pred_box_xy = pred_box_xy / stride - xy_offset

    ### adjust w and h => relative size to the containing cell
    true_box_wh_logit = true_box_wh / (anchors * stride)
    pred_box_wh_logit = pred_box_wh / (anchors * stride)

    true_box_wh_logit = tf.where(condition=tf.equal(true_box_wh_logit, 0),
                                 x=tf.ones_like(true_box_wh_logit), y=true_box_wh_logit)
    pred_box_wh_logit = tf.where(condition=tf.equal(pred_box_wh_logit, 0),
                                 x=tf.ones_like(pred_box_wh_logit), y=pred_box_wh_logit)

    true_box_wh = tf.log(true_box_wh_logit)
    pred_box_wh = tf.log(pred_box_wh_logit)

    object_mask = y_true[..., 4:5]
    conf_mask = tf.to_float(iou_scores < 0.6) * (1 - object_mask) * NO_OBJECT_SCALE
    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + object_mask * OBJECT_SCALE

    ### adjust class probabilities
    class_mask = object_mask * CLASS_SCALE
    ### class mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = object_mask * COORD_SCALE

    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

    loss_coord = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (
            nb_coord_box + 1e-6) / 2.
    loss_sizes = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (
            nb_coord_box + 1e-6) / 2.
    loss_confs = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (
            nb_conf_box + 1e-6) / 2.
    loss_class = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:],
                                                         logits=pred_box_class_logits)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

    return loss_coord, loss_sizes, loss_confs, loss_class

def _detection_layer(inputs, num_anchors, num_classes):
    return slim.conv2d(inputs, num_anchors * (5 + num_classes), 1,
                       stride=1, normalizer_fn=None,
                       activation_fn=None,
                       biases_initializer=tf.zeros_initializer())


def _fmap_to_detections(predictions, anchors, num_classes, img_size, data_format):
    num_anchors = len(anchors)
    shape = predictions.get_shape().as_list()
    grid_size = _get_size(shape, data_format)
    dim = grid_size[0] * grid_size[1]
    bbox_attrs = 5 + num_classes

    if data_format == 'NCHW':
        predictions = tf.reshape(
            predictions, [-1, num_anchors * bbox_attrs, dim])
        predictions = tf.transpose(predictions, [0, 2, 1])

    predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])

    stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])

    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]

    box_centers, box_sizes, confidence, classes = tf.split(
        predictions, [2, 2, 1, num_classes], axis=-1)

    box_centers = tf.nn.sigmoid(box_centers)
    confidence = tf.nn.sigmoid(confidence)

    grid_x = tf.range(grid_size[0], dtype=tf.float32)
    grid_y = tf.range(grid_size[1], dtype=tf.float32)

    a, b = tf.meshgrid(grid_x, grid_y)

    x_offset = tf.reshape(a, (-1, 1))
    y_offset = tf.reshape(b, (-1, 1))

    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])

    box_centers = box_centers + x_y_offset
    box_centers = box_centers * stride

    anchors = tf.tile(anchors, [dim, 1])
    box_sizes = tf.exp(box_sizes) * tf.cast(anchors, tf.float32)
    box_sizes = box_sizes * stride

    detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)

    classes = tf.nn.sigmoid(classes)
    predictions = tf.concat([detections, classes], axis=-1)

    return predictions


def yolo_v3_tiny(inputs, num_classes, is_training=False, data_format='NCHW', reuse=False):
    """
    Creates YOLO v3 tiny model.

    :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
        Dimension batch_size may be undefined. The channel order is RGB.
    :param num_classes: number of predicted classes.
    :param is_training: whether is training or not.
    :param data_format: data format NCHW or NHWC.
    :param reuse: whether or not the network and its variables should be reused.
    :return:
    """
    # it will be needed later on
    img_size = inputs.get_shape().as_list()[1:3]

    # transpose the inputs to NCHW
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # normalize values to range [0..1]
    inputs = inputs / 255

    # set batch norm params
    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,  # Use fused batch norm if possible.
    }

    # Set activation_fn and parameters for conv2d, batch_norm.
    with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding, slim.max_pool2d],
                        data_format=data_format):
        with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):

                with tf.variable_scope('yolo-v3-tiny'):
                    for i in range(6):
                        inputs = _conv2d_fixed_padding(
                            inputs, 16 * pow(2, i), 3)

                        if i == 4:
                            route_1 = inputs

                        if i == 5:
                            inputs = slim.max_pool2d(
                                inputs, [2, 2], stride=1, padding="SAME", scope='pool2')
                        else:
                            inputs = slim.max_pool2d(
                                inputs, [2, 2], scope='pool2')

                    inputs = _conv2d_fixed_padding(inputs, 1024, 3)
                    inputs = _conv2d_fixed_padding(inputs, 256, 1)
                    route_2 = inputs

                    inputs = _conv2d_fixed_padding(inputs, 512, 3)

                    # Get the first feature map.
                    feature_map_1 = _detection_layer(inputs, len(_ANCHORS[3:6]), num_classes)
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    # Get the first detections layer.
                    detect_1 = _fmap_to_detections(
                        feature_map_1, _ANCHORS[3:6], num_classes, img_size, data_format)
                    detect_1 = tf.identity(detect_1, name='detect_1')

                    inputs = _conv2d_fixed_padding(route_2, 128, 1)
                    upsample_size = route_1.get_shape().as_list()
                    inputs = _upsample(inputs, upsample_size, data_format)

                    inputs = tf.concat([inputs, route_1],
                                       axis=1 if data_format == 'NCHW' else 3)

                    inputs = _conv2d_fixed_padding(inputs, 256, 3)

                    # Get the second feature map.
                    feature_map_2 = _detection_layer(inputs, len(_ANCHORS[0:3]), num_classes)
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    # Get the second detection layer.
                    detect_2 = _fmap_to_detections(
                        feature_map_2, _ANCHORS[0:3], num_classes, img_size, data_format)
                    detect_2 = tf.identity(detect_2, name='detect_2')

                    detections = tf.concat([detect_1, detect_2], axis=1)
                    detections = tf.identity(detections, name='detections')
                    return detections, feature_map_1, feature_map_2
