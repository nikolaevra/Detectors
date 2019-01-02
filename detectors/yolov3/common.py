# coding=utf-8

import tensorflow as tf
slim = tf.contrib.slim


def conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)

    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs


@tf.contrib.framework.add_arg_scope
def fixed_padding(inputs, kernel_size, mode='CONSTANT', *args, **kwargs):
    """
    Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      mode: The mode for tf.pad.

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]], mode=mode)
    return padded_inputs


def upsample(inputs, out_shape, data_format='NCHW'):
    # tf.image.resize_nearest_neighbor accepts input in format NHWC
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])

    if data_format == 'NCHW':
        new_height = out_shape[2]
        new_width = out_shape[3]
    else:
        new_height = out_shape[1]
        new_width = out_shape[2]

    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

    # back to NCHW if needed
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = tf.identity(inputs, name='upsampled')
    return inputs


def reshape(x_y_offset, boxes, confs, probs, num_classes):
    grid_size = x_y_offset.shape.as_list()[:2]
    boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
    confs = tf.reshape(confs, [-1, grid_size[0] * grid_size[1] * 3, 1])
    probs = tf.reshape(probs, [-1, grid_size[0] * grid_size[1] * 3, num_classes])

    return boxes, confs, probs


def get_predictions(feature_map_anchors, img_size, num_classes, data_format):
    boxes_list, confs_list, probs_list = [], [], []

    results = [
        reorg_layer(feature_map, anchors, img_size, num_classes, data_format)
        for (feature_map, anchors) in feature_map_anchors]

    for result in results:
        boxes, conf_logits, prob_logits = reshape(
            result[0], result[1], result[2], result[3], num_classes
        )

        confs = tf.sigmoid(conf_logits)
        probs = tf.sigmoid(prob_logits)

        boxes_list.append(boxes)
        confs_list.append(confs)
        probs_list.append(probs)

    boxes = tf.concat(boxes_list, axis=1)
    confs = tf.concat(confs_list, axis=1)
    probs = tf.concat(probs_list, axis=1)

    center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
    x0 = center_x - width / 2
    y0 = center_y - height / 2
    x1 = center_x + width / 2
    y1 = center_y + height / 2

    boxes = tf.concat([x0, y0, x1, y1], axis=-1)
    return boxes, confs, probs


def detection_layer(inputs, anchors, num_classes):
    num_anchors = len(anchors)
    feature_map = slim.conv2d(inputs, num_anchors * (5 + num_classes), 1,
                              stride=1, normalizer_fn=None,
                              activation_fn=None,
                              biases_initializer=tf.zeros_initializer())
    return feature_map


def get_size(shape, data_format):
    if len(shape) == 4:
        shape = shape[1:]
    return shape[1:3] if data_format == 'NCHW' else shape[0:2]


def feature_map_to_predictions(feature_map, num_classes, anchors, img_size, data_format):
    num_anchors = len(anchors)
    shape = feature_map.get_shape().as_list()
    grid_size = get_size(shape, data_format)
    dim = grid_size[0] * grid_size[1]
    bbox_attrs = 5 + num_classes

    predictions = feature_map
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


def reorg_layer(feature_map, anchors, img_size, num_classes, data_format):
    num_anchors = len(anchors)  # num_anchors=3
    # grid_size = tf.shape(feature_map)[1:3]
    grid_size = feature_map.shape.as_list()[1:3]

    stride = tf.cast(img_size // grid_size, tf.float32)
    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]

    feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], num_anchors,
                                           5 + num_classes])

    box_centers, box_sizes, conf_logits, prob_logits = tf.split(
        feature_map, [2, 2, 1, num_classes], axis=-1)

    box_centers = tf.nn.sigmoid(box_centers)

    grid_x = tf.range(grid_size[0], dtype=tf.int32)
    grid_y = tf.range(grid_size[1], dtype=tf.int32)

    a, b = tf.meshgrid(grid_x, grid_y)
    x_offset = tf.reshape(a, (-1, 1))
    y_offset = tf.reshape(b, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2])
    x_y_offset = tf.cast(x_y_offset, tf.float32)

    box_centers = box_centers + x_y_offset
    box_centers = box_centers * stride

    box_sizes = tf.exp(box_sizes) * anchors
    box_sizes = box_sizes * stride

    boxes = tf.concat([box_centers, box_sizes], axis=-1)
    return x_y_offset, boxes, conf_logits, prob_logits


def loss_layer(feature_map_i, y_true, anchors, img_size, num_classes):
    NO_OBJECT_SCALE = 1.0
    OBJECT_SCALE = 5.0
    COORD_SCALE = 1.0
    CLASS_SCALE = 1.0

    grid_size = tf.shape(feature_map_i)[1:3]
    stride = tf.cast(img_size // grid_size, dtype=tf.float32)

    pred_result = reorg_layer(feature_map_i, anchors, img_size, num_classes)
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

    # adjust x and y => relative position to the containing cell
    true_box_xy = true_box_xy / stride - xy_offset
    pred_box_xy = pred_box_xy / stride - xy_offset

    # adjust w and h => relative size to the containing cell
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

    # penalize the confidence of the boxes, which are reponsible for corresponding ground
    # truth box
    conf_mask = conf_mask + object_mask * OBJECT_SCALE

    # adjust class probabilities
    class_mask = object_mask * CLASS_SCALE

    # class mask: simply the position of the ground truth boxes (the predictors)
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
