import tensorflow as tf
from data.utils import utils, yolov3_tiny
from utils.visualize import save_image_with_boxes

INPUT_SIZE = 416
BATCH_SIZE = 1
EPOCHS = 20
LR = 0.001
SHUFFLE_SIZE = 1

sess = tf.Session()
classes = utils.read_coco_names('./data/obj.names')
num_classes = len(classes)
file_pattern = "./data/train_data/custom_data/tfrecords/custom_data*.tfrecords"
anchors = utils.get_anchors('./data/yolo_tiny_anchors.txt')

is_training = tf.placeholder(dtype=tf.bool, name="phase_train")
dataset = tf.data.TFRecordDataset(filenames=tf.gfile.Glob(file_pattern))
dataset = dataset.map(
    utils.Parser(anchors, num_classes, input_shape=[416, 416]).parser_example,
    num_parallel_calls=10
)
dataset = dataset.repeat().shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).prefetch(BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
example = iterator.get_next()

images, y_true_13, y_true_26, y_true_52 = example
model = yolov3_tiny.yolov3_tiny

with tf.variable_scope('yolov3-tiny'):
    feature_maps = model.init_feature_maps(images, is_training=is_training)
    y_train_pred = model.predict(feature_maps)

load_ops = utils.load_weights(
    tf.global_variables(scope='yolov3-tiny'),
    "/home/nikolaevra/dev/tensorflow-yolov3/checkpoint/yolov3-tiny_retrained.weights"
)
sess.run(load_ops)

for epoch in range(EPOCHS):
    run_items = sess.run([y_train_pred, [y_true_13, y_true_26], images[0]], feed_dict={is_training: False})

    filtered_boxes = utils.get_prediction_boxes(run_items[0], len(classes))

    # Visualize single image.
    save_image_with_boxes(run_items[2], filtered_boxes)

    # Run evaluation on batch.
    rec, prec, mAP, avg_iou = utils.evaluate(run_items[0], run_items[1], num_classes, num_feature_maps=2)
    print("=> EPOCH: %2d  recall: %.2f  precision: %.2f  mAP: %.2f  AVG_IOU: %.2f"
          % (epoch, rec, prec, mAP, avg_iou))
