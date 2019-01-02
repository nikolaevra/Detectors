import data.data_utils as utils
import argparse
import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
IMAGES_FILE = 'image_names.txt'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        default='data/train_data/custom_data'
    )
    parser.add_argument(
        "--tfrecord_path_prefix",
        default='data/train_data/custom_data/tfrecords/custom_data'
    )
    parser.add_argument(
        "--num_tfrecords",
        default=3,
        type=int
    )
    parser.add_argument(
        "--img-height",
        default=540,
        type=int
    )
    parser.add_argument(
        "--img-width",
        default=960,
        type=int
    )
    return parser.parse_args()


def main():
    flags = parse_args()

    images_base = os.path.join(BASE_DIR, flags.dataset_dir)
    dataset = utils.read_image_box_from_text(os.path.join(images_base, IMAGES_FILE), images_base)

    image_paths = list(dataset.keys())
    images_num = len(image_paths)
    print(">> Processing %d images" % images_num)
    per_tfrecord_images = images_num // flags.num_tfrecords
    print(">> %d images per record" % per_tfrecord_images)

    n = 0
    while n <= flags.num_tfrecords:
        tfrecord_file = flags.tfrecord_path_prefix + "%04d.tfrecords" % n
        with tf.python_io.TFRecordWriter(tfrecord_file) as record_writer:
            st = n * per_tfrecord_images
            en = (n + 1) * per_tfrecord_images if n < flags.num_tfrecords else len(image_paths)
            for i in range(st, en):
                image = tf.gfile.GFile(image_paths[i], 'rb').read()
                bboxes, labels = dataset[image_paths[i]]
                bboxes = utils.convert_yolo_to_min_max_boxes(
                    bboxes,
                    flags.img_height,
                    flags.img_width
                )
                bboxes = bboxes.tostring()

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                        'bboxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bboxes])),
                        'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
                    }
                ))

                record_writer.write(example.SerializeToString())
            print(">> Saving %5d images in %s" % (en - st, tfrecord_file))
            n += 1


if __name__ == "__main__":
    main()
