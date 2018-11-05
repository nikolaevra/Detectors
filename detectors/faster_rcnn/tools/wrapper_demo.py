#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cv2
import os

# Import this to initialize all of the paths for imported local modules.
import _init_paths

import matplotlib.pyplot as plt
import numpy as np
from model.config import cfg
from wrapper.faster_rcnn_wrapper import DetectorWrapper

from model.nms_wrapper import nms

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {
    'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),
    'res101': ('res101_faster_rcnn_iter_110000.ckpt',)
}

DATASETS = {
    'pascal_voc': ('voc_2007_trainval',),
    'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)
}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title('{} detections with p({} | box) >= {:.1f}'.format(class_name, class_name, thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()


def vis(image, scores, boxes):
    # TODO: load images in one place inatead of twice.
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image)
    im = cv2.imread(im_file)

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='extraction_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset',
                        help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # model path
    extraction_net = args.extraction_net
    dataset = args.dataset

    images = [
        'img00001.jpg',
        # 'img00002.jpg',
        # 'img00003.jpg',
        # 'img00006.jpg',
        # 'img00005.jpg'
    ]

    # Use all the default parameters for now
    detector = DetectorWrapper()
    detections = detector.detect(images)
    for image in images:
        print(image)
        vis(image, detections[image]['scores'], detections[image]['boxes'])
