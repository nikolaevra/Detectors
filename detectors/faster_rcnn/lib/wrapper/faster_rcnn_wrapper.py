from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os

import tensorflow as tf
from model.config import cfg
from model.test import im_detect
from nets.resnet_v1 import resnetv1
from nets.vgg16 import vgg16
from utils.timer import Timer

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


class DetectorWrapper:
    def __init__(self, extraction_net='res101', dataset='pascal_voc_0712', num_classes=21,
                 tag='default', anchor_scales=[8, 16, 32], anchor_ratios=(0.5, 1, 2)):
        cfg.TEST.HAS_RPN = True

        # model path
        self.extraction_net = extraction_net
        self.dataset = dataset
        self.tfmodel = os.path.join(
            'output',
            extraction_net,
            DATASETS[dataset][0],
            'default',
            NETS[extraction_net][0]
        )

        if not os.path.isfile(self.tfmodel + '.meta'):
            raise IOError('{:s} not found.\n'.format(self.tfmodel + '.meta'))

        # Make sure we allow using CPU when GPU is not available
        self.tfconfig = tf.ConfigProto(allow_soft_placement=True)
        # make sure we first allocate small amount of GPU power and grow it as needed
        self.tfconfig.gpu_options.allow_growth = True

        # init tf session
        self.sess = tf.Session(config=self.tfconfig)

        self.net = None
        # load network
        if extraction_net == 'vgg16':
            self.net = vgg16()
        elif extraction_net == 'res101':
            self.net = resnetv1(num_layers=101)
        else:
            raise NotImplementedError

        self.net.create_architecture(
            "TEST",
            num_classes=num_classes,
            tag=tag,
            anchor_scales=anchor_scales,
            anchor_ratios=anchor_ratios
        )

        # Saver is an easy interface to save/load models and its weights based on a checkpoint
        # number.
        self.saver = tf.train.Saver()
        # Load model and weights for the pre-trained extraction model.
        self.saver.restore(self.sess, self.tfmodel)

        print('Loaded network {:s}'.format(self.tfmodel))

    def detect(self, images):
        """ Detect images from array of image filenames.

        :param images: list of image filenames to be detected.
        :return: dict(dict()) of detections
        """
        detections = {}

        for image in images:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

            # Load the demo image
            im_file = os.path.join(cfg.DATA_DIR, 'demo', image)
            im = cv2.imread(im_file)

            timer = Timer()
            timer.tic()

            # Get image detections
            scores, boxes = im_detect(self.sess, self.net, im)
            timer.toc()
            total_t = timer.total_time

            print('Detection took {:.3f}s for {:d} proposals'.format(total_t, boxes.shape[0]))

            detections[image] = {
                "scores": scores,
                "boxes": boxes,
                "detection_time": total_t
            }

        return detections
