# -*- coding: utf-8 -*-

from yolo_wrapper import YoloDetectorWrapper
import os


def main():
    base = '/home/nikolaevra/datasets/traffic/Insight-MVT_Annotation_Train/MVI_20011'
    image = 'img00286.jpg'

    img_file = os.path.join(base, image)

    detector = YoloDetectorWrapper(tiny=True)
    detector.detect(img_file)


if __name__ == '__main__':
    main()
