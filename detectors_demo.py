from detectors.yolo.yolo_wrapper import YoloDetectorWrapper

import os


def main():
    base = '/home/nikolaevra/datasets/traffic/Insight-MVT_Annotation_Train/MVI_20011'
    image = 'img00286.jpg'

    img_file = os.path.join(base, image)

    # Test YOLO
    yolo = YoloDetectorWrapper(tiny=False)
    yolo.detect(img_file)


if __name__ == '__main__':
    main()
