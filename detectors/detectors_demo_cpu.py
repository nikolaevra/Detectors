from yolo.yolo_wrapper import YoloDetectorWrapper

import os

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    base = '../data/demo'
    image = 'img00011.jpg'

    img_file = os.path.join(BASE_DIR, base, image)
    ckpt_path = os.path.join(BASE_DIR, '../models/full/model.ckpt')
    model_path = os.path.join(BASE_DIR, '../models')
    cls_path = os.path.join(BASE_DIR, 'yolo/coco.names')

    # Test YOLO
    yolo = YoloDetectorWrapper(
        tiny=False,
        ckpt_path=ckpt_path,
        cls_path=cls_path,
        frozen_model='',
        gpu=0
    )
    detections = yolo.detect(img_file)
    # yolo.save_image(detections, detections, output_img='out12')

    # yolo.export(path=model_path)


if __name__ == '__main__':
    main()
