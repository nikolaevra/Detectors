from yolo.yolo_wrapper import YoloDetectorWrapper
from PIL import Image
from yolo.module_utils.timer import time

import os

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    base = '../data/demo'
    image = 'img00001.jpg'

    img_file = os.path.join(BASE_DIR, base, image)
    ckpt_path = os.path.join(BASE_DIR, '../models/tiny/model.ckpt')
    cls_path = os.path.join(BASE_DIR, '../config/obj.names')

    # Test YOLO
    yolo = YoloDetectorWrapper(
        tiny=True,
        ckpt_path=ckpt_path,
        cls_path=cls_path,
        frozen_model=''
    )

    img = Image.open(img_file)

    for i in range(5):
        t0 = time.time()
        detections = yolo.detect(img)
        print('Inference Time {:.2f}s'.format(time.time()-t0))

    # yolo.save_image(img, detections, output_img='out')

    # yolo.export(path=model_path)


if __name__ == '__main__':
    main()
