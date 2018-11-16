from detectors.yolo.yolo_wrapper import YoloDetectorWrapper

import os


def main():
    base = os.getcwd()
    models_dir = 'models'

    print("Restoring Model")
    yolo = YoloDetectorWrapper(tiny=False)

    print("Saving Model")
    yolo.export_model(os.path.join(base, models_dir))


if __name__ == '__main__':
    main()
