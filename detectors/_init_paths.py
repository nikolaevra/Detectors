import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
add_path(osp.join(this_dir, 'faster_rcnn', 'lib'))
add_path(osp.join(this_dir, 'yolo'))
# add_path(osp.join(this_dir, 'data', 'coco', 'PythonAPI'))
