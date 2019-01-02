import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

from PIL import Image

IMG_WIDTH = 960
IMG_HEIGHT = 540

base_path = '/home/nikolaevra/datasets/traffic/parsed_data'
filename = 'MVI_41063_img01274'

im = np.array(Image.open(os.path.join(base_path, filename + '.jpg')), dtype=np.uint8)

with open(os.path.join(base_path, filename + '.txt'), 'r') as f:
    detections = f.read().split('\n')

# Create figure and axes
fig, ax = plt.subplots(1)

# Display the image
ax.imshow(im)

for detection in detections:
    detection = detection.split(' ')

    width = float(detection[3]) * IMG_WIDTH
    height = float(detection[4]) * IMG_HEIGHT

    top = (float(detection[1]) * IMG_WIDTH) - (width / 2)
    left = (float(detection[2]) * IMG_HEIGHT) - (height / 2)

    # Create a Rectangle patch
    rect = patches.Rectangle((top, left), width, height, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

plt.show()