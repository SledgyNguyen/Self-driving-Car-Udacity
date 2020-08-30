import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2

from utils import preprocess1
from utils import data_generator
from PIL import Image

import h5py
import tensorflow as tf

f = h5py.File('D:/TUAN/Workspace/Python/sdc-tf2/model-013-0.007684.h5', 'r')
print(f.attrs.get('keras_version'))
print(tf.__version__)

# center = mpimg.imread(os.path.join("../test_imgs", "center.png"))
# left = mpimg.imread(os.path.join("../test_imgs", "left.png"))
# right = mpimg.imread(os.path.join("../test_imgs", "right.png"))
#
#
# fig, ax = plt.subplots(1,3)
# ax[0].imshow(left)
# ax[1].imshow(center)
# ax[2].imshow(right)
#
# ax[0].axis(False)
# ax[1].axis(False)
# ax[2].axis(False)

# plt.show()