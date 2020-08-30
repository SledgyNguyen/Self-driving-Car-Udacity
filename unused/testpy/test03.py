import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import os
import cv2

from utils import preprocess1
from utils import data_generator
from PIL import Image
from sklearn.model_selection import train_test_split


img_dir = '../images/IMG'
csv_file = '../images/driving_log.csv'

print(os.path.isfile(csv_file))

data_df = pd.read_csv(csv_file, names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
x = data_df[['center', 'left', 'right']].values
y = data_df['steering'].values
y = np.ndarray.tolist(y)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=0)

result = data_generator(x_train, y_train, batch_size=1, is_training=True, edge_detect=False)

result = next(result)

timg = result[0]
timg = timg[0]
tsteer = result[1]
tsteer = tsteer[0]

oimg = x_train[0][0]
oimg = mpimg.imread(oimg)
osteer = y_train[0]
# center = preprocess1(center, edge_detect=False)
# translated = preprocess1(translated, edge_detect=False)

fig, ax = plt.subplots(2)
ax[0].imshow(oimg)
ax[0].axis(False)
ax[0].set_title("ORIGINAL\nsteering = %.6f" % osteer)

# ax[1].imshow(timg)
# ax[1].imshow(timg.astype('uint8'))
ax[1].imshow(timg.astype(np.float32))
ax[1].axis(False)
ax[1].set_title("MODIFIED\nsteering = %.6f" % tsteer)

plt.show()
