import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import os
import random
import cv2


np.random.seed(0)

net = 'ResNet'
image_size = 100


def img_pp(image):
    image = image[60:135, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (image_size, image_size))
    image = image / 255
    return image


def main():
    # == CREATE DATA ===================================================================================================
    print('Creating data for training with Tensorflow 2 and Tensorflow-gpu')

    print("-- Loading data...")
    data_dir = "./images/"
    log_name = "driving_log.csv"
    data_df = pd.read_csv(os.path.join(data_dir, log_name),
                          names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    training_df = data_df[['center', 'left', 'right', 'steering']].values
    training_data = []

    print("-- Cloning data...")
    for frame in training_df:
        center = mpimg.imread(frame[0])
        p_center = img_pp(center)

        # left = mpimg.imread(frame[1])
        # p_left = img_pp(left)
        #
        # right = mpimg.imread(frame[2])
        # p_right = img_pp(right)

        angle = frame[3]

        # training_data.append([p_center, p_left, p_right, angle])
        training_data.append([p_center, angle])

    random.shuffle(training_data)

    # == SAVING DATA ===================================================================================================
    print("-- Saving data...")
    file_name = 'SDC_lake_{}_{}.npy'.format(net, image_size)
    np.save(file_name, training_data)


if __name__ == '__main__':
    main()
