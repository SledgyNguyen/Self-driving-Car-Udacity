import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import os
import random


np.random.seed(0)


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
        left = mpimg.imread(frame[1])
        right = mpimg.imread(frame[2])
        angle = frame[3]
        training_data.append([center, left, right, angle])

    random.shuffle(training_data)

    # == SAVING DATA ===================================================================================================
    print("-- Saving data...")
    file_name = 'SDC_lake.npy'
    np.save(file_name, training_data)


if __name__ == '__main__':
    main()
