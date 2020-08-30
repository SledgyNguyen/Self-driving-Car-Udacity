import time
import os

import numpy as np
import pandas as pd

from nvidia_model import nvidia_model
from utils import IMG_W, IMG_H, IMG_C, data_generator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# == DEFINE VARIABLES ==================================================================================================
training_file = "SDC_jungle.npy"
LR = 1.0e-4

EPOCHS = 10
simulator = "unity"
net = "nvidia"
MODEL_NAME = 'SDC-{}-{}'.format(EPOCHS, time.time())
img_dir = 'images/IMG'
csv_file = 'images/driving_log.csv'


# == FUNCTION DEFINE ===================================================================================================
def main():
    # load training data
    data_df = pd.read_csv(csv_file, names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    x = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    y = np.ndarray.tolist(y)

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=0)

    # load model
    model = nvidia_model()
    
    checkpoint = ModelCheckpoint('model-{epoch:03d}-{val_loss:.6f}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
    
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=LR))

    # Tensorflow 1.x
    # model.fit_generator(data_generator(x_train, y_train, batch_size=40, edge_detect=False, is_training=True),
    #                     steps_per_epoch=5000,
    #                     epochs=EPOCHS,
    #                     max_queue_size=1,
    #                     validation_data=data_generator(x_valid, y_valid, batch_size=40, edge_detect=False, is_training=False),
    #                     nb_val_samples=len(x_valid),
    #                     callbacks=[checkpoint],
    #                     verbose=1)

    # Tensorflow 2.x
    model.fit(data_generator(x_train, y_train, batch_size=40, is_training=True),
                        steps_per_epoch=5000,
                        epochs=EPOCHS,
                        max_queue_size=1,
                        validation_data=data_generator(x_valid, y_valid, batch_size=40,
                                                       is_training=False),
                        # nb_val_samples=len(x_valid),
                        validation_steps=len(x_valid) / 40,
                        callbacks=[checkpoint],
                        verbose=1)

    # model.save("model_jungle_{}_{}.h5".format(EPOCHS, LR))


if __name__ == '__main__':
    main()
