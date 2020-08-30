import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as npimg
import os

## Keras
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense

import cv2
import pandas as pd
import random
import ntpath

## Sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.applications import ResNet50
from resnet_model import resnet_model


image_size = 100
print('Loading data...')
training_file = 'SDC_lake_ResNet_{}.npy'.format(image_size)
training_data = np.load(training_file, allow_pickle=True)
feature = []
label = []

print('Splitting data...')
for item in training_data:
    feature.append(item[0])
    label.append(item[1])

feature = np.asarray(feature)
label = np.asarray(label)
x_train, x_valid, y_train, y_valid = train_test_split(feature, label, test_size=0.2, random_state=0)

print('Loading model...')
model = resnet_model()
print(model.summary())

print('Training model...')
history = model.fit(x_train, y_train, epochs=25, validation_data=(x_valid, y_valid), batch_size=128, verbose=1, shuffle=1)

print('Saving model...')
model.save('model_lake_ResNet.h5')


