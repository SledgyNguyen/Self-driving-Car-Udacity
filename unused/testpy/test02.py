import os
import numpy as np
import matplotlib.pyplot as plt
# from keras.applications.xception import Xception
from sklearn.model_selection import train_test_split


# print(os.path.isfile('../nvidia_model.py'))
training_file = '../SDC_jungle.npy'
training_data = np.load(training_file, allow_pickle=True)
feature = []
label = []

# print(len(training_data), len(training_data[0]))

for item in training_data:
    feature.append([item[0], item[1], item[2]])
    label.append(item[3])

feature = np.asarray(feature)
label = np.asarray(label)

x_train, x_valid, y_train, y_valid = train_test_split(feature, label, test_size=0.2, random_state=0)

center = x_train[0][0]
left = x_train[0][1]
right = x_train[0][2]
print(y_train[0])

plt.imshow(center)
plt.imshow(left)
plt.imshow(right)
plt.show()

