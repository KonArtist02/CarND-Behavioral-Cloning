import csv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import numpy as np
import os
import pickle
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten


# Load training data

with open('mini_test_set.p', 'rb') as f:
    data = pickle.load(f)

X_train = data['images']
y_train = data['angles']

print("Input shape: ", X_train.shape)
print("Output shape: ", y_train.shape)

# Shuffle data
X_train, y_train = shuffle(X_train, y_train)

model = Sequential()
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"),input_shape=(64,32))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, batch_size=200, nb_epoch=5, validation_split=0.2)