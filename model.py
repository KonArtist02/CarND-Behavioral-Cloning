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
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.models import model_from_json

# Load training data

with open('train.p', 'rb') as f:
    data = pickle.load(f)

X_train = data['images']
y_train = data['angles']

# Shuffle data
X_train, y_train = shuffle(X_train, y_train)

model = Sequential()
model.add(Convolution2D(16, 8, 8, subsample=(2, 2), border_mode="same",input_shape=(32,64,1)))
model.add(ELU())
model.add(Convolution2D(30, 5, 5, subsample=(2, 2), border_mode="same")) #subsample=(2, 2)
model.add(ELU())
model.add(Convolution2D(30, 3, 3, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, batch_size=128, nb_epoch=5, validation_split=0.2)

model_json = model.to_json()
with open("model.json","w") as json_file:
	json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")
