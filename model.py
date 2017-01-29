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
import time
from sklearn.model_selection import train_test_split
import config
from keras.layers.pooling import MaxPooling2D

def myGenerator(X_train, y_train, BATCH_SIZE):
	while True:
		X_train, y_train = shuffle(X_train, y_train)
		
		num_samples = y_train.size

		for offset in range(0, num_samples, BATCH_SIZE):
			end = offset + BATCH_SIZE
			batch_x, batch_y = X_train[offset:end], y_train[offset:end]
			yield (batch_x, batch_y)


BATCH_SIZE = 200


# Load training data

with open('train.p', 'rb') as f:
    data = pickle.load(f)

X_train = data['images']
y_train = data['angles']

# Shuffle data
X_train, y_train = shuffle(X_train, y_train)

batch_size = 5000
for offset in range(0, len(y_train), batch_size):
    # scale image data to [-1,1]
    scaling_factor = (X_train[offset:offset+batch_size] - 128.)/128.
    X_train[offset:offset+batch_size] = scaling_factor

start_time = time.clock()

model = Sequential()
model.add(Convolution2D(16, 5, 5, border_mode="same",input_shape=(config.img_size_y,config.img_size_x,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ELU())
print(model.layers[-1].output_shape)

model.add(Convolution2D(30, 5, 5, border_mode="same")) #subsample=(2, 2)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(ELU())
print(model.layers[-1].output_shape)

model.add(Convolution2D(30, 3, 3, border_mode="same"))
model.add(Flatten())
print(model.layers[-1].output_shape)

#model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(500))
print(model.layers[-1].output_shape)

#model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
#model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=5, validation_split=0.2)


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.10, random_state=42) 

model.fit_generator(myGenerator(X_train, y_train, BATCH_SIZE), samples_per_epoch=y_train.size, nb_epoch=5, validation_data=(X_test,y_test))

print ("Training Time: ", time.clock()-start_time)

model_json = model.to_json()
with open("model.json","w") as json_file:
	json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")



