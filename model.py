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
import utils
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2, activity_l2

# TO-DO:
# use three different bags for balancing data
# normalization


csv_path = '/home/hanqiu/Udacity/CarND-Behavioral-Cloning/Record/udacity_data/driving_log.csv'

steering_angles_zero, img_paths_zero, steering_angles_right, img_paths_right, steering_angles_left, img_paths_left = utils.read_csv(csv_path)


start_time = time.clock()

model = Sequential()

model.add(Convolution2D(32, 5, 5,
                        border_mode='valid',
                        input_shape=(config.img_size_y,config.img_size_x,3)))
print(model.layers[-1].output_shape)

model.add(Activation('relu'))

model.add(Convolution2D(16, 5, 5))
print(model.layers[-1].output_shape)

model.add(Activation('relu'))

model.add(Convolution2D(8, 3, 3))
print(model.layers[-1].output_shape)

model.add(Activation('relu'))

model.add(Convolution2D(8, 3, 3))
#print(model.layers[-1].output_shape)

model.add(Activation('relu'))

model.add(MaxPooling2D())
print(model.layers[-1].output_shape)

model.add(Dropout(0.25))


model.add(Flatten())
print(model.layers[-1].output_shape)

model.add(Dense(500, W_regularizer=l2(0.01)))
print(model.layers[-1].output_shape)

model.add(Activation('relu'))

model.add(Dense(500, W_regularizer=l2(0.01)))
print(model.layers[-1].output_shape)

model.add(Activation('relu'))

model.add(Dense(200, W_regularizer=l2(0.01)))
print(model.layers[-1].output_shape)

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

num_samples = len(img_paths_zero)+len(img_paths_left)+len(img_paths_right)

#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.10, random_state=42) 
#model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=3, validation_split=0.2)
model.fit_generator(utils.myGenerator(steering_angles_zero, img_paths_zero, steering_angles_right, \
	img_paths_right, steering_angles_left, img_paths_left, config.batch_size), samples_per_epoch=5000, nb_epoch=5)

print ("Training Time: ", time.clock()-start_time)

model_json = model.to_json()
with open("model.json","w") as json_file:
	json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")



