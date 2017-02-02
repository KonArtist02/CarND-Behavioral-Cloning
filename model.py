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

# TO-DO:
# use three different bags for balancing data
# normalization


csv_path = '/home/hanqiu/Udacity/CarND-Behavioral-Cloning/Record/udacity_data/driving_log.csv'

steering_angles_zero, img_paths_zero, steering_angles_right, img_paths_right, steering_angles_left, img_paths_left = utils.read_csv(csv_path)



BATCH_SIZE = 200


# Load training data

#with open('train.p', 'rb') as f:
#    data = pickle.load(f)
#
#X_train = data['images']
#y_train = data['angles']
#
## Shuffle data
#X_train, y_train = shuffle(X_train, y_train)
#
#batch_size = 5000
#for offset in range(0, len(y_train), batch_size):
#    # scale image data to [-1,1]
#    scaling_factor = (X_train[offset:offset+batch_size] - 128.)/128.
#    X_train[offset:offset+batch_size] = scaling_factor

start_time = time.clock()

nb_filters1 = 16
nb_filters2 = 8
nb_filters3 = 4
nb_filters4 = 2

# size of pooling area for max pooling
pool_size = (2, 2)

# convolution kernel size
kernel_size = (3, 3)

# Initiating the model
model = Sequential()

# Starting with the convolutional layer
# The first layer will turn 1 channel into 16 channels
model.add(Convolution2D(nb_filters1, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=(config.img_size_y,config.img_size_x,3)))
print(model.layers[-1].output_shape)
# Applying ReLU
model.add(Activation('relu'))
# The second conv layer will convert 16 channels into 8 channels
model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1]))
print(model.layers[-1].output_shape)
# Applying ReLU
model.add(Activation('relu'))
# The second conv layer will convert 8 channels into 4 channels
model.add(Convolution2D(nb_filters3, kernel_size[0], kernel_size[1]))
print(model.layers[-1].output_shape)
# Applying ReLU
model.add(Activation('relu'))
# The second conv layer will convert 4 channels into 2 channels
model.add(Convolution2D(nb_filters4, kernel_size[0], kernel_size[1]))
print(model.layers[-1].output_shape)
# Applying ReLU
model.add(Activation('relu'))
# Apply Max Pooling for each 2 x 2 pixels
model.add(MaxPooling2D(pool_size=pool_size))
print(model.layers[-1].output_shape)
# Apply dropout of 25%
model.add(Dropout(0.25))

# Flatten the matrix. The input has size of 360
model.add(Flatten())
# Input 360 Output 16
model.add(Dense(16))
print(model.layers[-1].output_shape)
# Applying ReLU
model.add(Activation('relu'))
# Input 16 Output 16
model.add(Dense(16))
print(model.layers[-1].output_shape)
# Applying ReLU
model.add(Activation('relu'))
# Input 16 Output 16
model.add(Dense(16))
print(model.layers[-1].output_shape)
# Applying ReLU
model.add(Activation('relu'))
# Apply dropout of 50%
model.add(Dropout(0.5))
# Input 16 Output 1
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")



#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.10, random_state=42) 
#model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=3, validation_split=0.2)
model.fit_generator(myGenerator(steering_angles_zero, img_paths_zero, steering_angles_right, \
	img_paths_right, steering_angles_left, img_paths_left, config.batch_size), samples_per_epoch=len(img_paths), nb_epoch=5)



print ("Training Time: ", time.clock()-start_time)

model_json = model.to_json()
with open("model.json","w") as json_file:
	json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")



