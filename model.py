import csv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import numpy as np
import os
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
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
from keras.callbacks import TensorBoard

# Read csv file and split data into three bags with left, right and nearly zero steering angles
csv_path = '/home/hanqiu/Udacity/CarND-Behavioral-Cloning/Record/udacity_data/driving_log.csv'
steering_angles_zero, img_paths_zero, steering_angles_right, img_paths_right, steering_angles_left, img_paths_left = utils.read_csv(csv_path)

# Shuffle and split data into train and test sets
steering_angles_zero, img_paths_zero = shuffle(steering_angles_zero, img_paths_zero)
steering_angles_right, img_paths_right = shuffle(steering_angles_right, img_paths_right)
steering_angles_left, img_paths_left = shuffle(steering_angles_left, img_paths_left)

X_train_zero, X_test_zero, y_train_zero, y_test_zero = train_test_split(img_paths_zero, steering_angles_zero, test_size=0.05, random_state=58)
X_train_left, X_test_left, y_train_left, y_test_left = train_test_split(img_paths_left, steering_angles_left, test_size=0.05, random_state=58)
X_train_right, X_test_right, y_train_right, y_test_right = train_test_split(img_paths_right, steering_angles_right, test_size=0.05, random_state=58)

start_time = time.clock()
num_samples = len(steering_angles_zero) + len(steering_angles_right) + len(steering_angles_left)
print ('Training set size: ', len(X_train_zero)+len(X_train_left)+len(X_train_right))
print ('Test set size: ', len(X_test_zero)+len(X_test_left)+len(X_test_right))
print ('Percentage zero angles', len(steering_angles_zero)/float(num_samples))
print ('Percentage left angles', len(steering_angles_left)/float(num_samples))
print ('Percentage right angles', len(steering_angles_left)/float(num_samples))

# Create model
model = Sequential()

model.add(Convolution2D(32, 5, 5,
                        border_mode='valid',
                        input_shape=(config.img_size_y,config.img_size_x,3), name='conv_1') )
model.add(Activation('relu'))
print(model.layers[-1].output_shape)

model.add(Convolution2D(16, 5, 5, name='conv_2'))
model.add(Activation('relu'))
print(model.layers[-1].output_shape)

model.add(Convolution2D(8, 3, 3, name='conv_3'))
model.add(Activation('relu'))
print(model.layers[-1].output_shape)

model.add(Convolution2D(8, 3, 3, name='conv_4'))
model.add(Activation('relu'))
print(model.layers[-1].output_shape)

model.add(MaxPooling2D(name='max_pool'))
print(model.layers[-1].output_shape)

model.add(Flatten(name='flatten'))
model.add(Dropout(0.2, name='dropout_1'))
print(model.layers[-1].output_shape)

model.add(Dense(500, W_regularizer=l2(0.01), name='fully_1'))
model.add(Activation('relu'))
model.add(Dropout(0.3, name='dropout_2'))
print(model.layers[-1].output_shape)

model.add(Dense(500, W_regularizer=l2(0.01), name='fully_2'))
model.add(Activation('relu'))
model.add(Dropout(0.3, name='dropout_3'))
print(model.layers[-1].output_shape)

model.add(Dense(200, W_regularizer=l2(0.01), name='fully_3'))
model.add(Activation('relu'))
model.add(Dropout(0.5, name='dropout_4'))
print(model.layers[-1].output_shape)

model.add(Dense(1, name='output'))

model.compile(optimizer="adam", loss="mse")


#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.10, random_state=42) 
#model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=3, validation_split=0.2)

# Create call back for tensorboard
tb_callback = TensorBoard(log_dir='./logs/02', write_graph=True, write_images=False) #histogram_freq=1


history = model.fit_generator(
		utils.myGenerator(y_train_zero, X_train_zero, y_train_right, \
						X_train_right, y_train_left, X_train_left, config.batch_size), \
		samples_per_epoch=10000, nb_epoch=4, callbacks=[tb_callback], \
		validation_data=utils.myGenerator(y_test_zero, X_test_zero, y_test_right,X_test_right, 
		y_test_left, X_test_left, config.batch_size), nb_val_samples=config.batch_size*4)

print ("Training Time: ", time.clock()-start_time)
#print(history.history.keys())

# Save model and weights
model.save('model.h5')
#model_json = model.to_json()
#with open("model.json","w") as json_file:
#	json_file.write(model_json)
#
#model.save_weights("model.h5")
print("Saved model to disk")



