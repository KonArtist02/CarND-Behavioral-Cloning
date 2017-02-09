import csv
import pickle
import matplotlib.pyplot as plt
from random import randint
import numpy as np
import config
import utils
import cv2
import config
from keras.models import model_from_json
#csv_path = '/home/hanqiu/Udacity/CarND-Behavioral-Cloning/Record/udacity_data/driving_log.csv'
#
#
#
#with open(csv_path,'r') as csvfile:
#	csv_reader = csv.reader(csvfile, delimiter=',',skipinitialspace=True)
#	for csv_length,row in enumerate(csv_reader):
#		pass
#	csvfile.close()
#
#with open('train.p', 'rb') as f:
#    data = pickle.load(f)
#
#X_train = data['images']
#y_train = data['angles']
#
#print ('Input data shape: ', X_train.shape)
#print (X_train[0].shape)
#
#plt.figure(figsize=(15,10))
#for i in range(48):
#    rand = randint(0,csv_length)
#    sub = plt.subplot(6,8,i+1)
#    sub.set_title(str(y_train[rand]))
#    
#    plt.imshow(X_train[rand,:,:,:])
#
#plt.figure(figsize=(15,10))
#for i in range(48):
#    rand = randint(0,csv_length)
#    sub = plt.subplot(6,8,i+1)
#    sub.set_title(str(y_train[rand]))
#    
#    plt.imshow(X_train[rand,:,:,:])
#
#plt.show()


## Histogram

csv_path = '/home/hanqiu/Udacity/CarND-Behavioral-Cloning/Record/udacity_data/driving_log.csv'

steering_angles_zero, img_paths_zero, steering_angles_right, img_paths_right, steering_angles_left, img_paths_left = utils.read_csv(csv_path)

steering_angles_zero_np = np.zeros(len(steering_angles_zero),dtype=np.float16)
steering_angles_left_np = np.zeros(len(steering_angles_left),dtype=np.float16)
steering_angles_right_np = np.zeros(len(steering_angles_right),dtype=np.float16)

for i in range(len(steering_angles_zero)):
    steering_angles_zero_np[i] = steering_angles_zero[i]
for i in range(len(steering_angles_left)):
    steering_angles_left_np[i] = steering_angles_left[i]
for i in range(len(steering_angles_right)):
    steering_angles_right_np[i] = steering_angles_right[i]

plt.figure(0)
a = np.histogram(steering_angles_zero_np)
plt.hist(steering_angles_zero_np, bins='auto')

plt.figure(1)
a = np.histogram(steering_angles_left_np)
plt.hist(steering_angles_left_np, bins='auto')

plt.figure(2)
a = np.histogram(steering_angles_right_np)
plt.hist(steering_angles_right_np, bins='auto')


batch_x = np.zeros((config.batch_size,config.img_size_y,config.img_size_x,3),dtype=np.float16)
batch_y = np.zeros(config.batch_size,dtype=np.float16)

assert (len(steering_angles_zero) == len(img_paths_zero))
#print(len(img_paths_right))



for i in range(config.batch_size):
    bag = randint(1,2)
    if (bag == 0):
        rand = randint(0, len(steering_angles_zero)-1)
        img = cv2.imread('./Record/udacity_data/' + img_paths_zero[rand],cv2.IMREAD_COLOR)
        img = utils.process_image(img)
        batch_x[i] = img
        batch_y[i] = steering_angles_zero[rand]
    if (bag == 1):
        rand = randint(0, len(steering_angles_right)-1)
        img = cv2.imread('./Record/udacity_data/' + img_paths_right[rand],cv2.IMREAD_COLOR)
        img = utils.process_image(img)
        batch_x[i] = img
        batch_y[i] = steering_angles_right[rand]
    if (bag == 2):
        rand = randint(0, len(steering_angles_left)-1)
        img = cv2.imread('./Record/udacity_data/' + img_paths_left[rand],cv2.IMREAD_COLOR) 
        img = utils.process_image(img)
        batch_x[i] = img
        batch_y[i] = steering_angles_left[rand]

with open('model.json', 'r') as jfile:
    model = model_from_json(jfile.read())
model.compile("adam", "mse")
weights_file = 'model.h5'
model.load_weights(weights_file)

plt.figure(3,figsize=(15,10))
for i in range(16):
    rand = randint(0,config.batch_size-1)
    image_array = batch_x[rand,:,:,:]
    transformed_image_array = image_array[None, :, :, :]
    steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
    sub = plt.subplot(4,4,i+1)
    sub.set_title(str(steering_angle))
    plt.imshow(batch_x[rand,:,:,:])
    print (batch_x[rand,:,:,:])


plt.figure(4)
for i in range(48):
    img2 = cv2.imread('./Record/udacity_data/' + img_paths_left[42],cv2.IMREAD_COLOR) 
    img2 = utils.process_image(img2)
    sub = plt.subplot(6,8,i+1)
    plt.imshow(img2)

plt.figure(5)
rand = randint(0, len(steering_angles_zero)-1)
img = cv2.imread('./Record/udacity_data/' + img_paths_zero[rand],cv2.IMREAD_COLOR)
img = utils.process_image(img)
plt.imshow(img2)



plt.show()