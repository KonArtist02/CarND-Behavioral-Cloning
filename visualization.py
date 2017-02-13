import csv
import pickle
import matplotlib.pyplot as plt
from random import randint
import numpy as np
import config
import utils
import cv2
import config
from keras.models import load_model



## Loading data and initialization
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

# Create one batch with bags
batch_x = np.zeros((config.batch_size,config.img_size_y,config.img_size_x,3),dtype=np.float16)
batch_y = np.zeros(config.batch_size,dtype=np.float16)
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


# Load model
model = load_model('model.h5')
#with open('model.json', 'r') as jfile:
#    model = model_from_json(jfile.read())
#model.compile("adam", "mse")
#weights_file = 'model.h5'
#model.load_weights(weights_file)



## Histogram of steering angles
fig = plt.figure(0)
a = np.histogram(steering_angles_zero_np)
plt.hist(steering_angles_zero_np, range=(-config.angle_thres, config.angle_thres),bins=30)
fig.suptitle('Near zero angles')

fig = plt.figure(1)
a = np.histogram(steering_angles_left_np)
plt.hist(steering_angles_left_np, range=(-1,-config.angle_thres), bins=18)
fig.suptitle('Left angles')

fig = plt.figure(2)
a = np.histogram(steering_angles_right_np)
plt.hist(steering_angles_right_np, range=(config.angle_thres,1),bins=18)
fig.suptitle('Right angles')

## Images with predicted angles
fig = plt.figure(3,figsize=(15,10))
fig.suptitle('Processed images with predicted angles', fontsize=16)
for i in range(16):
    rand = randint(0,config.batch_size-1)
    image_array = batch_x[rand,:,:,:]
    transformed_image_array = image_array[None, :, :, :]
    steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
    sub = plt.subplot(4,4,i+1)
    sub.set_title(str(steering_angle))
    plt.imshow(batch_x[rand,:,:,:])
    #print (batch_x[rand,:,:,:])


## Plot same image with randomized preprocessing
plt.figure(4)
for i in range(48):
    img = cv2.imread('./Record/udacity_data/' + img_paths_left[42],cv2.IMREAD_COLOR) 
    img = utils.process_image(img)
    sub = plt.subplot(6,8,i+1)
    plt.imshow(img)

#plt.figure(5)
#rand = randint(0, len(steering_angles_zero)-1)
#img = cv2.imread('./Record/udacity_data/' + img_paths_zero[rand],cv2.IMREAD_COLOR)
#img = utils.process_image(img)
#plt.imshow(img)


plt.show()