import csv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import numpy as np
import os
import pickle
from random import randint
import config

augment_factor = 0.05

images = []
steering_angles = []

csv_path = '/home/hanqiu/Udacity/CarND-Behavioral-Cloning/Record/udacity_data/driving_log.csv'

with open(csv_path,'r') as csvfile:
	csv_reader = csv.reader(csvfile, delimiter=',',skipinitialspace=True)
	for csv_length,row in enumerate(csv_reader):
		pass
	csvfile.close()

#batch_size = 1000
#for offset in range(0, len(y_train), batch_size):
#    # scale image data to [-1,1]
#    X_train[offset:offset+batch_size] = (X_train[offset:offset+batch_size] - 128.)/128.

with open(csv_path,'r') as csvfile:
	csv_reader = csv.reader(csvfile, delimiter=',',skipinitialspace=True)
	
	next(csv_reader, None)
	for row in tqdm(csv_reader,total=csv_length):
		for i in range(3):
			#img = cv2.imread('./Record/udacity_data/' + row[i],cv2.IMREAD_COLOR)
			img = mpimg.imread('./Record/udacity_data/' + row[i])
			img = config.process_image(img,config.resize_x,config.resize_y)

			#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			#img.shape += (1,)
			images.append(img)
			angle = float(row[3])
			steering_angles.append(0)


			#steering_angles.append(angle)
			#if (i == 0):
			#	steering_angles.append(angle)
			#elif (i == 1):
			#	angle = angle * (1 + np.sign(angle) * augment_factor)
			#	if (angle > 1):
			#		angle = 1
			#	elif (angle < -1):
			#		angle = -1
			#	elif (angle == 0):
			#		angle = augment_factor/2
			#	steering_angles.append(angle)
			#elif (i == 2):
			#	angle = angle * (1 - np.sign(angle) * augment_factor)
			#	if (angle > 1):
			#		angle = 1
			#	elif (angle < -1):
			#		angle = -1
			#	elif (angle == 0):
			#		angle = -augment_factor/2
			#	steering_angles.append(angle)

	csvfile.close()


images_np = np.zeros((len(images),config.img_size_y,config.img_size_x,3),dtype=np.uint8)
steering_angles_np = np.zeros(len(images),dtype=np.float16)


for i in range(len(images)):
	images_np[i] = images[i]
	steering_angles_np[i] = steering_angles[i]
#print (steering_angles)
#print(type(images_np))
print(images_np.shape)

pickle_file = 'train.p'

print('Saving data to pickle file...')
try:
    with open('train.p', 'wb') as pfile:
        pickle.dump(
            {
                'images': images_np,
                'angles': steering_angles_np,
            },
            pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

print('Data cached in pickle file.')

plt.figure(figsize=(15,10))
for i in range(48):
    rand = randint(0,csv_length)
    sub = plt.subplot(6,8,i+1)
    sub.set_title(str(steering_angles_np[rand]))
    
    plt.imshow(images_np[rand,:,:,:])

plt.figure(figsize=(15,10))
for i in range(48):
    rand = randint(0,csv_length)
    sub = plt.subplot(6,8,i+1)
    sub.set_title(str(steering_angles_np[rand]))
    
    plt.imshow(images_np[rand,:,:,:])

plt.show()

