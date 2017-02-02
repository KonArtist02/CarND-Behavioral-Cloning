import csv
import config
import numpy as np
from sklearn.utils import shuffle
from random import randint
import cv2

def read_csv(csv_path):
	img_paths_left = []
	img_paths_zero = []
	img_paths_right = []

	steering_angles_left = []
	steering_angles_zero = []
	steering_angles_right = []

	with open(csv_path,'r') as csvfile:
		csv_reader = csv.reader(csvfile, delimiter=',',skipinitialspace=True)
		next(csv_reader, None)
		for csv_length,row in enumerate(csv_reader):
			for i in range(3):
				angle = float(row[3])
				if (i == 0):
					if (abs(angle) <= config.angle_thres):
						steering_angles_zero.append(angle)
						img_paths_zero.append(row[i])
					elif(angle > config.angle_thres):
						steering_angles_right.append(angle)
						img_paths_right.append(row[i])
					elif(angle < -config.angle_thres):
						steering_angles_left.append(angle)
						img_paths_left.append(row[i])
				elif (i == 1):
					angle = angle * (1 + np.sign(angle) * config.augment_factor)
					if (angle > 1):
						angle = 1
					elif (angle < -1):
						angle = -1
					if (angle != 0):
						if (abs(angle) <= config.angle_thres):
							steering_angles_zero.append(angle)
							img_paths_zero.append(row[i])
						elif(angle > config.angle_thres):
							steering_angles_right.append(angle)
							img_paths_right.append(row[i])
						elif(angle < -config.angle_thres):
							steering_angles_left.append(angle)
							img_paths_left.append(row[i])
				elif (i == 2):
					angle = angle * (1 - np.sign(angle) * config.augment_factor)
					if (angle > 1):
						angle = 1
					elif (angle < -1):
						angle = -1
					if (angle != 0):
						if (abs(angle) <= config.angle_thres):
							steering_angles_zero.append(angle)
							img_paths_zero.append(row[i])
						elif(angle > config.angle_thres):
							steering_angles_right.append(angle)
							img_paths_right.append(row[i])
						elif(angle < -config.angle_thres):
							steering_angles_left.append(angle)
							img_paths_left.append(row[i])
		csvfile.close()
	return steering_angles_zero, img_paths_zero, steering_angles_right, img_paths_right, steering_angles_left, img_paths_left

def process_image(img):
	img = cv2.resize(img,dsize=(config.resize_x,config.resize_y))
	img = img[int(config.resize_y*0.43):int(config.resize_y*0.84),:]
	img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	
	brightness_variation = randint(-70,70)
	if(brightness_variation >= 0):
		mask = img[:,:,2] + brightness_variation < 255
		img[:,:,2] = np.where(mask,img[:,:,2]+brightness_variation,255)
	elif(brightness_variation < 0):
		mask = img[:,:,2] + brightness_variation > 0
		img[:,:,2] = np.where(mask,img[:,:,2]+brightness_variation,0)
	img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
	#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


	return img

def myGenerator(steering_angles_zero, img_paths_zero, steering_angles_right, img_paths_right, steering_angles_left, img_paths_left, batch_size):
	while True:
		#steering_angles_zero, img_paths_zero = shuffle(steering_angles_zero, img_paths_zero)
		#steering_angles_right, img_paths_right = shuffle(steering_angles_right, img_paths_right)
		#steering_angles_left, img_paths_left = shuffle(steering_angles_left, img_paths_left)

		batch_x = np.zeros((batch_size,config.img_size_y,config.img_size_x,3),dtype=np.uint8)
		batch_y = np.zeros(batch_size,dtype=np.float16)

		assert (len(steering_angles_zero) == len(img_paths_zero))

		for i in range(batch_size):
			bag = randint(0,2)
			if (bag == 0):
				rand = randint(0, len(steering_angles_zero)-1)
				img = cv2.imread('./Record/udacity_data/' + img_paths_zero[rand],cv2.IMREAD_COLOR)
				img = process_image(img)
				batch_x[i] = img
				batch_y[i] = steering_angles_zero[rand]
			if (bag == 1):
				rand = randint(0, len(steering_angles_right)-1)
				img = cv2.imread('./Record/udacity_data/' + img_paths_right[rand],cv2.IMREAD_COLOR)
				img = process_image(img)
				batch_x[i] = img
				batch_y[i] = steering_angles_right[rand]
			if (bag == 2):
				rand = randint(0, len(steering_angles_left)-1)
				img = cv2.imread('./Record/udacity_data/' + img_paths_left[rand],cv2.IMREAD_COLOR)
				img = process_image(img)
				batch_x[i] = img
				batch_y[i] = steering_angles_left[rand]

		yield (batch_x, batch_y) 



		#num_samples = len(steering_angles)
		#for offset in range(0, num_samples, batch_size):
		#	if (num_samples >= offset + batch_size):
		#		end = batch_size
		#		batch_x = np.zeros((batch_size,config.img_size_y,config.img_size_x,3),dtype=np.uint8)
		#		batch_y = np.zeros(batch_size,dtype=np.float16)
		#	else:
		#		end = num_samples-offset
		#		batch_x = np.zeros((num_samples-offset,config.img_size_y,config.img_size_x,3),dtype=np.uint8)
		#		batch_y = np.zeros(num_samples-offset,dtype=np.float16)
		#	for i in range(offset,offset+end):
		#		img = mpimg.imread('./Record/udacity_data/' + img_paths[i])
		#		img = config.process_image(img,config.resize_x,config.resize_y)
		#		batch_x[i-offset] = img
		#		batch_y[i-offset] = steering_angles[i]
		#	yield (batch_x, batch_y)