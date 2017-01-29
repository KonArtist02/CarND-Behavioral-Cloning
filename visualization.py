import csv
import pickle
import matplotlib.pyplot as plt
from random import randint

csv_path = '/home/hanqiu/Udacity/CarND-Behavioral-Cloning/Record/udacity_data/driving_log.csv'



with open(csv_path,'r') as csvfile:
	csv_reader = csv.reader(csvfile, delimiter=',',skipinitialspace=True)
	for csv_length,row in enumerate(csv_reader):
		pass
	csvfile.close()

with open('train.p', 'rb') as f:
    data = pickle.load(f)

X_train = data['images']
y_train = data['angles']

print (X_train[0].shape)

plt.figure(figsize=(15,10))
for i in range(48):
    rand = randint(0,csv_length)
    sub = plt.subplot(6,8,i+1)
    sub.set_title(str(y_train[rand]))
    
    plt.imshow(X_train[rand,:,:,:])

plt.figure(figsize=(15,10))
for i in range(48):
    rand = randint(0,csv_length)
    sub = plt.subplot(6,8,i+1)
    sub.set_title(str(y_train[rand]))
    
    plt.imshow(X_train[rand,:,:,:])

plt.show()

