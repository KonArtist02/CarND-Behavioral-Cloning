
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./doc_images/0_left_angles.png "Left angles"
[image2]: ./doc_images/0_near_zero_angles.png "Near zero angles"
[image3]: ./doc_images/0_right_angles.png  "Right angles"
[image4]: ./doc_images/1_preprocessing.png "Preprocessing and predicted angles"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* visualization.py for visualizing the preprocessed data
* utils.py containing the python generator and the preprocessing function
* conifg.py containin hyper parameters and global constants
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results
* video.py for converting a sequence of images into a video

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model consists of four convolutional layers and four fully connected layers. Deeper models have been tested but the current network is sufficient for the task.


####2. Attempts to reduce overfitting in the model

The model contains dropout layers for all fully connected layers with l2 regularization as standard methods to prevent overfitting. 
Since training data is augmented randomly the network will less likely memorize the data. 
A small portion of the data (5%) is used to validate the model. The model was also tested on track 2 to check the ability to generalize unknown data.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

## Searching for existing networks
I started out with an architecture from [comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py) and reduced it for my purposes. Since I expected my model to use only few high level features since most of the road is just blank gray with some edge detection on the sides of the road, my convolutional layers become less deeper after each layer.

## Data Augmentation
Looking at the given dataset from Udacity, the steering angle is heavily biased to zero. For balancing the data is split into three bags. One for close to zero angles and one for each positive and negative angles. Like in bootstrapping a single batch is sampled from these three bags with the same possibility. This enables the car to drive smoothly on straight lines and also predicting large angles in big turns.
![alt text][image1]
![alt text][image2]
![alt text][image3]

Not only are the center images used but also the images from the left and right cameras. The network has no means to differ from which camera the image come from, a solution as proposed in the NVIDIA paper is to augment side images. This teaches the car to revover from the side to the middle of the road. The augmentation increases for greater steering angles. 

The image is resized and the horizon is cropped out. I convert the image to HSV color space to randomly vary the brightness in a chosen bound to make the model invariant to lighting which is especially import for the second track. Then the image is converted back to RGB. Afterwards the image is flipped with a possiblity of 50% to cover turns to the right since most of the data consists of left turns. 

At last the image is min-max normalized and scaled to [-1, 1]. This step was essential because otherwise the model got stuck in a local minimum. This resulted in nearly constant steering angle outputs near zero disregarding the input image.

## Hyper Parameters
In this project most of the parameter tuning was done in the data augmentation step. The learning rate was chosen by the adams optimizer. Since the training data is sampled randomly in a python generator, the number of epochs is rather useless as parameter. What counts is how many images pass through the network. It is currently a value between 40000 and 50000 which is 200-250 batches with a batch size of 200 samples.


####2. Final Model Architecture

##Input: ##  
- 33x160x3 image

##Convolutional layer 1: ##  
- Convolution: 29x156x32

##Convolutional layer 2: ##  
- Convolution: 25x152x16
- Max pooling: 5x5x70

##Convolutional layer 3: ##
- Convolution: 23x150x8

##Convolutional layer 4: ##    
- Convolution: 21x148x8
- Max Pooling: 10x74x8
- Flatten: 5920

##Fully connected 1: ##
- Fully: 5920x500 (l2-regulatization)
- Dropout: 0.2

##Fully connected 2: ##
- Fully: 500x500 (l2-regulatization)
- Dropout: 0.3

##Fully connected 3: ##
- Fully: 500x200 (l2-regulatization)
- Dropout: 0.3

##Output layer: ##  
- Fully: 200x1 (l2-regulatization)
- Dropout:0.5

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
