
# Behavioral Cloning Project

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

---


### Files

This project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* visualization.py for visualizing the preprocessed data
* utils.py containing the python generator and the preprocessing function
* conifg.py containing hyper parameters and global constants
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* video.py for converting a sequence of images into a video

### Commands
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
For training a model and saving the model to 'model.h5' use
```sh
python model.py
```
For visualization of the dataset and the preprocessing use
```sh
python visualization.py
```


### Model Architecture and Training Strategy

##### Searching for existing networks
I started out with an architecture from [comma.ai](https://github.com/commaai/research/blob/master/train_steering_model.py) and reduced it for my purposes. Since I expected my model to use only few high level features since most of the road is just blank gray with some edge detection on the sides of the road, my convolutional layers become less deeper after each layer.

##### Dataset
The Udacity dataset is used. In the simulation three cameras are installed at the front of the car. One at the center and the other two left and right. All images are used for training. The images are labeled with a steering angle.

##### Data Augmentation
Looking at the given dataset from Udacity, the steering angle is heavily biased to zero. For balancing the data is split into three bags. One for close to zero angles and one for each positive and negative angles. Like in bootstrapping a single batch is sampled from these three bags with the same possibility. This enables the car to drive smoothly on straight lines and also predicting large angles in big turns.
![alt text][image1]
![alt text][image2]
![alt text][image3]

Not only are the center images used but also the images from the left and right cameras. The network has no means to differ from which camera the image come from, a solution as proposed in the NVIDIA paper is to augment side images. This teaches the car to recover from the side to the middle of the road. The augmentation increases for greater steering angles.

The image is resized and the horizon and the lower part of the image which contains the car's hood is cropped out. I convert the image to HSV color space to randomly vary the brightness in a chosen bound to make the model invariant to lighting which is especially important for the second track. Then the image is converted back to RGB. Afterwards the image is flipped with a possibility of 50% to cover turns to the right since most of the data consists of left turns.
![alt text][image4]

At last the image is min-max normalized and scaled to [-1, 1]. This step was essential because otherwise the model got stuck in a local minimum. This resulted in nearly constant steering angle outputs near zero disregarding the input image.

##### Model parameter tuning
In this project most of the parameter tuning was done in the data augmentation step. The learning rate was chosen by the adam optimizer. Since the training data is sampled randomly in a python generator, the number of epochs is rather useless as parameter. What counts is how many images pass through the network. It is currently a value between 40000 and 50000 which is 200-250 batches with a batch size of 200 samples.

##### Prevent overfitting
The model contains dropout layers for all fully connected layers with l2 regularization as standard methods to prevent overfitting.
Since training data is augmented randomly the network will less likely memorize the data.
A small portion of the data (5%) is used to validate the model. The model was also tested on track 2 to check the ability to generalize unknown data.

##### Final Model Architecture

I use convolutional layers and fully connected layers with Relu-activation functions. Dropout and l2-regularization are used to prevent overfitting. A single max pooling layer is used to reduce the size of the network. The output is a single steering angle.

**Input:**  
- 33x160x3 image

**Convolutional layer 1:**
- Convolution: 29x156x32

**Convolutional layer 2:**
- Convolution: 25x152x16
- Max pooling: 5x5x70

**Convolutional layer 3:**
- Convolution: 23x150x8

**Convolutional layer 4:**
- Convolution: 21x148x8
- Max Pooling: 10x74x8
- Flatten: 5920

**Fully connected 1:**
- Fully: 5920x500 (l2-regularization)
- Dropout: 0.2

**Fully connected 2:**
- Fully: 500x500 (l2-regularization)
- Dropout: 0.3

**Fully connected 3:**
- Fully: 500x200 (l2-regularization)
- Dropout: 0.3

**Output layer:**
- Fully: 200x1 (l2-regularization)
- Dropout:0.5

### Car Behavior and Further Work
The car is able to complete track 1 for any number of rounds. To predict big enough steering angles the augmentation factor for recovery images (left and right images) are increased greatly. This introduces slight oscillation especially after recovery. A more sophisticated angle augmentation could be used to produce a smoother drive for straight lines which still can react strong enough to big turns. An exponential angle augmentation could be the solution.

The trained car is also able to drive most of track 2 except for one turn. In this case a big part of the sky is inside the picture which could be responsible for a wrong prediction.
The throttle is currently set to a constant value. Instead I could use some control theory for this task.
