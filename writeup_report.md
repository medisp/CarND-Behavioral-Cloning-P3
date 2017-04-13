#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
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
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

Convolutional Neural Networks have famously used in many deep learning solutions ranging from image recognition, traffic sign classification, and self driving car decision making. My model closely follows the architecture of NVidia and comma.ai.
The final model was a completed during an iterative process of modifying the parameters of the Neural Network architecture and testing performance.
First the image pixel values are normalized between -0.5 and 0.5. Then the Image is cropped to remove the top and bottom of the image, resizing it to 80x320x3. Following this is the first Convolution2D layer with 6x6 kernel window, stride 2x2, 'VALID' padding, depth of 24 and ReLu activation. After this is the second Convolution2D layer with 5x5 kernel window, stride 2x2, 'VALID' padding, depth of 36 and ReLu activation. The third Convolution2D layer has a 5x5 kernel window, stride 2x2, 'VALID' padding, depth of 48 and ReLu activation. The fourth Convolution2D layer has 3x3 kernel window, stride 2x2, 'VALID' padding, depth of 64 and ReLu activation. The fifth and last Convolution2D layer has 3x3 kernel window, stride 2x2, 'VALID' padding, depth of 64 and ReLu activation.This is outputted to a Flatten Layer, and then fed into a Dense Layer of size 200. A Dropout layer follows this with keep_prob=0.5. Another Dense layer of size 100 follows with a Dropout layer keep_prob=0.5.Another Dense layer of size 50 follows with a Dropout layer keep_prob=0.5. Finally Another Dense layer of size 50 follows with a Dropout layer keep_prob=0.5, and a final layer of 1 to predict the steering angle.
Please see the table below for an easier visualization:

| Layer                 | Output Format    | Layer Settings    |
|-----------------------|------------------|-------------------|
| Lambda_Input          | (None,160,320,3) |                   |
| Cropping (Cropping2D) | (None,80,320,3)  | (60,20) , (0,0)   |
| Convolution2D_1       | (None,37,157,24) | (24,6,6) k=(2,2)  |
| Relu_1                | (None,37,157,24) |                   |
| Convolution2D_2       | (None,16,76,36)  | (36,5,5) k=(2,2)  |
| Relu_2                | (None,16,76,36)  |                   |
| Convolution2D_3       | (None,6,36,48)   | (48,5,5) k=(2,2)  |
| Relu_3                | (None,6,36,48)   |                   |
| Convolution2D_4       | (None,4,17,64)   | (64,3,3) k=(2,2)  |
| Relu_4                | (None,4,17,64)   |                   |
| Convolution2D_5       | (None,1,7,128)   | (128,3,3) k=(2,2) |
| Relu_5                | (None,1,7,128)   |                   |
| Flatten               | (None,896)       |                   |
| Dense_1               | (None,200)       |                   |
| Dropout_1             |                  | Keep_Prob=0.5     |
| Dense_2               | (None,100)       |                   |
| Dropout_1             |                  | Keep_Prob=0.5     |
| Dense_3               | (None,50)        |                   |
| Dropout_3             | Keep_Prob=0.5    |                   |
| Dense_4_Final         | (None,1)         |                   |

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

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
