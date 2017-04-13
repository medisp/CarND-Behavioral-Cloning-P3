# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/figure1.jpg "Standard Data"
[image2]: ./examples/figure2.jpg "Terrain Change 1"
[image3]: ./examples/figure3.jpg "Terrain Change 2"
[image4]: ./examples/figure4.jpg "Terrain Change 3"
[image5]: ./examples/figure5.jpg "Recovery Steering 1"
[image6]: ./examples/figure6.jpg "Recovery Steering 2"
[image7]: ./examples/figure7.jpg "Recovery Steering 3"
[image8]: ./examples/figure8.png "Training MSE Curve"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Convolutional Neural Networks have famously used in many deep learning solutions ranging from image recognition, traffic sign classification, and self driving car decision making. My model closely follows the architecture of NVidia and comma.ai.

The overall strategy for deriving a model architecture was to iteratively test features of the model and assess accuracy and impact on driving behavior. In Conv2D, things I modified included kernel sizes, strides, padding, and activation. I settled with relu activation as it seemed better than elu.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that added dropout layers in the fully connected layers, I also used different activation functions like elu and relu.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track during sharper turns, terrain changes at the border, and during the dirt path. To improve the driving behavior in these cases, I recorded more data during these specific areas and tried to make the model more powerful to better learn these behaviors.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
The final model was a completed during an iterative process of modifying the parameters of the Neural Network architecture and testing performance. Please refer to model.py for code references.

Here is how my Neural Network model was designed:
First the image pixel values are normalized between -0.5 and 0.5 (line 67). Then the Image is cropped to remove the top and bottom of the image, resizing it to 80x320x3 (line 68). Following this is the first Convolution2D layer with 6x6 kernel window, stride 2x2, 'VALID' padding, depth of 24 and ReLu activation (line 69). After this is the second Convolution2D layer with 5x5 kernel window, stride 2x2, 'VALID' padding, depth of 36 and ReLu activation (line 71). The third Convolution2D layer has a 5x5 kernel window, stride 2x2, 'VALID' padding, depth of 48 and ReLu activation (line 73). The fourth Convolution2D layer has 3x3 kernel window, stride 2x2, 'VALID' padding, depth of 64 and ReLu activation (line 75). The fifth and last Convolution2D layer has 3x3 kernel window, stride 2x2, 'VALID' padding, depth of 64 and ReLu activation (line 76). This is outputted to a Flatten Layer (line 77), and then fed into a Dense Layer of size 200 (line 79). A Dropout layer follows this with keep_prob=0.5 (line 80). Another Dense layer of size 100 (line 81) follows with a Dropout layer keep_prob=0.5 (line 82).Another Dense layer of size 100  (line 83)follows with a Dropout layer keep_prob=0.5 (line 84). Finally Another Dense layer of size 50 follows with a Dropout layer keep_prob=0.5, and a final layer of 1 to predict the steering angle.

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


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 80,82,84). 

The model was trained and validated on randomly sampled data sets to ensure that the model was not overfitting (code line 5,11,18). We have used segments of testing and validation sets, where training was 80% and validation was 20%. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Creation of the Training Set & Training Process

I had a baseline of good driving data from Udacity's hosted dataset. This includes nice center-lane driving without any drastic turns. I included driving data from going on the other side of the track:

![alt text][image1]

This helped follow much of the road and helped benchmark the model and see what improvements can be achieved from a modeling perspective. 

Following this, I recorded more data of specific problem areas that occur, such as when seeing different terrain and boundaries, dirt roads that might trick the model:

![alt text][image2]
![alt text][image3]
![alt text][image4]

I also collected data when the car intentionally had to turn to straighten out and follow the path.

![alt text][image5]
![alt text][image6]
![alt text][image7]

After the collection process, I had (8037 1597) = 9624 images/number of data points. I attempted to preprocess the images with image normalization, brightness normalization, histogram equalization, and affine transformations, but this actually caused my model to accidentally run off the rode, hence I discontinued use of this but included the code in this repo under image_utils.py.

I finally randomly shuffled the data set and put 20% of the data into a validation set. I used generators for making sure the model runs quickly without running out of memory for having loaded nearly 10k images at a time.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary. The ideal number of epochs was 8 as evidenced by the loss curves between training and validation. There were times that training and test dropped together up to epoch 4, where the model overfit rather than underfit, then there was nicer convergence at epoch 8: 

![alt text][image8]


#### 3. Conclusion and Final Thoughts 

The model ran well around Track 1 and could use some other techniques to help the model infer and generalize on track 2. I will try image normalization to see the benefit on this track, as long as the accuracy on track 1 isn't lost.
This was a nice project, using Convolutional Neural Networks, Model building and Testing fundamentals to teach a simulator to drive around a track. Please check out this video for performance, along with video.mp4 in the repo.
https://youtu.be/BUwiPpCjlwM


