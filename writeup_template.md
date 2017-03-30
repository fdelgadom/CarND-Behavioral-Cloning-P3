#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
* drive.py for driving the car in autonomous mode with speed modification
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

It uses a modified version of the generator suggested in item 17. It seems that it is considered good practice, to perform the processes of increase of data, outside the generator, but in this case, the size of the batch is large and I use 4 images, two of them coming from the left and right cameras and their flipped ones.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I have used from the beginning the NVIDIA architecture of item 14. It is a tested architecture and it has the feature that requires a reduced number of epochs.

The data is normalized in the model using a Keras lambda layer followed by a cropping layer to obtain the relevant portion of the image. NVIDIA architecture consists of five convolutional layers with RELU activation to introduce nonlinearity, and four full connected layers.

####2. Attempts to reduce overfitting in the model

The original NVIDIA architecture does not contains dropout layers, and overfitting is prevented by a very reduced number of epochs and the data-augmentation techniques. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

The general approach for this project was to use the provided training data from Udacity and as many data augmentation techniques as necessary until successfully steer around the first track in the simulator. Tested in the second track, the car is able to drive itself a portion. In next days, I will try to pass second track changing the steering correction parameter and also in the drive.py script reducing the speed.  

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to confirm that a NVIDIA architecture with data augmentation techniques and the use of the right and left cameras was a good solution.

My first step was to test the right and left cameras instead of the central one, as suggested in https://hackernoon.com/training-a-deep-learning-model-to-steer-a-car-in-99-lines-of-code-ba94e0456e6a and confirmed in several slack and forums messages. This method worked surprisingly well from the beginning and from this moment, it was to go testing and adding data augmentation techniques and varying the steering correction factor.

I tested and discarded brightness adjustment a technique sugested by https://medium.com/@subodh.malgonde/teaching-a-car-to-mimic-your-driving-behaviour-c1f0ae543686 and https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9 that yield no improvements in the result

Finally the key technique was Flipping Images but applied to the left and right cameras instead of the central one. With just this technique te model perform almost flawlessly. 

Also, in order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that the model had a low mean squared error on the training set and validation set. This implied that the model was not overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. At low speed it was perfect, but I decided to increase at a medium level (set_speed = 18) At this speed, the car need a higher steering-correction-factor (0,25) and there are a few spots where the vehicle almost fell off the track but is able to recover.

The final result is that the vehicle is able to drive autonomously around the track without leaving the road at half speed.

####2. Final Model Architecture

The final model architecture it is a convolution neural network proposed by NVIDIA consisted of a normalization lambda layer, a cropping layer, five convolutional layers with RELU activation to introduce nonlinearity, and four full connected layers.


####3. Creation of the Training Set & Training Process

I have used Udacity sample data for track 1, and inside the generator the data set it is 

and in


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
