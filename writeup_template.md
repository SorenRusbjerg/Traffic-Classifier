# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/barplot.jpg "Visualization"
[image2]: ./examples/grayscale2.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/Sign1.jpeg "Traffic Sign 1"
[image5]: ./examples/Sign3.jpeg "Traffic Sign 2"
[image6]: ./examples/Sign6.jpeg "Traffic Sign 3"
[image7]: ./examples/Sign7.jpeg "Traffic Sign 4"
[image8]: ./examples/Sign8.jpeg "Traffic Sign 5"
[image9]: ./examples/Sign9.jpeg "Traffic Sign 6"
[image10]: ./examples/Sign10.jpeg "Traffic Sign 7"
[image11]: ./examples/Sign11.jpeg "Traffic Sign 8"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://https://github.com/SorenRusbjerg/Traffic-Classifier/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 34799
* The size of the validation set is ? 4410
* The size of test set is ? 12630
* The shape of a traffic sign image is ? (32, 32)
* The number of unique classes/labels in the data set is ? 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data statistics, with labels on the x-axis,
and occurences on the y-axis.

![alt text][image1]

In the notebook html [project html](https://github.com/SorenRusbjerg/Traffic-Classifier/Traffic_Sign_Classifier.html)
6 images from the traning set is also visualized.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I started out by reading the paper [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). I used this paper as inspiration to my NN. Therefore as a first step, I decided to convert the images to grayscale. 

Here is an example of a traffic sign image after grayscaling.

![alt text][image2]

As a last step, I normalized the image data to have zero mean and unit standard deviation. This should make the data 
easier to train. 

I decided to not generate additional data in the first run, and only do this if needed to improve the test score, even though it for sure would help improve the score. 

If needed, I would use rotational translation, size variation, horizontal and vertical translation, and maybe add some gausian noise.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model *(MyNet)* consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x108 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x108 				|
| Dropout layer			|												|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x108 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x108 				    |
| Dropout layer			|												|
| Flatten	            | Flatten layer 1 and 2 output after dropout layer and concatinate them before to sending them to FC layer 	|
| Fully connected		| output 100   									|
| RELU					|												|
| Dropout layer			|												|
| Fully connected		| output 43   									|
| Softmax				|        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyper parameters:
* Learning rate: 0.001
* Epochs: 20
* Batch size: 128
* Drop-out prob.: 0.5
* Optimizer: Adam



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ? 100.0%
* validation set accuracy of ? 97.7%
* test set accuracy of ? 95.5%

The approach chosen:
* What was the first architecture that was tried and why was it chosen? I chose a model simular to that in the paper, after modification done there, called V2. This proved very good results.
* What were some problems with the initial architecture? I had a lot of issues, when training due to having my weights initialised with a too large variance of 1.0. My model was not able to train at all, until I changed the variance to 0.1.
* How was the architecture adjusted and why was it adjusted? I inserted a dropout layer to prevent overfitting the training data. I also also increased the epochs to 15, to get an even better validation score.
* What are some of the important design choices and why were they chosen? The design choices that had a great effect I think, was having two convolutional layers, with a skip layer, which was fed into the FC layer. Also the addtion of the dropout layer is important, and the Adam optimizer also made the training fast.

If a well known architecture was chosen:
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
As the accuracy kept increasing on both the training and validation up to above 95%, tells me that the NN is a good candidate
for the task.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11] 

The second image might be difficult to classify due to the big size.
The third image becomes skewed when resizing and is not centered.
Fifth image is also skewed, with text in the midle. 
Image 6, 7 and 8 also have text overlaying the image.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).


 
| Label         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection        | Right-of-way at the next intersection     |
| Stop                                  | Stop                               |
| Keep right                            | No passing for vehicles over 3.5 metric tons     |
| Stop                                  | Right-of-way at the next intersection     |
| Traffic signals                       | General caution                    |
| Slippery road                         | Traffic signals                    |
| Speed limit (30km/h)                  | Speed limit (30km/h)               |
| Roundabout mandatory                  | Roundabout mandatory               |

The model was able to correctly guess 4 of the 8 traffic signs, which gives an accuracy of 50%. This compares much worse than seen in the test set. I think having more training samples generated with different translations might have helped here.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in Chapter 1.6.4 in the Ipython notebook.

For the first image, the model is very sure that this is a 'Right-of-way at the next intersection' sign (probability of 1.00), and it is correct. 

The same goes for image 2. Image 3 has a 73% probablity on the wrong sign, and the the right sign is not in top5. 
In general it seems that the NN is very certain on its result often with above 90% probability, no matter if it is right or wrong.

Prediction for the "1" image: 

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        | Right-of-way at the next intersection     |
| 0.00        | Beware of ice/snow                 |
| 0.00        | Pedestrians                        |
| 0.00        | Double curve                       |
| 0.00        | Road narrows on the right          |

Prediction for the "2" image: 

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        | Stop                               |
| 0.00        | Speed limit (60km/h)               |
| 0.00        | Speed limit (50km/h)               |
| 0.00        | Speed limit (80km/h)               |
| 0.00        | Speed limit (30km/h)               |

Prediction for the "3" image: 

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 0.93        | No passing for vehicles over 3.5 metric tons     |
| 0.06        | No passing                         |
| 0.00        | Traffic signals                    |
| 0.00        | Dangerous curve to the right       |
| 0.00        | Right-of-way at the next intersection     |

Prediction for the "4" image: 

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 0.73        | Right-of-way at the next intersection     |
| 0.11        | Roundabout mandatory               |
| 0.11        | Priority road                      |
| 0.02        | Speed limit (50km/h)               |
| 0.01        | Slippery road                      |

Prediction for the "5" image: 

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        | General caution                    |
| 0.00        | Keep right                         |
| 0.00        | Traffic signals                    |
| 0.00        | Wild animals crossing              |
| 0.00        | Slippery road                      |

Prediction for the "6" image: 

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        | Traffic signals                    |
| 0.00        | Dangerous curve to the right       |
| 0.00        | General caution                    |
| 0.00        | Bicycles crossing                  |
| 0.00        | Road narrows on the right          |

Prediction for the "7" image: 

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 0.30        | Speed limit (30km/h)               |
| 0.25        | Speed limit (50km/h)               |
| 0.16        | Stop                               |
| 0.10        | Speed limit (80km/h)               |
| 0.08        | Speed limit (70km/h)               |

Prediction for the "8" image: 

| Probability         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 0.99        | Roundabout mandatory               |
| 0.01        | Priority road                      |
| 0.00        | Vehicles over 3.5 metric tons prohibited     |
| 0.00        | Right-of-way at the next intersection     |
| 0.00        | Speed limit (100km/h)              |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
In the notebook html [project html](https://github.com/SorenRusbjerg/Traffic-Classifier/Traffic_Sign_Classifier.html)
chapter 1.7, Conv layer '1', 108 featuremaps are visualized before it is maxpool'ed. It seems to emphazise the outer edges on the input image some of the featuremaps also seems to emphasize the middle of the sign.  

