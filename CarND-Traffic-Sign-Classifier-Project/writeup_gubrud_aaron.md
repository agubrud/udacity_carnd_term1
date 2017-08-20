**Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[class_decomp]: ./visuals/Class_Decomposition.JPG "Class Decomposition"
[class_decomp_chart]: ./visuals/Class_Decomposition_Chart.JPG "Class Decomposition Chart"
[dataset_accuracy]: ./visuals/Train_Val_Test.JPG "Dataset Accuracy"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test/german-roundabout-part.jpg "Roundabout"
[image5]: ./test/german-speed-limit-50-kph.jpg "Speed Limit: 50 KPH"
[image6]: ./test/german-stop-sign.jpg "Stop Sign"
[image7]: ./test/german-straight-or-left.jpg "Straight or Left"
[image8]: ./test/german-yield.jpg "Yield"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
#### Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! My project code will be provided along with my project submisison

#### Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used standard Python list methods to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 px wide * 32 px high * 3 px deep = 3072 elements
* The number of unique classes/labels in the data set is 43 (this is known)

####2. Include an exploratory visualization of the dataset.

I started exploring the datasets by tallying the number of class examples for each dataset. The below image is a decomposition of these datasets facilitated by MS Excel, red and green colors indicating the most and least represented classes respectively. I noticed here that between training, validation, and test sets, each class was approximately equally represented. This is to say that if "class x" accounts for N% of the training set, it will also account for about N% of the validation and test datasets. I view this as favorable because it shouldn't be the case where the training set is composed disproportionately and then the other two datasets are mostly composed of classes that the model hasn't been trained on.

The different classes in the datasets are not uniformly represented, however, which raises some concerns about the trained model developing a bias for the classes with stronger representations.

![alt text][class_decomp]

Graphically, the class distribution looks like this:

![alt text][class_decomp_chart]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

In researching dataset agumentation techniques, I came across a [TensorFlow example](https://stackoverflow.com/questions/37529571/how-is-data-augmentation-implemented-in-tensorflow) of an image classification pipeline that applies a set of preprocessing techniques to each image. The pipeline is as follows:

1. Randomly crop the input image, given a particular crop resolution (using tf.image.random_crop)
2. Randomly flip the input image across the horizontal axis (using tf.image.random_flip_left_right)
3. Randomly adjust the brightness of the image (using tf.image.random_brightness)
4. Randomly adjust the contrast of the image (using tf.image.random_contrast)
5. Subtract the image mean from itself and then normalize with respect to the variance of the pixels (using tf.image.per_image_standardization)

The first step I omitted from my pipeline since the images I take in are already cropped to focus only on the subject of the image. The rest of the steps I incorporated into my own pipeline. One challenge was working this pipeline into my network architecture when the network is designed to work on an arbitrary batch size of input images while the pipeline functions listed can only work on single images. Some research pointed me towards TensorFlow's lambda functions which can apply the preprocessing pipleine to each image in your batch without you having to deal with for loops and the inability to modify a tensor in the normal ways I'm used to modifying a list entry. An example of putting a lambda function to use withing my network architecture is as follows:

x = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x)

In the above example, x is my input tensor with dimensions (batch_size, image_size=32, image_size=32, depth=3) and each image in x is treated with the "per_image_standardization" function.

I did not play with colorspace conversion, but this could be the subject of future work. I ruled out moving to grayscale because I know that traffic sign color can be an important differentiator and it didn't make sense to me to throw that helpful chroma information away.

The image standardization step is an important one and contributed significantly to the model's ability to effectively train itself. Without this step, with the default LeNet hyperparameters, multiple epochs showed negligible progress in tuning the model weights for an effective model.

I decided against generating additional data because the preprocessing pipeline that I've included in the model architecture randomly applies processing which should in practice be fairly similar to augmenting the dataset and then running the entire augmented dataset through the network.

I was hoping to augment my dataset on my home PC and then move it to my AWS node but I was disappointed in the transfer speeds to the AWS node. Since I'm paying for the time taken either to transfer the data or again preprocess the original dataset on the AWS node, I abandoned that technique for the time being.

Ideally I would like to apply some random noise to my images - particularly to the under-represented classes in the datasets to build up a more robust model.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x48 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x96     									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x96 				|
| Fully connected		| input: 2400, output:192      									|
| RELU					|												|
| Fully connected		| input: 192, output:96      									|
| RELU					|												|
| Fully connected		| input: 96, output:43      									|
| RELU					|												|           |


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I reused the same pipeline from the LeNet lab:

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

Results from the network are fed into a softmax that is compared with the known label vectors. The cross entropy is reduced and ged into an Adam optimizer with learning rate of 0.0005. The Adam optimizer then modifies the network weights as it sees fit.

The hyperparameters I played with and the values I settled on are:

layer1 filter depth = 48
layer2 filter depth = 96
layer3 output depth = 192
layer4 output depth = 96
learning rate = 0.0005
epochs = 70
batch size = 64

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were as follows:

![alt text][dataset_accuracy]

It should be noted that since image preprocessing is incorprated into the network architecture, accuracy varies between "evaluate" runs.

* 99.7% accuracy on the training set is very good, although it can be a sign of overfitting to that dataset_accuracy
* 93.1% accuracy on the validation set satisfies the accuracy requirements for this assignment and also shows that the model does generalize
* 91.6% accuracy on the test set again supports generalization

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  * The first architecture that was chosen was the out-of-the-box LeNet-5 model provided in the previous lab, with modifications to input size (since this architecture needs 3 channels) and output class size (43 instead of LeNet's 10).
* What were some problems with the initial architecture?
  * For one, the LeNet-5 architecture as provided in the previous lab does not include any preprocessing of the data. In particular, the absence of the image mean subtraction and variance normalization crippled the architecture from the beginning. The convolutional layer filter depths also proved to be insufficiently deep, since augmenting them lead to accuracy increases. In that lab, only 10 epochs were used, which was also found to be insufficient for training this more complex model.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  * I thought the general architecture of LeNet-5 was suitable and by tweaking hyperparameters, I was encouraged with the capability of the architecture. I didn't experiment with adding more layers, but that could be an area of future work. I read somewhere that dropout is more advantageous in deeper models so I didn't put any time into that method. I also didn't play with the activation function since I was happy with ReLU's performance. I did notice without the random image preprocessing the training model was overfitting which led to my inclusion of those image preprocessing steps into the pipeline.
* Which parameters were tuned? How were they adjusted and why?
  * I would say I spent most of my time playing with layer1 filter depth, layer2 filter depth, layer3 output width, layer4 output width, batch size, and learning rate. The convolutional layer filter depth I thought was important because the differentiating features seemed more complex than those necessary in LeNet-5. For the same reason, I grew layer3 and layer4 output widths. After playing with batch size and learning rate, I started to perceive a relationship between the two, which makes sense. A smaller batch size means the model is updated more frequently and if the learning rate is too large, while the batch size is too small, you can overadjust the weights at each batch and block optimization. Trial and error got me the final batch size of 64 and learning rate of 0.0005 but it was guided by these observations. Modification of the layer depths and widths showed that too drastic of an increase in either will result in a network that quickly plateaued at a certain training error and more modest dimension increases were much more effective.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  * Using convolution layers makes sense because the nature of image data formats lends itself to this technique. By keeping 3 channels, the filters in the convolutional layers should be able to identify both textural patterns and color patterns. Dropout can in theory be helpful in cases where the training model shows signs of overfitting. Instead of dropout, I employed the random image preprocessing steps into my network instead.

If a well known architecture was chosen:
* What architecture was chosen?
  * LeNet-5 from the previous lab
* Why did you believe it would be relevant to the traffic sign application?
  * The videos for this assignment suggested it was a good place to start ;)
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  * Performing "well" is up to the person defining the problem, and in this case the project requires 93% validation set accuracy which was achieved. The training set is clearly performing well with 99.7% accuracy. The training set is performing reasonably well, although a dropoff of ~2% in accuracy vs. the validation dataset indicates some room for improvement. My next move would be either adding new unique images to the test sets to equalize the class representation or doing some noising post processing on some existing images to augment the dataset.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

It should be noted that all of them were cropped approximatley to square perspective and then resized to 32x32 in MS Paint.

The first image I chose because the roundabout sign is one of the least represented classes in the training, validation, and test datasets. For this reason I anticipated there could be some difficulty classifying it if training didn't develop a complete understanding of features making up this class.

The second image I chose was the 50 KPH speed limit sign. I thought this would be a challenging example because although it is represented well as a proportion of the data set contents, there are several other sign types that should look very similar except for the speed limit text.

The third image I chose was a stop sign. The stop sign is middle of the road as far as proportional representation. It does have some background content, however, which could contribute to difficulty in classifying.

The "go straight or left" sign is also one of the least represented classes in the datasets. I also saw some potential struggle with this class because it has a specific directionality. Since my network architecture contains random image flipping, I wasn't sure how that would affect the features the network had learned for this class and how well it would generalize.

The yield sign is an exmaple of a well represented class. I would expect the network to do well with this sign.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Roundabout      		| Roundabout  									|
| Speed Limit 50 KPH     			| Speed Limit 30/80 KPH 										|
| Stop					| Stop											|
| Stright or Left      		| Stright or Left					 				|
| Yield		| Yield      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This seems impressive to me because the validation set saw an accuracy percentage of 93%, but with a much larger number of examples. The sample size is much smaller in this case and since these examples were chosen with the intention of stressing the network performance I would consider 80% accuracy to be fairly successful.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is very sure that this is a yield sign (probability of 100%), and the image does contain a yield sign. The top five soft max probabilities are below. I find it interesting that most of the top 5 candidates for this class have the same color scheme (red/white) which indicate to me the importance of keeping the chroma infomration for this task.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 100%         			| Yield (13)  									|
| 0%     				|  Bumpy road (22)										|
| 0%					| 	Speed limit 50KPH (2)										|
| 0%      			| End of no passing by vehicles over 3.5 metric tons (42)					 				|
| 0%				    | Priority road (12)      							|


For the second image, the model is very sure that this is a roundabout sign (probability of 100%), and the image does contain a roundabout sign. The top five soft max probabilities are below. It's interesting that the top 3 predictions are also circular and predominantly blue. Oddly, the final two are triangular and red/white.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 100%         			|  Roundabout (40)  									|
| 0%     				|  Keep right (38)										|
| 0%					| 	Beware of ice/snow (30)										|
| 0%      			|  Slippery road (23)					 				|
| 0%				    |  Road work (25)      							|

For the third image, the model is pretty sure that this is a 80 KPH speed limit sign (probability of 78%), and the image does not contain a 80 KPH speed limit sign. It is a speed limit sign, but should be 50 KPH. The top five soft max probabilities are below. Notice that all of the predictions belong to the "speed limit" superclass.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 92.1%         			| Speed limit 50 KPH (2)  									|
| 7.9%     				| Speed limit 80 KPH (5)										|
| 0%					| 	Speed limit 70 KPH (4)										|
| 0%      			| Speed limit 30 KPH (1)					 				|
| 0%				    | Speed limit 120 KPH (8)     							|

For the first image, the model is very sure that this is a yield sign (probability of 100%), and the image does contain a stop sign. The top five soft max probabilities are below. I notice a common theme again of red/white grouping.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 100%         			| Stop (14)  									|
| 0%     				| Yield (13)										|
| 0%					| Wild animals crossing (31)										|
| 0%      			| Speed limit 120 KPH (8)					 				|
| 0%				    | No entry (17)      							|

For the first image, the model is very sure that this is a yield sign (probability of 100%), and the image does contain a stop sign. The top five soft max probabilities are below. The children crossing is an interesting outlier, but the rest share the circular theme with predominantly blue/white coloration.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 100%         			| Go straight or left (37)  									|
| 0%     				|  Go straight or right (36)										|
| 0%					| 	Ahead only (35)										|
| 0%      			| Children crossing (28)					 				|
| 0%				    | Keep left (39)      							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I did not end up implementing this...
