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

[center_driving]: ./examples/center_2017_08_27_21_25_27_117.jpg "Center lane driving"
[center_driving_cropped]: ./examples/center_2017_08_27_21_25_27_117_cropped.jpg "Center lane driving cropped"
[recovery_1]: ./examples/center_2017_08_27_21_28_49_482.jpg "Recovery sequence, number 1"
[recovery_2]: ./examples/center_2017_08_27_21_28_50_245.jpg "Recovery sequence, number 2"
[recovery_3]: ./examples/center_2017_08_27_21_28_52_003.jpg "Recovery sequence, number 3"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[first_nvidia_attempt]: ./model_images/model-gold.png "First model to successfully round the test track"
[second_nvidia_attempt]: ./model_images/model-gold-alt.png "More compact version of the nVidia architecture"
[model_gold_alt_10_epoch]: ./examples/model-gold-alt-10-epoch.jpg "10 epochs of training on the gold alt model"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model_trainer.ipynb Jupyter notebook used for model training since I've grown fond of it
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model_trainer.ipynb Jupyter notebook contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model leverages the nVidia architecture presented in the "Even More Powerful Network" video in the project video content. The architecture is as follows:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 320x160x3 (WxHxD) normalized RGB image   							|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 158x78x12 	|
| RELU					|												|
| Dropout					|	25% dropout rate											|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 77x37x18     									|
| RELU					|												|
| Dropout					|	25% dropout rate											|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 37x17x24     									|
| RELU					|												|
| Dropout					|	25% dropout rate											|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 35x15x32     									|
| RELU					|												|
| Fully connected		| input: 16800, output:30      									|
| Fully connected		| input: 30, output:10      									|
| Fully connected		| input: 10, output:1      									|

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. The normalization function is as follows:

model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))

As suggested in the project videos, the architecture also uses the "Cropping2D" Keras layer to crop out the top and bottom regions of the image which contain the sky and the hood respectively.

model.add(Cropping2D(cropping=((70,20), (0,0))))

I originally started with an exact replica of the nVidia architecture but experimentation with the architecture showed that I could get away with reducing the filter depth in each layer (50% reduction in number of filters ended up working!). I haven't read nVidia's paper but I suspect their model is capable of handling more complex types of driving environments than the consistent 3D environment provided by the simulator. By moving from the original architecture to my reduced experiment, the model size reduces significantly. My model is 1.9MB versus the original size of 6.4MB. Both versions are able to complete the test course.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. e.g.

model.add(Dropout(0.25))

The model was trained and validated on different data sets to help combat overfitting. I achieve this with Keras' model.fit() function which allows the programmer to specify a fraction of the training data to be used for validation (validation_split).

model.fit(X_train, y_train, validation_split=0.3, shuffle=True, nb_epoch=10)

Other notable parameters passed to the model.fit() function are the shuffle parameter which scrambles the order of the training data as an overfitting reducing measure and nb_epoch which specifies the number of epochs used for training the model. I found that raising the number of ephochs proved effective in creating more robust models since each epoch has a dropout percentage of 25% for certain layers and multiple passes allowed the model to build up its internal representation iteratively.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually:

model.compile(loss='mse', optimizer='adam')

####4. Appropriate training data

As suggested in the project overview, the strategy for the training process was to do a combination of center line driving and edge recovery. I did some experimenting with adding additional training data that only addressed the problematic areas (e.g. turns) by restricting my training data gathering around those regions. For some regions, I completed the sections counter-clockwise and also clockwise. I also collected a lap of the alternate course to help generalize my model.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to not reinvent the wheel and pay attention to suggestions made by the project videos. Following along with the videos, I started with the LeNet architecture which had somewhat promising initial performance. As I made it through the videos, I wanted to try the nVidia architecture which allowed me to complete my first successful lap around the test course. I referred to this configuration as my "gold" configuration which referred to both my .hd5 model as well as the training data that was fed into it. Although the model led to a technically successful lap, the behavior of the autonomous car was somewhat erratic so I tried adding more training data to my "gold" data set around the areas where it came close to failing. Since this technique had already been employed in my "gold" training set, the additional data around these regions didn't appear to add much to the model and in some cases made autonomous behavior worse.

Due to Keras' validation_split parameter for the model.fit() function, I didn't have to explicitly separate my training and validation sets, but I was able to track the progression of the MSE for each set for every epoch. Generally I found that the MSE of the training and validation sets came down at about the same rates and didn't indicate obvious signs of overfitting. However, many models along the way with encouragingly low MSE reported at the end of the training process (for both sets) still resulted in autonomous behavior that didn't perform to expectations. To me, this is an interesting characteristic of behavior modeling since the feedback you get from your training exercise isn't as actionable as in classification tasks. For instance, you might not be able to say "as long as my training and validation sets achieve MSE of < 0.1, my model is ready for deployment".

Initially the reference nVidia architecture didn't include any droput layers but I decided to experiment with them to combat overfitting. I believe this architectural change is likely one of the biggest contributors to my ability to trim back filter depth at each convolutional layer and still allow the model to generalize.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture can be seen in model_trainer.ipynb with comments and is as follows:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 320x160x3 (WxHxD) normalized RGB image   							|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 158x78x12 	|
| RELU					|												|
| Dropout					|	25% dropout rate											|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 77x37x18     									|
| RELU					|												|
| Dropout					|	25% dropout rate											|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 37x17x24     									|
| RELU					|												|
| Dropout					|	25% dropout rate											|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 35x15x32     									|
| RELU					|												|
| Fully connected		| input: 16800, output:30      									|
| Fully connected		| input: 30, output:10      									|
| Fully connected		| input: 10, output:1      									|

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][second_nvidia_attempt]

The above version was implemented after an initial attempt which just copied the nVidia architecture. The initial attempt is captured below.

![alt text][first_nvidia_attempt]

Code to generate these visualizations can be found in model_trainer.ipynb. Let the record show that I went out of my way to get these visualizations and I'm pretty disappointed in them... Was hoping for something like the Netscope project (http://ethereon.github.io/netscope/quickstart.html). I quickly adapted my network to Caffe's prototxt definition and generated this: http://ethereon.github.io/netscope/#/gist/5053d3e3641c848bb4bb97f2dd722ca4

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center_driving]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct its path if it gets too close to the edge of the road. These images show what a recovery looks like starting from the right side of the road and coming back to the center:

![alt text][recovery_1] ![alt text][recovery_2] ![alt text][recovery_3]

I did a lap repeating this process in addition to a lap of center lane driving.

As mentioned earlier, each training image is also flipped and the both the original and flipped versions constitute the training set. The steering angle list is augmented similarly, mutliplying the measurement by -1.0 to account for the flipping. To augment the data sat, I also flipped images and angles thinking that this would help the model generalize and not over-learn making left hand turns for instance since there is a bias in the track for left vs. right turns.

I mentioned earlier that each input image was cropped. An example of before and after can be seen below:

![alt text][center_driving]
![alt text][center_driving_cropped]

After the collection process, I had 6923 number of data points. I did not do any preprocessing on the data aside from the flipping and cropping augmenting steps I listed previously. With the flipping preprocessing doubling my training set, I arrive at 13,846 training examples.

I did not explicitly separate my training set, but I did use the "validation_split" parameter in Keras' model.fit() function to allocate 30% of my training set for validation purposes.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. Through playing with different epoch values, I found that 10 was a suitable number. Looking at the MSE for training and validation sets across the epochs, improvements seem to level off at around the 10 epoch point. Since I used an ADAM optimizer, tuning the learning rate hyper parameter wasn't necessary.

![alt text][model_gold_alt_10_epoch]

We do see at the end here that the training set does appear to improve with each pass, but the validation set is not following at the same rate which suggests overfitting. This makes 10 epochs a good place to stop so the model does not continue any more on this path.
