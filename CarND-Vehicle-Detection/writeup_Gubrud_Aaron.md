**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/object_classes.jpg
[image2a]: ./examples/car_hog_y.jpg
[image2b]: ./examples/notcar_hog_y.jpg
[image2c]: ./examples/car_hog_cr.jpg
[image2d]: ./examples/notcar_hog_cr.jpg
[image2e]: ./examples/car_hog_cb.jpg
[image2f]: ./examples/notcar_hog_cb.jpg
[input_img]: ./test_images/test6.jpg
[clipped_heat]: ./examples/clipped_heat.jpg
[pipeline_out_1]: ./output_images/test1.jpg
[pipeline_out_2]: ./output_images/test2.jpg
[pipeline_out_3]: ./output_images/test3.jpg
[pipeline_out_4]: ./output_images/test4.jpg
[pipeline_out_5]: ./output_images/test5.jpg
[pipeline_out_6]: ./output_images/test6.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

Yer lookin at it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook `Vehicle-Detection.ipynb`).  

I started by reading in all the images of `cars` and the images images without `cars`.  Here is an example of one of each of the `cars` and `notcar` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`, ).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2a]
![alt text][image2b]
![alt text][image2c]
![alt text][image2d]
![alt text][image2e]
![alt text][image2f]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and even out-of-the-box HoG parameters suggested in the class videos worked pretty well (orient=9, pixels_per_cell=8x8, cells_per_block=2). Using the SVM linear classifier described in the next section, I found that even if I played with these parameters, I still would get upwards of 98% test accuracy. I noticed that playing with the number of orientations led to a large variation in processing time so I decided to keep that parameter small since the SVM classifier would end up training well regardless. I even found that larger numbers of orientations would lead to worse vehicle detection in the end, particularly for falsely identifying vehicle patches.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained and tested a linear SVM using `LinearSVM` from `sklearn.svm`. You can see this in cell 5 from line 65 until the end of the cell. I found the performance to be pretty satisfactory out-of-the-box and didn't require any parameter tuning to get something workable.

On my development system (Intel Core i7-3770K w/ 16GB DRAM) training time took around 0.25 seconds (based on 1000 images of cars and 1000 images without cars).

Since I have some randomization built into splitting my training and test set, exact test accuracy can vary. It tends to be above 99% which sounds impressive until the classifier gets dropped into the real pipeline (more on that later).

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to reuse the `slide_window` function from the "Search and Classify" unit of the project videos to handle my sliding window functionality. This function takes in the following arguments:

```python
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5))
```

You can specify x and y offsets, a window size, and how much you want each of your sliding windows to overlap eachother with the arguments above. The default values for these functions are seen in the snippet above. You can find the code for `slide_window` in cell 4 starting on line 99.

I found that a y range of 400-700 tended to work well. This deals with unnecessary processing above where the horizon commonly lies in the frame.

I chose an offset of 250 for my x range because (at least in the US) this is where you'll find oncoming traffic and my classifier liked to pickup false positives over there for some reason.

`slide_window` provides a list of windows that will pass patches to the linear SVM classifier I described earlier. Iterating through these slide window proposals is handled by the function `search_windows` which also comes from the "Search and Classify" unit of the project videos. Its arguments look like this:

```python
def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):
```

`search_windows` is also from the "Search and Classify" unit. It starts on line 187 of cell 4. It takes in your input image, your classifer, a scale factor (if desired), the color space you want to process your image within, spatial and histogram bin sizes, histogram range, and the HoG parameters discussed earlier. It iterates through the list of windows generated by `slide_window` and for each patch it extracts the HoG features, spatial features, and histogram features to be fed into the trained classifier so it can predict whether the patch in question contains a vehicle. The feature extraction phase calls another function `single_img_features`. Within `search_windows`, the scaler function from `sklearn.preprocessing` scales the feature vector before passing it to the classifier.

`single_img_features` is also from the "Search and Classify" unit. It starts on line 140 of cell 4. It propagates the HoG parameters, spatial binning parameters, and histogram binning parameters and actually carries out each of those tasks, resulting in a feature vector. It also handles the input image normalization task.

`search_windows` returns a list of so-called "hot-windows" which are believed to contain cars. I iterate over each of these hot windows and start populating a heatmap. I then define a heatmap confidence level which determines how many "votes" on the heatmap I must have before I truly consider that region to contain a car. I found a thresholded of 25% of the maximum heatmap depth to be sufficient enough. An example of the threholded heatmap can be seen below:

![alt text][input_img]
![alt text][clipped_heat]

Using the label function of `sklearn.ndimage.measurements` I extract labels from my heatmap indicating where there are sufficient car votes. I reuse the method of expanding a single bounding box around a cluster of votes as was seen in the "HoG Subsampling Window Search" project section. This is in lines 50-59 of cell 9.

Finally, I'm provided with bounding boxes around each of my detected cars.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][pipeline_out_1]
![alt text][pipeline_out_2]
![alt text][pipeline_out_3]
![alt text][pipeline_out_4]
![alt text][pipeline_out_5]
![alt text][pipeline_out_6]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to project_video.mp4](./output_images/project_video.mp4)
Here's a [link to test_video.mp4](./output_images/test_video.mp4)

In the test video, you can see that the pipeline does a pretty good job of determining which regions contain cars and which dont.

Exapnding to the longer, more difficult project video, it becomes clear that there are still some details to iron out before the solution would be considered complete enough to be used in a real car.

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The slide_window function takes in an argument which allows me to set how much I want my windows to overlap. I found that 75% overlap between windows worked well and this provides the basis for the voting scheme that I described in a previous section of my writeup. Since each window votes for whether it contains a car and I add each windows' vote to a heatmap I can choose which regions are most likely to contain a car. With my heatmap confidence I specify the lower bound for how confident my heamap should be that a given region has a car based on the number of votes. As described previously, the regions with the adequate amount of votes are then consumed by `scipy.ndimage.measurements.label()` which identifies individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a frame and the resulting bounding boxes then overlaid on the that frame:

### Here are six frames and their corresponding heatmaps:

![alt text][clipped_heat]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][pipeline_out_6]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest issue I faced in this project was false positives. I found early on that the HoG features, spatial binning, and histogram binning were all effective at leading to feature vectors that classified the training set well. When applying that model to the test images for the whole road view, I would often get false positives. I think this could be remedied by training on more images, prefereably from a different data set. I also experimented with a convolutional neural network approach that learns the necessary features itself which seemed to perform better than the HoG classifier approach when using the same training set.

Unfortunately I didn't have the time to experiment with processing across frames but I think this could be effective. One challenge I could see with this, however, is that vehicles traveling on roadways can be moving quite fast and that could pose some challenges to how tolerant you region of interest is and where you expect things to be in the next frame.

As in other projects for this course, lighting and weather conditions are ideal and I'm not sure how well my current pipeline would stand up to changes in either. One other thing I thought of is that all of these clips are on a freeway/highway so onconming traffic isn't accounted for. Depending on how you define your problem you may want to account for those vehicles or you may not. I also didn't notice any pictures of oncoming vehicles in the test set. Motorcycles were also absent from the testing set so I wouldn't expect my pipeline to deal with them very well either. Even trucks don't seem to be a part of the dataset so that would also be a challenge.

From the experimentation I did, the two things I would do to make my pipeline more robust would be to switch to the CNN model instead of HoG and then also implement some logic that accounts for sequential frames in a video. To make my solution more robust, I would want to gather many more training images and be sure to represent all types of vehicles I would expect to see, along with all types of roads I would expect to see, and all types of weather/lighting conditions I would expect to deal with.
