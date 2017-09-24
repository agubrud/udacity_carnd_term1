# Writeup For Aaron Gubrud's Advanced Lane Lines Project

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistorted_img]: ./output_images/calibration1_undistorted.jpg "Undistorted"
[road_orig]: ./output_images/examples/example_orig_test6.jpg "Test 6, Road Original"
[road_undist]: ./output_images/examples/example_undist_test6.jpg "Test 6, Road Undistorted"
[road_perspective_corners]: ./output_images/examples/example_perspective_corners_test6.jpg "Test 6, Road With Perspective Warp Target Points"
[road_perspective_unwarp]: ./output_images/examples/example_perspective_unwarped_test6.jpg "Test 6, Road Unwarped"
[road_threshold]: ./output_images/examples/example_thresholded_test6.jpg "Test 6, HLS Threshold"
[road_threshold_sobel]: ./output_images/examples/example_thresholded_challenge_dark.jpg "Test 6, Sobel Threshold"
[road_lines]: ./output_images/examples/example_lines_test6.jpg "Test 6, Lines"
[road_overlaid]: ./output_images/examples/example_overlaid_test6.jpg "Test 6, Overlaid"
[road_final]: ./output_images/test6.jpg "Test 6, Overlaid"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the code cells 2-4 of the IPython notebook located in "AdvancedLaneLines.ipynb".  

I start with a call to my chessboard_data function which expects an RGB image and the size of the chessboard grid as arguments. In this function, I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I return the objpoints and imgpoints arrays for use in the next stage of the calibration pipeline.

I then feed the output `objpoints` and `imgpoints` to OpenCV's `cv2.calibrateCamera()` function to compute the camera calibration and distortion coefficients.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][undistorted_img]

You can see in the resulting figure that the distortion has been corrected.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][road_orig]

Since I already have the distortion characterized with the variables `mtx` and `dst` (which hold the camera matrix and the distortion coefficients respectively), I can simply provide the image I want to undistort along with these variables to the `cv2.undistort()` function. From this operation, I get the below undistorted road image. (It looks the same unless you a/b them on top of each other but I promise they're different :) ). You can see this in line 23 of code cell 12.

![alt text][road_undist]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I decided to do the perspective transform after undistorting the image because it makes sense to me to restrict the region of interest as early as possible. The code for my perspective transform includes a function called `corners_unwarp`, which appears in code cell 8.  The `corners_unwarp` function takes as inputs an image (`img`), as well as the polygon corners which define my region of interest (`corners`).  I chose the hardcode the corners as follows:

```python
corners = np.float32(
        [
            [430, 500],
            [850, 500],
            [1170, 720],
            [110, 720]               
         ])
```

If I overlay these corners on top of the input image, I get the following:

![alt text][road_perspective_corners]

I then choose the destination points as follows:

```python
xoffset = 150 # offset for dst points
yoffset = 0 # offset for dst points
dst = np.float32([[xoffset, yoffset], [img_size[0]-xoffset, yoffset],
                                 [img_size[0]-xoffset, img_size[1]-yoffset],
                                 [xoffset, img_size[1]-yoffset]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][road_perspective_unwarp]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

My thresholding pipeline takes place in the `pipeline` function in code cell 7. I got a lot of mileage from just converting the input image to the HSL color space with OpenCV and just doing thresholding on those values. I investigated some common characteristics of the test images we were provided and defined the following rules:

If the saturation channel is >= 90 and the lightness channel is >= 100, then we can consider this of interest (a yellow line).
If the saturation channel is < 32 and the lightness channel is >= 85% of the max lightness in the region, then we can consider this of interest as well (a white line).

Since the region of interest has already been restricted to the road in front of the car, the saturation and lightness thresholding proves pretty effective. In fact, with this thresholding alone, I get satisfying results from all of the provided test images. Where I do get into trouble, though, is areas with inconsistent lighting (e.g. particularly light or dark images). Continuing with the example image, here is the region of interest with HLS thresholding:

![alt text][road_threshold]

Since my pipeline does struggle with difficult lighting scenarios, I fall back to an edge-detection based thresholding method when HLS thresholding doesn't produce enough lane line guesses. This is less ideal because it's noisier, as seen in this example:

![alt text][road_threshold_sobel]

Also noticeable with this transform is my choice to mask out a region in the lower middle part of the frame. The reasoning for this is because we don't expect a lane line to be there and it also deals with HOV lane symbols which were providing some problems.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To sort out the polynomial fitting, I followed the guidance of the instructional videos for this project. This happens in the function called `compute_lane_line_polynomials` which lives in code cell 10 and takes in the binary thresholded representation of the image. With the binary thresholded image, I start with a histogram of the columns of the image data. The peaks of this histogram data provide the starting points for the line approximation. From there, I use a sliding window to find where to continuation of the left and right lines lie as I move upwards in the frame. I chose to stick with the project video's suggested 9 vertical windows. I also tried playing a bit with the minimum number of pixels required to recenter the window, eventually settling on 100.

This worked well for frames under ideal lighting conditions, but edge cases urged for error handling. At this point in the project, I also start using the `Line()` class, as suggested by the project's tips. I use this class to keep track of all of the x and y coordinates I've associated with the lane lines:

```python
    if (len(left_line.allx) < 2):
        left_line.allx = np.copy(leftx)
    else:
        left_line.allx = np.append(left_line.allx, leftx, axis=0)

    if (len(right_line.allx) < 2):
        right_line.allx = np.copy(rightx)
    else:
        right_line.allx = np.append(right_line.allx, rightx, axis=0)

    if (len(left_line.ally) < 2):
        left_line.ally = np.copy(lefty)
    else:
        left_line.ally = np.append(left_line.ally, lefty, axis=0)

    if (len(right_line.ally) < 2):
        right_line.ally = np.copy(righty)
    else:
        right_line.ally = np.append(right_line.ally, righty, axis=0)   
```

I use this information later, taking all x and y coordinates into account for the last 5 frames.

```python
n_units = 5
if (len(left_line.len_per_batch_x) <= n_units):
    go_back_n_leftx = np.sum(left_line.len_per_batch_x)
    go_back_n_lefty = np.sum(left_line.len_per_batch_y)
    go_back_n_rightx = np.sum(right_line.len_per_batch_x)
    go_back_n_righty = np.sum(right_line.len_per_batch_y)
else:
    go_back_n_leftx = np.sum(left_line.len_per_batch_x[len(left_line.len_per_batch_x)-n_units:])
    go_back_n_lefty = np.sum(left_line.len_per_batch_y[len(left_line.len_per_batch_y)-n_units:])
    go_back_n_rightx = np.sum(right_line.len_per_batch_x[len(right_line.len_per_batch_x)-n_units:])
    go_back_n_righty = np.sum(right_line.len_per_batch_y[len(right_line.len_per_batch_y)-n_units:])

if (go_back_n_leftx == 0 or go_back_n_lefty == 0 or go_back_n_rightx == 0 or go_back_n_rightx == 0):
    left_fit = np.copy(left_line.best_fit)
    right_fit = np.copy(right_line.best_fit)
else:
    left_fit = np.polyfit(left_line.ally[len(left_line.ally)-go_back_n_lefty:],
                      left_line.allx[len(left_line.allx)-go_back_n_leftx:], 2)
    right_fit = np.polyfit(right_line.ally[len(right_line.ally)-go_back_n_righty:],
                      right_line.allx[len(right_line.allx)-go_back_n_rightx:], 2)
```

I also keep track of the last successful fit coefficients. In an effort to control noisy line predictions, I have some handling code that sees if the difference between the second coefficient in left_fit or right_fit and the tracked best fit is more than 15%. If it is, I keep only a small proportion of the noisy coefficient's vote, but having it mostly draw from the previous best fit. I did find this to help a little bit in situations with less lighting.

With the edge case handling, I have a pipeline that provides results for the entirety of the provided test videos. Performance is acceptable for "project_video.mp4" and "challenge_video.mp4". "harder_challenge_video.mp4" with its curvy path and highly varied lighting conditions (especially bright as well) has some runs of success but the proposed lane lines are wonky for most of the duration.

The projected line for the running example is here:

![alt text][road_lines]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius calculation comes in the 11th code cell of my notebook. It takes in the x and y points for the left and right lines, extracts the polynomial coefficients (with some extra pixels-to-meters conversions) and calculates the radius of the curve with the equation provided in the project materials:

```python
left_curverad = ((1 + (2*left_fit_cr[0]*np.max(ploty)*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*np.max(ploty)*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

The position of the car is more straightforward. Once I have the polynomial equation of the lane lines I create a filled polygon with that information. I can then do a reverse perspective transformation to get it from the region of interest to the perspective of the car's camera. This is shown overlaid on top of the input image:

![alt text][road_overlaid]

Since the polygon lies within a rectangle the same dimensions of the input image, I can correlate position of the polygon image to the input image. I scan along the bottom row of the polygon image and find the leftmost and rightmost elements where the polygon starts. This is the width of the lane. The midpoint between those two elements is the middle of the lane. Since we can expect the camera to be in the middle of the car, the offset of the car in the lane is the difference between the middle of the lane and the middle of the frame. With the same conversion for the radius calculation, we can put this in terms of meters again.

I noticed that my calculations produce results that are within an order of magnitude. Based on the materials provided for the project, including the [U.S. government specifications for highway curvature](http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm#BGBHGEGC) that we can expect measurements ranging from about 1km to several km. My solution seems in line with those results with particularly high radius measurements (>10km) when the road section is particularly straight, which follows expectations.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

With my polygon defined and my lane measurements sorted out, I just have to display it all. I do this in the 12th code cell of my project in lines 99-134. At the end, I get the following:

![alt text][road_final]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a link to my video results [project_video.mp4](./output_images/project_video.mp4), [challenge_video.mp4](./output_images/challenge_video.mp4), [harder_challenge_video.mp4](./output_images/harder_challenge_video.mp4)

My video pipeline is slightly different to interface with MoviePy's VideoFileClip module (fl_image specifically). The image pipeline includes more debug output that is absent from the video pipeline. The video pipeline can be found in code cell 14.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Issues I faced in this project were mainly centered around varying lighting conditions. My pipeline is somewhat improved versus my first implementation but videos with lighting conditions as varied as are found in the "harder_challenge_video" clip break my solution. Even this clip doesn't deal with nighttime driving which I think my pipeline would struggle with as well. To make it more robust I would have to find a combination of HLS and edge detection (possibly other approaches as well) that hold up to changes in lighting. In the well lit section of the "harder_challenge_video" clip I noticed the grassy area off the edge of the road also got picked up so I would want to make that component more selective as well. All three test clips also provided sunny-day conditions. I think my pipeline could struggle with rain or especially snow on the ground.

Using the last n frames to deal with troublesome frames did help some but the more frames you draw from, the worse you handle rapidly curving roads. The methods I devised for working around troublesome frames had some success but more development and research would be necessary just to get acceptable performance on the "harder_challenge_video" clip.
