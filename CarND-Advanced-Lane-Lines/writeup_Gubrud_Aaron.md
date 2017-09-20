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
[road_lines]: ./output_images/examples/example_lines_test6.jpg "Test 6, Lines"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
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

Since I already have the distortion characterized with the variables `mtx` and `dst` (which hold the camera matrix and the distortion coefficients respectively), I can simply provide the image I want to undistort along with these variables to the `cv2.undistort()` function. From this operation, I get the below undistorted road image. (It looks the same unless you a/b them on top of each other but I promise they're different :) )

![alt text][road_undist]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I decided to do the perspective transform after undistorting the image because it makes sense to me to restrict the region of interest as early as possible. The code for my perspective transform includes a function called `corners_unwarp`, which appears in code cell 9.  The `corners_unwarp` function takes as inputs an image (`img`), as well as the polygon corners which define my region of interest (`corners`).  I chose the hardcode the corners as follows:

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

I got a lot of mileage from just converting the input image to the HSL color space with OpenCV and just doing thresholding on those values. I investigated some common characteristics of the test images we were provided and defined the following rules:

If the saturation channel is >= 90 and the lightness channel is >= 100, then we can consider this of interest (a yellow line).
If the saturation channel is < 32 and the lightness channel is >= 85% of the max lightness in the region, then we can consider this of interest as well (a white line).

Since the region of interest has already been restricted to the road in front of the car, the saturation and lightness thresholding proves pretty effective. In fact, with this thresholding alone, I get satisfying results from all of the provided test images. Where I do get into trouble, though, is areas with inconsistent lighting (e.g. particularly light or dark images). Continuing with the example image, here is the region of interest with HLS thresholding:

![alt text][road_threshold]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To sort out the polynomial fitting, I followed the guidance of the instructional videos for this project. This happens in the function called `compute_lane_line_polynomials` which takes in the binary thresholded representation of the image. With the binary thresholded image, I start with two

![alt text][road_lines]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
