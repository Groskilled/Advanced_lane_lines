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

[image1]: ./test_images/test4.jpg "Original Image"
[image2]: ./output_images/undistort_test.png "Undistorted"
[image3]: ./output_images/binary_output.png "Binary Example"
[image4]: ./output_images/lane_fit.png "Fit Visual"
[image5]: ./output_images/final_output.png "Output"
[video1]: project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 163 through 174 of the main.py.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

![alt text][image2]

If you want to see the effect on the calibration image, there is the outuput of it in the output_images folder (calibration3_undist)

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 103 through 133 in main.py).  Here's an example of my output for this step.  

![alt text][image3]

The code for my perspective transform includes a function called perspective_transform(), which appears in lines 95 through 101 (output_images/examples/example.py). The `perspective_transform()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[500, 482],[780, 482],[1250, 720],[40, 720]])
dst = np.float32([[0, 0], [1280, 0],[1250, 720],[40, 720]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 500, 482      | 0, 0        | 
| 40, 720      | 40, 720      |
| 1250, 720     | 1250, 720      |
| 780, 462      | 1280, 0        |

I verified that my perspective transform was working as expected by taking a image of a straight road and made sure the lines were still straight after the transformation 

Then I had to find the lines in the lower half of the image and which point belong to the left or the right line. When that was done, I used the np.polyfit function to fit fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image4]

I calculated the radius of curvature in lines 83 through 90 in my code in `main.py` and the position of the vehicule with respect to center at line 182.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I finally plotted the lane area back to the image. Here is an example of my result on a test image:

![alt text][image5]

---

### Pipeline (video)

You can find the output in the result.mp4 video

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had problems with the image process because I did not understand that the calibration using the chessboard image was of use for the road image. When I figured this out, it was pretty easy to get a good result
except for the parts where the road changes color (black to grey) and another part. To deal with this I stored the previous points and equation of lines. If the equation found for the current image was too far from the previous, I discard it.
This project could be way better with more time spent on it, calculating a confidence score for each line, keeping the best and set the other line from that and other things can be added.
