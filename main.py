import numpy as np
import os
from moviepy.editor import VideoFileClip
import sys
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy.linalg import inv
import pickle


class Line():
    def __init__(self):
        self.detected = False
        self.X = None
        self.Y = None
        self.base = 0
        self.recent_xfitted = None
        self.radius_of_curvature = None
        self.line_base_pos = None
        self.prev_poly = None
        self.xint = None

    def find_lane(self, binary_warped):
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        midpoint = np.int(histogram.shape[0]/2)
        if self.detected == False:
            x_base = np.argmax(histogram[self.base:midpoint + self.base]) + self.base
        else:
            x_base = self.X
            self.detected = True
        nwindows = 9
        window_height = np.int(binary_warped.shape[0]/nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        x_current = x_base
        margin = 100
        minpix = 50
        lane_inds = []
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]
            lane_inds.append(good_inds)
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))
        lane_inds = np.concatenate(lane_inds)
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds] 
        y = np.array(y).astype(np.float32)
        x = np.array(x).astype(np.float32)
        fit = np.polyfit(y, x, 2)
        if self.prev_poly is not None:
            p1 = self.prev_poly[0]
            p2 = self.prev_poly[1]
            p3 = self.prev_poly[2]
            if np.sqrt(pow(p1 - fit[0],2)) < 0.5 and np.sqrt(pow(p2 - fit[1],2)) < 0.5 :
                self.prev_poly = fit
            else:
                return self.recent_xfitted
        else :
            self.prev_poly = fit
        fitx = fit[0]*y**2 + fit[1]*y + fit[2]
        x_int = fit[0]*720**2 + fit[1]*720 + fit[2]
        self.xint = x_int
        x = np.append(x, x_int)
        y = np.append(y, 720)
        x = np.append(x,fit[0]*0**2 + fit[1]*0 + fit[2])
        y = np.append(y, 0)
        lsort = np.argsort(y)
        self.Y = y[lsort]
        self.X = x[lsort]
        fit = np.polyfit(self.Y, self.X, 2)
        fitx = fit[0]*self.Y**2 + fit[1]*self.Y + fit[2]
        tmp = np.vstack([fitx, self.Y])
        self.recent_xfitted = tmp

        ploty = np.linspace(0, 719, num=720)
        quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
        leftx = np.array([200 + (self.Y**2)*quadratic_coeff + np.random.randint(-50, high=51)for y in ploty])
        y_eval = np.max(ploty)
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        fit_cr = np.polyfit(self.Y*ym_per_pix, self.X*xm_per_pix, 2)
        self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        
        return self.recent_xfitted

def perspective_transform(img):
    src = np.float32([[500, 482],[780, 482],[1250, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0],[1250, 720],[40, 720]])
    img_size = (img.shape[1], img.shape[0])
    offset = 0
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M
    
def get_binary_img(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    
    r_channel = img[:,:,0]
    l_channel = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)[:,:,0]
    s_channel = hls[:,:,2]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >=10) & (scaled_sobel <= 255)] = 1

    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 150) & (s_channel <= 255)] = 1

    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= 200) & (r_channel <= 255)] = 1

    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= 200) & (l_channel <= 255)] = 1
    
    binary_warped = np.zeros_like(s_binary)
    binary_warped[(sx_binary == 1) & (s_binary == 1) | (r_binary == 1) & (l_binary == 1)] = 1

    return binary_warped

def undistort(objpoints, imgpoints, img):
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    return cv2.undistort(img, mtx, dist, None, mtx)

def img_process(img):
    image = undistort(objpoints, imgpoints, img)
    warped, mat = perspective_transform(image)
    binary_warped = get_binary_img(warped)
    out_img = np.copy(binary_warped)

    pts_left = np.array([np.flipud(np.transpose(left.find_lane(out_img)))])
    pts_right = np.array([np.transpose(np.vstack(right.find_lane(out_img)))])
    pts = np.hstack((pts_left, pts_right))
    tmp = (640 - (left.xint + right.xint)/2) * 3.7/700
    
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[500, 482],[780, 482],[1250, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0],[1250, 720],[40, 720]])
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0,0,255), thickness = 40)
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (binary_warped.shape[1], binary_warped.shape[0]))
    cv2.putText(newwarp, 'Vehicle is {:.2f}m away of center'.format(tmp), (100,80),fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 2)
    cv2.putText(newwarp, 'Radius of Curvature {}(m)'.format(int((left.radius_of_curvature+right.radius_of_curvature)/2)), (120,140),fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 2)
    return cv2.addWeighted(image, 1, newwarp, 0.5, 0)

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
objpoints = []
imgpoints = []
images = glob.glob('camera_cal/*.jpg')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

left = Line()
right = Line()
right.base = 640
video_output = 'result.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(img_process)
white_clip.write_videofile(video_output, audio=False)
