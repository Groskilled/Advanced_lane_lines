import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def perspective_transform(img):
    #src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
    src = np.float32([[200,650], [1100,650], [750,450], [550,450]])
    #plt.imshow(img)
    img_size = (img.shape[1], img.shape[0])
    #for p in src:
    #    plt.scatter(p[0],p[1])
    offset = 0
    dst = np.float32([[offset, offset], [img_size[0]-offset, offset],[img_size[0]-offset, img_size[1]-offset],[offset, img_size[1]-offset]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size)
    plt.imshow(warped)
    plt.show()
    

    #plt.show()

def get_binary_img(img, thresh=(100,255)):
    #print(img.shape)
    bgr = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    red = bgr[:,:,-1]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    # Plotting thresholded images
    plt.imshow(combined_binary)
    plt.show()


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
#images = glob.glob('camera_cal/*.jpg')
images = glob.glob('test_images/*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images[:5]):
    img = mpimg.imread(fname)
    perspective_transform(img)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## Find the chessboard corners
    #ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    ## If found, add object points, image points
    #if ret == True:
    #    objpoints.append(objp)
    #    imgpoints.append(corners)
    #    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    #    dst = cv2.undistort(img, mtx, dist, None, mtx)

    #    # Draw and display the corners
    #    #cv2.drawChessboardCorners(img, (9,6), corners, ret)
    #    #cv2.imshow('img', img)
    #    cv2.imshow('img', dst)
    #    cv2.waitKey(500)

cv2.destroyAllWindows()
