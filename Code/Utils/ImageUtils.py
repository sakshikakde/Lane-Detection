import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import math


grad_x_min = 211
grad_x_max =255 
grad_y_min =211
grad_y_max = 255

mag_min = 200
mag_max = 255

angle_min  = 0.7
angle_max = 1.3
kernal_size = 3

def extractWhite(image, threshold = 250):
    
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.equalizeHist(image_gray)
    _, image_thresh = cv2.threshold(image_gray, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)

    return image_thresh

def sobelXYThresh(image):
    img = image.copy()
    sobel_x = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = kernal_size))
    sobel_y = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = kernal_size))
 
    sobel_x_rescaled = np.uint8(255 * sobel_x / np.max(sobel_x))
    sobel_y_rescaled = np.uint8(255 * sobel_y / np.max(sobel_y))

    # x and y thresh
    bin_x = np.zeros_like(sobel_x)
    bin_y = np.zeros_like(sobel_y)

    bin_x[(sobel_x_rescaled >= grad_x_min) & (sobel_x_rescaled <= grad_x_max)] = 1
    bin_y[(sobel_y_rescaled >= grad_y_min) & (sobel_y_rescaled <= grad_y_max)] = 1    

    # magnitude thresh
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    grad_mag = (grad_mag * 255/np.max(grad_mag)).astype(np.uint8) 

    bin_mag = np.zeros_like(grad_mag)
    bin_mag[(grad_mag >= mag_min) & (grad_mag <= mag_max)] = 1

    # angle thresh
    grad_angle = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))

    bin_angle = np.zeros_like(grad_angle)
    bin_angle[(grad_angle >= angle_min) & (grad_angle <= angle_max)] = 1

    # combine all
    combined = np.zeros_like(bin_x)
    combined[((bin_x == 1) & (bin_y == 1)) | ((bin_mag == 1) & (bin_angle == 1))] = 255

    return combined


def getROI(image, discard_percent):    
    img = image.copy()
    if len(img.shape) > 2:
        h, w, _ = img.shape
    else:
        h, w = img.shape
    roi_corners = np.array([[0, discard_percent * h], [w, discard_percent * h], [w, 0], [0, 0]], np.int32).reshape((-1,1,2))
    cv2.drawContours(img, [roi_corners], -1, 0, cv2.FILLED)  
    cropped_image = image[int(discard_percent * h):h, 0:w]
    return img, cropped_image