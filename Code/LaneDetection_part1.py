import gi
gi.require_version('Gtk', '2.0')
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import math
import os
from Utils.ImageUtils import *
from Utils.MiscUtils import *
from Utils.GeometryUtils import *
import random
import argparse


def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='./', help='Base path of project2')
    Parser.add_argument('--ImageFilePath', default='./Data/Project2_Dataset2/data_1/data', help='absolute path')
    Parser.add_argument('--CamConfigFile', default='./Data/Project2_Dataset2/data_1/camera_params.yaml', help='.yaml config file name')
    Parser.add_argument('--SaveFileName', default='Results/Problem2/lane_result_1.avi', help='Saved video file name')

    Args = Parser.parse_args()
    BasePath = Args.BasePath
    ImageFilePath = Args.ImageFilePath
    CamConfigFile = Args.CamConfigFile
    SaveFileName = BasePath + Args.SaveFileName    

    images = loadImages(ImageFilePath)
    K, D = getCamera(CamConfigFile)

    frame_height, frame_width = images[0].shape[:2]
    result = cv2.VideoWriter(SaveFileName, cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height)) 

    old_right_line = None
    old_left_line = None
    while images:
        image = images.pop(0)
        
        h, w, _ = image.shape
        #undistort the image
        image_undistorted = cv2.undistort(image, K, D)
        image_overlay = image_undistorted.copy()

        #remove noise
        image_blur = cv2.GaussianBlur(image_undistorted, (5,5), 0)

        #get bin
        image_bin = extractWhite(image_blur, threshold = 250)

        #roi
        image_roi, cropped_image = getROI(image_bin, 0.45) 

        #extract points
        filtered_image_roi = sobelXYThresh(image_roi)

        #get lines
        linesP = cv2.HoughLinesP(image_roi, 1, np.pi / 180, 50, None, 50, 10)

        #select left and right lines
        right_line, left_line = filterLines(linesP, old_right_line, old_left_line)
        old_right_line = right_line
        old_left_line = left_line

        # extend line
        extended_right_line = extendLines(right_line,  0.55 * h,  h)
        extended_left_line = extendLines(left_line,  0.55 * h,  h)

        # fill polygon
        corners = []
        
        corners.append([extended_left_line[0], extended_left_line[1]])
        corners.append([extended_right_line[0], extended_right_line[1]])
        corners.append([extended_right_line[2], extended_right_line[3]])
        corners.append([extended_left_line[2], extended_left_line[3]])
        corners = np.array(corners)

        cv2.fillPoly(image_overlay, pts =[corners], color=(0,0,255))

        #overlay
        cv2.addWeighted(image_overlay, 0.4, image_undistorted, 0.6, 0, image_overlay)

        #draw lines    
        cv2.line(image_overlay, (extended_right_line[0], extended_right_line[1]), (extended_right_line[2], extended_right_line[3]), (0, 0, 255), 3, cv2.LINE_AA)
        cv2.line(image_overlay, (extended_left_line[0], extended_left_line[1]), (extended_left_line[2], extended_left_line[3]), (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow('frame', image_overlay)
        result.write(image_overlay) 
        cv2.waitKey(10)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




