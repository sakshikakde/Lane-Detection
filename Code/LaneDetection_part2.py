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
from Utils.MovingAverage import *
import random
import argparse
  

def getWarpedLane(image):
    h, w = image.shape[:2]
    image_size_x = 300
    image_size_y = 500
    set1 = np.array([[60, 715], [600, 465], [740, 465], [w, h]])
    set2 = np.array([[0, image_size_y], [0, 0], [image_size_x, 0], [image_size_x, image_size_y]])
    H = cv2.getPerspectiveTransform(np.float32(set1), np.float32(set2))
    warped_image = cv2.warpPerspective(image, H, (image_size_x, image_size_y))
    return warped_image, H

def addBasePoint(points, max_x):
    pts = np.array(points)
    y = pts[:,1]
    y_avg = y[0]#np.mean(y)
    y_avg = int(y_avg)
    points.append((max_x, y_avg))
    return points
    

def getLeftRightPoints(image):
    h_w, w_w = image.shape
    strip_width = 20
    left_indexes = []
    right_indexes = []
    thresh = 30

    iy_r_old = 0
    iy_l_old = 0

    thresh_iy = 5.0

    left_found = False
    right_found = False

    for h in range(int(h_w/strip_width), 1, -1):

        left_strip = image[(h-1) * strip_width: h * strip_width, 0: int(w_w/3)]
        right_strip = image[(h-1) * strip_width: h * strip_width, int(2 * w_w/3): w_w]

        if np.sum(left_strip) / 255 > thresh:
            ix_l, iy_l = np.where(left_strip == 255)
            ix_l = np.uint8(np.mean(ix_l))
            iy_l = np.uint8(np.mean(iy_l))
            ix_l = ix_l + (h-1) * strip_width
            if left_found:
                del_y = np.abs(float(iy_l)-float(iy_l_old))
                if del_y < thresh_iy:
                    index = (ix_l, iy_l)
                    left_indexes.append(index)
                    iy_l_old = iy_l
            else:
                index = (ix_l, iy_l)
                left_indexes.append(index)
                left_found = True
                iy_l_old = iy_l


        if np.sum(right_strip) / 255 > thresh:
            ix_r, iy_r = np.where(right_strip == 255)
            ix_r = np.uint8(np.mean(ix_r))
            iy_r = np.uint8(np.mean(iy_r))
            ix_r = ix_r + (h-1) * strip_width
            iy_r = iy_r + int(2 * w_w/3)
            index = (ix_r, iy_r)
            right_indexes.append(index)

            if right_found:
                del_y = np.abs(float(iy_r)-float(iy_r_old))
                if del_y < thresh_iy:
                    index = (ix_r, iy_r)
                    right_indexes.append(index)
                    iy_r_old = iy_r
            else:
                index = (ix_r, iy_r)
                right_indexes.append(index)
                right_found = True
                iy_r_old = iy_r


    return left_indexes, right_indexes


def getCurve(points, order = 2):
    indexes = np.array(points)
    x = indexes[:,0]
    y = indexes[:,1]
    fit = np.polyfit(x, y, order)
    return fit


def findCurvature(coef, x):
    a, b, c = coef
    dy = 2*a*x + b
    d2y = 2*a

    R = (1 + dy**2) **(3/2)
    R = R / d2y

    R = np.min(R)
    return R


def finalDisplay(image_undistorted, image_bin, image_warped, display_image, image_overlay, left_curvature, right_curvature, old_turn):

    image_undistorted = cv2.resize(image_undistorted, (300, 168))
    image_bin = cv2.resize(image_bin, (300, 168))
    image_bin = cv2.merge((image_bin, image_bin, image_bin))
    image_warped = cv2.merge((image_warped, image_warped, image_warped))

    image_undistorted = cv2.putText(image_undistorted, '(1)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    image_bin = cv2.putText(image_bin, '(2)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    image_warped = cv2.putText(image_warped, '(3)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    display_image = cv2.putText(display_image, '(4)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    side_pannel = np.zeros((720, 600, 3), np.uint8)
    side_pannel[0:168, 0:300, :] = image_undistorted
    side_pannel[0:168, 300:600, :] = image_bin
    side_pannel[168:668, 0:300, :] = image_warped
    side_pannel[168:668, 300:600, :] = display_image

    full_pannel = np.zeros((900, 1880, 3), np.uint8)
    full_pannel[0:720, 0:1280, :] = image_overlay
    full_pannel[0:720, 1280:1880, :] = side_pannel

    info_pannel = np.zeros((180, 1880, 3), np.uint8)
    info_pannel[:,:,2] = 200
    info_pannel[:,:,0] = 255
    info_pannel[:,:,1] = 200
    info_pannel = cv2.putText(info_pannel, '(1) : Undistorted image, (2) : Detected white and yellow lane markings, (3) : Warped image, (4) : Detected points and curve fitting', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    curve_string = "Left Curvature : " + str(round(left_curvature, 2)) + ", Right Curvature : " + str(round(right_curvature,2))
    info_pannel = cv2.putText(info_pannel, curve_string, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    turn_curvature = "Curvature not found!"
    turn = old_turn
    if round(left_curvature/ np.abs(left_curvature)) == round(right_curvature/ np.abs(right_curvature)):
        av = 0.3 * left_curvature + 0.7 * right_curvature
        turn_curvature = "Average Curvature : " + str(round(av, 2))
        if av < 4000 and av > 0:
            turn = "Turn Right"
        if np.abs(av) > 4000:
            turn = "Go Straight"
        if av < 0 and av > -4000:
            turn = "Turn Left"
        

    info_pannel = cv2.putText(info_pannel, turn_curvature, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    
    full_pannel[720:900, 0:1880, :] = info_pannel

    full_pannel = cv2.putText(full_pannel, turn, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return full_pannel, turn

def main():

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='./', help='Base path of project2')
    Parser.add_argument('--VideoFilePath', default='./Data/Project2_Dataset2/data_2/challenge_video.mp4', help='relative image files path')
    Parser.add_argument('--CamConfigFile', default='./Data/Project2_Dataset2/data_2/cam_params.yaml', help='.yaml config file name')
    Parser.add_argument('--SaveFileName', default='Results/lane_detection/lane_result_2.avi', help='Saved video file name')

    Args = Parser.parse_args()
    BasePath = Args.BasePath
    VideoFilePath = Args.VideoFilePath
    CamConfigFile = Args.CamConfigFile
    SaveFileName = BasePath + Args.SaveFileName    

    cap = cv2.VideoCapture(VideoFilePath)
    K, D = getCamera(CamConfigFile)

    frame_width = int(cap.get(3)) + 600
    frame_height = int(cap.get(4)) + 180
    result = cv2.VideoWriter(SaveFileName, cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height)) 

    old_left_fit = []
    old_right_fit = []

    left_lane_detected = False
    right_lane_detected = False
    first_time = False

    moving_average_left = MovingAverage(window_size = 10)
    moving_average_right = MovingAverage(window_size = 10)

    moving_average_R_r = MovingAverage(window_size = 10)
    moving_average_R_l = MovingAverage(window_size = 10)

    R_l_old = 0
    turn = "Not found!"
    while(True):
        ret, frame = cap.read()
        
        if not ret:
            print("Stream ended..")
            break
        cv2.imshow('frame',frame)

        image = frame
        h, w, _ = image.shape
        # undistort image
        image_undistorted = cv2.undistort(image, K, D)
        image_overlay = image_undistorted.copy()

        #roi
        image_roi, cropped_image = getROI(image_undistorted, 0.62) 
        
        # get white and yellow area
        image_bin = extractWhiteYellow(image_roi)

        #filter noise
        kernel = np.ones((5,5),np.uint8)
        image_bin = cv2.morphologyEx(image_bin, cv2.MORPH_CLOSE, kernel)

        # warp image
        image_warped, H = getWarpedLane(image_bin) 

        #get left and right point
        left_indexes, right_indexes = getLeftRightPoints(image_warped)
        display_image = drawDetections(image_warped, left_indexes, right_indexes)

        #left curve
        if(len(left_indexes) > 5):
            left_indexes = addBasePoint(left_indexes, image_warped.shape[0])
            left_fit = getCurve(left_indexes, order = 2)
            old_left_fit = left_fit
            #moving average
            moving_average_left.addMarkers(left_fit)
            left_fit = moving_average_left.getAverage()
            left_lane_detected = True
        else:
            left_fit = old_left_fit

        #right curve
        if(len(right_indexes) > 5):
            #add base point for both
            right_indexes = addBasePoint(right_indexes, image_warped.shape[0])
            right_fit = getCurve(right_indexes, order = 2)
            #moving average
            moving_average_right.addMarkers(right_fit)
            right_fit = moving_average_right.getAverage()
            old_right_fit = right_fit
            right_lane_detected = True
        else:
            right_fit = old_right_fit
        
        #draw curves
        if left_lane_detected:
            display_image, draw_points_left = drawCurve(display_image, left_fit, (0, 0, 255))
        if right_lane_detected:
            display_image, draw_points_right = drawCurve(display_image, right_fit, (0, 255, 255))

        #reproject curve on image
        draw_points_left = draw_points_left.reshape(-1,1,2).astype(np.float32)
        draw_points_left_transformed = cv2.perspectiveTransform(draw_points_left, np.linalg.inv(H))
        draw_points_left_transformed= (draw_points_left_transformed.reshape(-1,2)).astype(np.int32)
        image_overlay = cv2.polylines(image_overlay, [draw_points_left_transformed], False, (0,0,255), 4)  

        draw_points_right = draw_points_right.reshape(-1,1,2).astype(np.float32)
        draw_points_right_transformed = cv2.perspectiveTransform(draw_points_right, np.linalg.inv(H))
        draw_points_right_transformed= (draw_points_right_transformed.reshape(-1,2)).astype(np.int32)
        image_overlay = cv2.polylines(image_overlay, [draw_points_right_transformed], False, (0, 255, 255), 4)  

        corners = np.vstack((draw_points_left_transformed, draw_points_right_transformed[::-1]))
        cv2.fillPoly(image_overlay, pts =[corners], color=(0,0,255))
        #overlay
        cv2.addWeighted(image_overlay, 0.4, image_undistorted, 0.6, 0, image_overlay)
        
        #curvature
        h_w = image_warped.shape[0]
        x = np.linspace(0, h_w - 1, h_w)
        R_l = findCurvature(left_fit, x)
        moving_average_R_l.addMarkers(R_l)
        R_l = moving_average_R_l.getAverage()


        R_r = findCurvature(right_fit, x)
        moving_average_R_r.addMarkers(R_r)
        R_r = moving_average_R_r.getAverage()

        #full display
        full_display, turn = finalDisplay(image_undistorted, image_bin, image_warped, display_image, image_overlay, R_l, R_r, turn)


        cv2.imshow('frame', full_display)
        result.write(full_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


