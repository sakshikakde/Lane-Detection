
import numpy as np
import cv2
import math

def filterLines(linesP, old_r_line, old_l_line):
    line_angles = []
    lines = []

    right_lines = []
    right_line_angles = []

    left_lines = []
    left_line_angles = []

    if linesP is not None:    
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            slope = (l[3] - l[1]) / (l[2] - l[0])
            angle = np.arctan(slope)*180 / np.pi

            if np.abs(angle) > 20 and np.abs(angle) < 70: #filter out horizontal and vertical lines
                if angle < 0:
                    if angle > -35 and angle < -25:
                        left_lines.append(l)
                        left_line_angles.append(angle)
                else:
                    if angle > 45 and angle < 65:
                        right_lines.append(l)
                        right_line_angles.append(angle)

        chosen_l_line = old_l_line

    if len(left_lines) is not 0:
        left_lines = np.array(left_lines).reshape(-1,4)
        chosen_l_index = np.argmax(left_lines[:,1])
        # print("left angle = ", left_line_angles[chosen_l_index])
        chosen_l_line = left_lines[chosen_l_index, :]
    else:
        chosen_l_line = old_l_line

    if len(right_lines) is not 0:
        right_lines = np.array(right_lines).reshape(-1,4)
        chosen_r_index = np.argmax(right_lines[:,1])
        # print("right angle = ", right_line_angles[chosen_r_index])
        chosen_r_line = right_lines[chosen_r_index, :]

    else:
        chosen_r_line = old_r_line
    
    return chosen_r_line, chosen_l_line

def extendLines(line, y_min, y_max):

    extended_line = np.zeros_like(line)
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    m = (y2 - y1) / (x2 - x1 + 1e-10)
    c = y2 - (m * x2)
    
    x_min = (y_max - c) / m 
    x_max = (y_min - c) / m
    
    extended_line[0] = x_min
    extended_line[1] = y_max
    extended_line[2] = x_max
    extended_line[3] = y_min
    return extended_line
    
