import gi
gi.require_version('Gtk', '2.0')
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
import math
import argparse

def adjustGamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


def AdaptivehistogramEqualizer(image):
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image_lab)

    clahe = cv2.createCLAHE(clipLimit = 5)
    l_eq = clahe.apply(l)
    equalized_image = cv2.merge((np.uint8(l_eq), a, b))

    # equalized_image = np.zeros_like(image)
    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_LAB2BGR)
    return equalized_image



def histogramEqualizer(image):
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(image_lab)

    channel = l
    channel_flat = channel.flatten()
    histogram, _ = np.histogram(channel_flat, 256, [0, 256], True)
    cdf = histogram.cumsum()

    new_bin = (255 * cdf).astype(int)
    new_channel = new_bin[channel_flat]
    equalized_l = new_channel.reshape(l.shape)
    equalized_image = cv2.merge((np.uint8(equalized_l), a, b))

    # equalized_image = np.zeros_like(image)
    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_LAB2BGR)
    return equalized_image


def main():  

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='./', help='Base path of project1, Default:./')
    Parser.add_argument('--VideoFilePath', default='./Data/Night Drive - 2689.mp4', help='MP4 file name, Default:Tag2.mp4')
    Parser.add_argument('--SaveFileName', default='Results/ImageCorrection.avi', help='Folder to save graphs, Default:Video1')

    Args = Parser.parse_args()
    BasePath = Args.BasePath
    VideoFilePath = Args.VideoFilePath
    SaveFileName = BasePath + Args.SaveFileName

    cap = cv2.VideoCapture(VideoFilePath)
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4))
    result = cv2.VideoWriter(SaveFileName, cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height)) 

    while(True):
        ret, frame = cap.read()      
        if not ret:
            print("Stream ended..")
            break

        image = adjustGamma(frame, 2)
        image_hist = histogramEqualizer(image)
        image_adaptive_hist = AdaptivehistogramEqualizer(image)

        cv2.imshow('frame',image_adaptive_hist)
        result.write(np.uint8(image_adaptive_hist))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




