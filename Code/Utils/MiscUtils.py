import cv2
import os
import yaml
import numpy as np

def loadImages(folder_name):
    files = os.listdir(folder_name)
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))
    print("Loading images from ", folder_name)
    images = []
    for f in files:
        # print(f)
        image_path = folder_name + "/" + f
        image = cv2.imread(image_path)
        if image is not None:
            images.append(image)			
        else:
            print("Error in loading image ", image)

    return images

def getCamera(config_file_name):
    config_file = open(config_file_name)
    config_data = yaml.load(config_file, Loader=yaml.FullLoader)

    K = config_data.get('K')
    D = config_data.get('D')

    K = np.array(K.split(), np.float32).reshape(3,3)
    D = np.array(D.split(), np.float32)

    return K, D