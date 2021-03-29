import cv2
import os

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

def getCamParam(yaml_file):
    pass