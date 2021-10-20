# Problem 1

## How to run the code
Change the directory where the .py files are present    
/home/sakshi/courses/ENPM673/sakshi_p2/Code

                python3 ImageCorrection.py --BasePath /home/sakshi/courses/ENPM673/sakshi_p1/ --VideoFilePath /home/sakshi/courses/ENPM673/sakshi_p2/Data/Night\ Drive\ -\ 2689.mp4 --SaveFileName Results/Problem1/ImageCorrection.avi
Note: Check the spaces in the video file name

## Parameters
1) BasePath - BasePath - project folder 
2) VideoFilePath - absolute path of the video file
3) SaveFileName - Path to the folder along with name where results are saved. Note: This path is relative to BasePath

# Problem 2: Lane Detection 
## Image Prepossessing
The following operations were performed before lane de-
tection:
1) Image undistortion using cv2.undistort.
2) Image blurring using a Gaussian kernel of size 5 ×5.
3) Thresholding the gray image to get a binary image.
4) Obtained ROI, which is almost 55% of the image.
![Preprocessing output](https://github.com/sakshikakde/Lane-Detection/blob/main/images/i_bin.png)

## How to run the code - Approach 1
Change the directory where the .py files are present    
/home/sakshi/courses/ENPM673/sakshi_p2/Code

                python3 LaneDetection_part1.py --BasePath /home/sakshi/courses/ENPM673/sakshi_p2/ --ImageFilePath /home/sakshi/courses/ENPM673/sakshi_p2/Data/Project2_Dataset2/data_1/data --CamConfigFile /home/sakshi/courses/ENPM673/sakshi_p2/Data/Project2_Dataset2/data_1/camera_params.yaml --SaveFileName Results/Problem2/lane_result_1.avi

## Parameters

1) BasePath - project folder 
2) ImageFilePath - absolute path of the image files
3) CamConfigFile - path where the .yml file is
4) SaveFileName - file name for the saved video. Note: this path is relative to the BasePath

## How to run the code - Approach 2
Change the directory where the .py files are present    
/home/sakshi/courses/ENPM673/sakshi_p2/Code

               python3 LaneDetection_part2.py --BasePath /home/sakshi/courses/ENPM673/sakshi_p2/ --VideoFilePath /home/sakshi/courses/ENPM673/sakshi_p2/Data/Project2_Dataset2/data_2/challenge_video.mp4 --CamConfigFile /home/sakshi/courses/ENPM673/sakshi_p2/Data/Project2_Dataset2/data_2/cam_params.yaml --SaveFileName Results/Problem2/lane_result_2.avi


## Parameters

1) BasePath - project folder 
2) VideoFilePath - absolute path of the video files
3) CamConfigFile - path where the .yml file is
4) SaveFileName - file name for the saved video. Note: this path is relative to the BasePath
