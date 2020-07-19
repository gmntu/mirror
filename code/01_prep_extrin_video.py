###############################################################################
### Prepare data for extrinsics calibration of camera                       ###
### Input : Video images of chessboard patterns placed at different pose    ###
### Output: Save data to multicam folder                                    ###
### Usage : python 01_prep_extrin_video.py                                  ###
### Note  : This program requires manual input to guide the process         ###
###############################################################################

import os
import cv2
import yaml
import argparse
import numpy as np

from utils_calibration import Calibration


# Folder that contains video of chessboard pattern
filepath = '../data/hp_video/extrin/'

#############################################
### First extract image frames from video ###
#############################################
cap = cv2.VideoCapture(filepath+'extrin.mp4')

frame = 0
index = 0
while(cap.isOpened()):
    ret, img = cap.read()
    if ret:
        frame += 1
        # Extract and save image every 30 frames
        if frame%30==0:
            cv2.imwrite(filepath+str(index).zfill(2)+'.jpg', img) 
            print('Index', index)
            index += 1
    else:
        break

cap.release()


########################################################
### Next perform calibration on the extracted images ###
########################################################
# To store list of color images
img_list = [] 
for i in range(index):
    img = cv2.imread(filepath+str(i).zfill(2)+'.jpg')
    # Note: Half resolution (3840 x 2160)->(1920 x 1080) to speed up processing
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    img_list.append(img)

# Extract intrinsics parameters
filepath = '../data/hp_video/intrin/'
filename = 'camera_param_'+ str(img.shape[1])+'x'+str(img.shape[0])+'.yaml'
if os.path.exists(filepath+filename):
    param   = yaml.load(open(filepath+filename), Loader=yaml.FullLoader)
    mat = np.asarray(param['color_camera_matrix'])
    dist = np.asarray(param['color_distortion_coeffs'])
else:
	print(filepath+filename, ' does not exist ... refer to 00_calib_intrin_video.py')

# Note: Use a smaller chessboard so easier to move around
calib = Calibration(chessboard_size=(6,5), chessboard_sq_size=0.015)

# Prepare data in multicam folder with manual input
# Note: Hardcode number of views to 3
filepath = '../data/hp_video/extrin/'
calib.get_extrin_manual(img_list, mat, dist, filepath, num_views=3)

# Clean up
cv2.destroyAllWindows()

print('End')
