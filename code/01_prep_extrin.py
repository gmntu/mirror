###############################################################################
### Prepare data for extrinsics calibration of camera                       ###
### Input : Color images of chessboard patterns placed at different pose    ###
### Output: Save data to multicam folder                                    ###
### Usage : python 01_prep_extrin.py --view 2                               ###
### Usage : python 01_prep_extrin.py --view 3                               ###
### Usage : python 01_prep_extrin.py --view 4                               ###
### Usage : python 01_prep_extrin.py --view 5                               ###
### Note  : This program requires manual input to guide the process         ###
###############################################################################

import os
import cv2
import yaml
import argparse
import numpy as np

from utils_calibration import Calibration


# Get user input
parser = argparse.ArgumentParser()
parser.add_argument('--view', default=2) # Set the number of views (2 to 5)
args = parser.parse_args()

# Folder that contains 15 color images of chessboard patterns with mirror
filepath = '../data/hp/'+str(args.view)+'views/extrin/'

# To store list of color images
img_list = [] 
# Note: Hardcode 15 images with name XX.jpg
for i in range(15):
    img = cv2.imread(filepath+str(i).zfill(2)+'.jpg')
    # Note: Half resolution (4032 x 3024)->(2016 x 1512) to speed up processing
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    img_list.append(img)

# Extract intrinsics parameters
filepath = '../data/hp/intrin/'
filename = 'camera_param_'+ str(img.shape[1])+'x'+str(img.shape[0])+'.yaml'
if os.path.exists(filepath+filename):
    param   = yaml.load(open(filepath+filename), Loader=yaml.FullLoader)
    mat = np.asarray(param['color_camera_matrix'])
    dist = np.asarray(param['color_distortion_coeffs'])
else:
	print(filepath+filename, ' does not exist ... refer to 00_calib_intrin.py')

# Note: Use a smaller chessboard so easier to move around
calib = Calibration(chessboard_size=(6,5), chessboard_sq_size=0.015)

# Prepare data in multicam folder with manual input
filepath = '../data/hp/'+str(args.view)+'views/extrin/'
calib.get_extrin_manual(img_list, mat, dist, filepath, args.view)

# Clean up
cv2.destroyAllWindows()

print('End')
