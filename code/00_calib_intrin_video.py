###############################################################################
### Calibrate intrinsics from video images                                  ###
### Input : Video images of chessboard pattern placed at different pose     ###
### Output: Save camera intrinsics                                          ###
### Usage : python 00_calib_intrin_video.py                                 ###
###############################################################################

import cv2
import yaml
import numpy as np

from utils_calibration import Calibration


# Note: Use a smaller chessboard so easier to move around
calib = Calibration(chessboard_size=(6,5), chessboard_sq_size=0.015)

# Folder that contains video of chessboard pattern
filepath = '../data/hp_video/intrin/'

#############################################
### First extract image frames from video ###
#############################################
cap = cv2.VideoCapture(filepath+'intrin.mp4')

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

# Use the list of images to get camera intrinsics matrix and distortion coeff
# Provide filepath to save image for checking calibration result
mat, dist = calib.get_intrin(img_list, filepath)

print('Intrinsics matrix:', mat)
print('Distortion coefficients:', dist)

# Store camera intrinsics
data = dict(color_camera_matrix=mat.tolist(),
            color_distortion_coeffs=dist.tolist())
filename = 'camera_param_'+ str(img.shape[1])+'x'+str(img.shape[0])+'.yaml'
with open(filepath+filename, 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)
    print('Saved to ' + filepath + filename)

# Clean up
cv2.destroyAllWindows()

print('End')

# Shold print the below outputs:
# python 00_calib_intrin_video.py
# Index 0
# ...
# Index 14
# Found ChessboardCorners 0
# Found ChessboardCorners 1
# Found ChessboardCorners 2
# Found ChessboardCorners 3
# Found ChessboardCorners 4
# Found ChessboardCorners 5
# Found ChessboardCorners 6
# Found ChessboardCorners 7
# Found ChessboardCorners 8
# Found ChessboardCorners 9
# Found ChessboardCorners 10
# Found ChessboardCorners 11
# Found ChessboardCorners 12
# Found ChessboardCorners 13
# Found ChessboardCorners 14
# Calibrating ...
# Calibrating done
# 0 [Calibration] Reprojection error 0.04246397043961722
# 1 [Calibration] Reprojection error 0.053674688302519306
# 2 [Calibration] Reprojection error 0.04714564721509655
# 3 [Calibration] Reprojection error 0.05048104628030333
# 4 [Calibration] Reprojection error 0.21179991601686748
# 5 [Calibration] Reprojection error 0.055511114876501774
# 6 [Calibration] Reprojection error 0.07863458819210213
# 7 [Calibration] Reprojection error 0.37735991341333697
# 8 [Calibration] Reprojection error 0.05044748331185496
# 9 [Calibration] Reprojection error 0.07353449320096685
# 10 [Calibration] Reprojection error 0.2063277664102814
# 11 [Calibration] Reprojection error 0.1858802266845775
# 12 [Calibration] Reprojection error 0.11639992805569678
# 13 [Calibration] Reprojection error 0.3927236689506335
# 14 [Calibration] Reprojection error 0.266752121456045
# Intrinsics matrix: [[1.47312495e+03 0.00000000e+00 9.18451265e+02]
#  [0.00000000e+00 1.47819493e+03 5.30696256e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# Distortion coefficients: [[ 4.79476184e-01 -1.89171897e+00 -2.98537172e-03 -2.61198074e-04
#    2.36838551e+00]]
# Saved to ../data/hp_video/intrin/camera_param_1920x1080.yaml
# End
