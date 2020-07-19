###############################################################################
### Calibrate intrinsics of camera                                          ###
### Input : Color images of chessboard pattern placed at different pose     ###
### Output: Save camera intrinsics                                          ###
### Usage : python 00_calib_intrin.py                                       ###
###############################################################################

import cv2
import yaml
import numpy as np

from utils_calibration import Calibration


# Note: Use a smaller chessboard so easier to move around
calib = Calibration(chessboard_size=(6,5), chessboard_sq_size=0.015)

# Folder that contains 30 color images of chessboard pattern
filepath = '../data/hp/intrin/'

# To store list of color images
img_list = [] 
# Note: Hardcode 30 images with name XX.jpg
for i in range(30):
    img = cv2.imread(filepath+str(i).zfill(2)+'.jpg')
    # Note: Half resolution (4032 x 3024)->(2016 x 1512) to speed up processing
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
# python 00_calib_intrin.py
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
# Found ChessboardCorners 15
# Found ChessboardCorners 16
# Found ChessboardCorners 17
# Found ChessboardCorners 18
# Found ChessboardCorners 19
# Found ChessboardCorners 20
# Found ChessboardCorners 21
# Found ChessboardCorners 22
# Found ChessboardCorners 23
# Found ChessboardCorners 24
# Found ChessboardCorners 25
# Found ChessboardCorners 26
# Found ChessboardCorners 27
# Found ChessboardCorners 28
# Found ChessboardCorners 29
# Calibrating ...
# Calibrating done
# 0 [Calibration] Reprojection error 0.039653353394177214
# 1 [Calibration] Reprojection error 0.0508938760633538
# 2 [Calibration] Reprojection error 0.046342772072731624
# 3 [Calibration] Reprojection error 0.056651895783794974
# 4 [Calibration] Reprojection error 0.03919690952934121
# 5 [Calibration] Reprojection error 0.04949912564834867
# 6 [Calibration] Reprojection error 0.03461006660191394
# 7 [Calibration] Reprojection error 0.03194143743035185
# 8 [Calibration] Reprojection error 0.034546431861955745
# 9 [Calibration] Reprojection error 0.03544314343785427
# 10 [Calibration] Reprojection error 0.04307466830417061
# 11 [Calibration] Reprojection error 0.03795348746145118
# 12 [Calibration] Reprojection error 0.04320160483134098
# 13 [Calibration] Reprojection error 0.038357063939472504
# 14 [Calibration] Reprojection error 0.05844503557976741
# 15 [Calibration] Reprojection error 0.03783754173063739
# 16 [Calibration] Reprojection error 0.0381847702960569
# 17 [Calibration] Reprojection error 0.035152216329644034
# 18 [Calibration] Reprojection error 0.0362574409812512
# 19 [Calibration] Reprojection error 0.03987069662750182
# 20 [Calibration] Reprojection error 0.04239152713988124
# 21 [Calibration] Reprojection error 0.04627035691230573
# 22 [Calibration] Reprojection error 0.05834675947294158
# 23 [Calibration] Reprojection error 0.056279943069989576
# 24 [Calibration] Reprojection error 0.04149479579041709
# 25 [Calibration] Reprojection error 0.03546410130854153
# 26 [Calibration] Reprojection error 0.04297851344408602
# 27 [Calibration] Reprojection error 0.037194478713213054
# 28 [Calibration] Reprojection error 0.038999311068954115
# 29 [Calibration] Reprojection error 0.05362213882285397
# Intrinsics matrix: [[1.55529014e+03 0.00000000e+00 9.57952642e+02]
#  [0.00000000e+00 1.55782154e+03 7.44354437e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
# Distortion coefficients: [[ 3.38766466e-01 -1.08768437e+00 -2.47025647e-03  3.21318846e-04
#    9.10315912e-01]]
# Saved to ../data/hp/intrin/camera_param_2016x1512.yaml
# End
