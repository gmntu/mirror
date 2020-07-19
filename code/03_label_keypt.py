###############################################################################
### Label 21 2D hand keypoints automatically with given roi                 ###
### Input : Color image                                                     ###
### Output: Save 21 2D hand keypoints                                       ###
### Usage : python 03_label_keypt.py                                        ###
### Usage : python 03_label_keypt.py --roi                                  ###
###############################################################################

import os
import cv2
import time
import yaml
import argparse
import numpy as np


######################
### Get user input ###
######################
parser = argparse.ArgumentParser()
# Possible file: 'open_00', 
#                'fist_00', 'hook_00', 'thumb_ip_00', 'thumb_mcp_00',
#                'fist_03', 'hook_03', 'thumb_ip_03', 'thumb_mcp_03'
# parser.add_argument('--file', default='open_00')
# parser.add_argument('--view', default=1) # Set the number of views (1 to 5)
parser.add_argument('--roi', action='store_true') # Set --roi flag to allow user to set ROI
args = parser.parse_args()

# path = '../data/hp/'+str(args.view)+'views/'
# file = args.file
# view = int(args.view)
roi  = args.roi


##################################
### Load 2D keypoint detection ###
##################################
from utils_hand_keypoint import MobileNetHand
mnh = MobileNetHand(top_left_x=0, top_left_y=0) # Load mobilenet network


########################################
### Loop through all files in one go ###
########################################
files = ['open_00', 'fist_00','fist_03','hook_00','hook_03',
         'thumb_mcp_00','thumb_mcp_03','thumb_ip_00','thumb_ip_03']
for view in range(1,6):
    for file in files:
        path = '../data/hp/'+str(view)+'views/'
        print(path+file)


        #####################
        ### Extract keypt ###
        #####################
        rois = []
        if os.path.exists(path+file+'.yaml'):
            param  = yaml.load(open(path+file+'.yaml'), Loader=yaml.FullLoader)
            keypts_auto = []
            for i in range(view):
                keypts_auto.append(np.asarray(param['keypt_auto'+str(i)]).reshape(-1,3))
                rois.append(np.asarray(param['roi'+str(i)]))
            keypts_auto = np.asarray(keypts_auto)
            keypts_auto = keypts_auto.reshape(-1, 3) # (n*21,3)        

        img = cv2.imread(path+file+'.jpg')
        img = cv2.resize(img, None, fx=0.25, fy=0.25) # Image for 2D keypoint detection
        img_roi = img.copy()
        color_roi = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255)]
        # Extract the keypt
        keypts = []
        for i in range(view):
            if roi:
                # User select ROI
                r = cv2.selectROI(img) # Return top left x, y, width, height
            else:
                # Use previously recorded roi
                r = rois[i]
            
            cv2.rectangle(img_roi, (int(r[0]),int(r[1])), 
                (int(r[0]+r[2]),int(r[1]+r[3])), color_roi[i], 2)

            start = time.time()
            r = np.asarray(r) # Convert tuple r into list so can modify the value
            param['roi'+str(i)] = r.tolist()
            # Scale down the image to allow the hand to fit into 224 by 224 bbox
            scale = 224.0/((r[2] + r[3])/2.0)
            # scale = 368.0/((r[2] + r[3])/2.0) # For openpose
            img_ = cv2.resize(img, None, fx=scale, fy=scale)
            flip = False
            if i==1: flip = True
            elif i==view-1 and view>2: flip = True
            if flip:
                img_ = cv2.flip(img_, flipCode=1) # Need to flip the mirror image
                r[0] = img.shape[1] - r[0] - r[2]
            # Update the top left x and y coordinate
            mnh.top_left_x = int(r[0]*scale)
            mnh.top_left_y = int(r[1]*scale)
            keypt_bb, keypt, img_bb = mnh.get_hand_keypoints(img_, crop=False)
            keypt[:,:2] = keypt[:,:2]/scale # (21, 3)
            if flip:
                keypt[:,0] = img.shape[1] - keypt[:,0] # Mirror flip the keypt
            keypts.append(keypt)
            end = time.time()   
            print('Time elasped', end-start, 'seconds')            

            mnh.draw_hand_keypoints(img_roi, keypt)
            mnh.draw_hand_keypoints(img_bb, keypt_bb)

            cv2.imshow('img_bb'+str(i), img_bb)
            
            keypt[:,:2] = keypt[:,:2]*2.0 # Need to multiply back two times as the image used for optimization is half of 4032 = 2016
            param['keypt_auto'+str(i)] = keypt.flatten().tolist()


        with open(path+file+'.yaml', 'w') as f:
            yaml.dump(param, f)          


        cv2.imshow('img_roi', img_roi)
        cv2.waitKey(0)
