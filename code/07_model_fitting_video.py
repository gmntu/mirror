###############################################################################
### Fit skeleton model to detected 2D hand keypoints                        ###
### Input : Video image                                                     ###
### Output: Save images and video of 2D keypoint and 3D hand pose           ###
### Usage : python 07_model_fitting_video.py --file ball                    ###
###############################################################################

import os
import cv2
import yaml
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from utils_hand_skeleton import HandSkeletonPyTorch


######################
### Get user input ###
######################
parser = argparse.ArgumentParser()
# Possible file: 'hand_only', 'hand_rom', 'ball', 'cup', cube_small', 'cube_big'
parser.add_argument('--file', default='ball')
args = parser.parse_args()

file = args.file
view = 3
n    = 21*view # Total number of joints (each view has 21 joints)


##################################
### Load 2D keypoint detection ###
##################################
from utils_hand_keypoint import MobileNetHand
mnh = MobileNetHand(top_left_x=0, top_left_y=0) # Load mobilenet network


###################################
### Get ROI for the first frame ###
###################################
path = '../data/hp_video/'
cap = cv2.VideoCapture(path+file+'/'+file+'.mp4')
ret, img = cap.read()
# cv2.imwrite(path+'3views/fist_00.jpg', img)
img = cv2.resize(img, None, fx=0.25, fy=0.25) # Image for 2D keypoint detection
img_opt = img.copy()
img_width = img.shape[1]*2.0 # Half of 3840 = 1920

roi = [] # To store roi
color_roi = [(0,0,255),(0,255,0),(255,0,0)]

# Allow user to select ROI for first frame
# print('Select ROI in a clockwise order starting from actual camera view of the hand')
# for i in range(view):
#     # User select ROI
#     r = cv2.selectROI(img) # Return top left x, y, width, height    
#     r = np.asarray(r) # Convert tuple r into array so can modify the value
#     # Draw bounding box
#     cv2.rectangle(img, (int(r[0]),int(r[1])), (int(r[0]+r[2]),int(r[1]+r[3])), color_roi[i], 2)    
#     roi.append(r)
#     print(r)
# cv2.destroyAllWindows()

# Or use hardcoded ROI 
if file=='hand_only':
    roi.append([377, 181, 382, 349]) # hand_only
    roi.append([ 83,  16, 238, 233])
    roi.append([671,  17, 258, 251])
else:
    roi.append([439, 222, 339, 316]) # hand_rom, ball, cup, cube_small, cube_big
    roi.append([ 63,   2, 240, 222])
    roi.append([698,  36, 253, 261])


#############################
### Load extrin parameter ###
#############################
device = 'cpu'
extrins = []
for i in range(view):
    serial_number = str(i)
    filename = path + 'extrin/model2camera_matrix_' + serial_number + '.yaml'
    if os.path.exists(filename):
        param   = yaml.load(open(filename), Loader=yaml.FullLoader)
        extrin = np.asarray(param['model2camera_matrix'])
        extrin = np.linalg.inv(extrin)
        extrin = torch.from_numpy(extrin).float().to(device) # Convert to tensor
        extrins.append(extrin)      


#############################
### Load intrin parameter ###
#############################
filename = path + 'intrin/camera_param_1920x1080.yaml'
if os.path.exists(filename):
    param   = yaml.load(open(filename), Loader=yaml.FullLoader)
    intrin = np.asarray(param['color_camera_matrix'])
    intrin = torch.from_numpy(intrin).float().to(device) # Convert to tensor
    dist = np.asarray(param['color_distortion_coeffs'])
    dist = torch.from_numpy(dist.flatten()).float().to(device) # Convert to tensor


#######################################
### Load 3D hand model optimization ###
#######################################
num_iter = 300
model = HandSkeletonPyTorch(path=path+'3views/', file='fist_00', bone_calib=False)
model = model.to(device)
model.train() # Set to TRAIN mode
optimizer = optim.Adam(model.parameters(), lr=0.03) # Setup optimizer


#################
### Main loop ###
#################
# Create folder to store images if it does not exist
if not os.path.exists(path+file+'/3views/2d_keypt/'):
    os.makedirs(path+file+'/3views/2d_keypt/')
if not os.path.exists(path+file+'/3views/2d_skeleton/'):
    os.makedirs(path+file+'/3views/2d_skeleton/')

loop_cnt = 0
while True:
    if loop_cnt<1:
        num_iter = 300
    else:
        num_iter = 10
    loop_cnt += 1

    #########################
    ### Get the keypoints ###
    #########################
    keypts = []
    for i in range(view):
        # Scale down the image to allow the hand to fit into 224 by 224 bbox
        scale = 224.0/((roi[i][2] + roi[i][3])/2.0)
        img_ = cv2.resize(img, None, fx=scale, fy=scale)
        if i>=1:
            img_ = cv2.flip(img_, flipCode=1) # Need to flip the mirror image
            roi[i][0] = img.shape[1] - roi[i][0] - roi[i][2]
        # Update the top left x and y coordinate
        if i==0 and file=='cube_big': # A quick hack to prevent the first bounding box from merging with the third bounding box
            roi[i][0] = min(roi[i][0], 624*scale)
        mnh.top_left_x = int(roi[i][0]*scale)
        mnh.top_left_y = int(roi[i][1]*scale)
        # Extract the keypt
        keypt_bb, keypt, img_bb = mnh.get_hand_keypoints(img_, crop=False)
        # Scale back the keypt
        keypt[:,:2] = keypt[:,:2]/scale # (21, 3)
        if i>=1:
            keypt[:,0] = img.shape[1] - keypt[:,0] # Mirror flip the keypt      

        # Update new roi top left x and top left y
        cnt, xx, yy = 0, 0, 0
        # Get mean position of the hand keypoints
        for kp in keypt:
            if kp[0]>0 and kp[1]>0:
                xx += kp[0]
                yy += kp[1]
                cnt += 1
        if cnt > 0:
            roi[i][0] = int(xx/cnt - 224/2/scale)
            roi[i][1] = int(yy/cnt - 224/2/scale)

        # Draw bounding box
        cv2.rectangle(img, (int(roi[i][0]),int(roi[i][1])), (int(roi[i][0]+224/scale),int(roi[i][1]++224/scale)), color_roi[i], 2)
        # Draw keypoints
        mnh.draw_hand_keypoints(img, keypt)
        
        keypt[:,:2] = keypt[:,:2]*2.0 # Need to multiply back two times as the image used for optimization is half of 3840 = 1920
        keypts.append(keypt.copy()) # Must make a copy of keypt else it will be modified in the next loop


    #############################
    ### Optimize 3D hand pose ###
    #############################
    # Convert keypt to tensor
    keypts = np.asarray(keypts)
    keypts = keypts.reshape(-1, 3) # (n*21,3)
    keypts_ = keypts.copy()
    keypts_[21*1:21*2,0] = img_width - keypts_[21*1:21*2,0] # Need to mirror image the keypoints
    keypts_[21*2:21*3,0] = img_width - keypts_[21*2:21*3,0]
    keypts_[:,2] = keypts_[:,2]**2 # Square the confidence
    # Convert numpy to torch.tensor
    keypt0 = torch.from_numpy(keypts_[:21,:]).float().to(device)
    keypt1 = torch.from_numpy(keypts_[21*1:21*2,:]).float().to(device)
    keypt2 = torch.from_numpy(keypts_[21*2:21*3,:]).float().to(device)

    for i in range(num_iter):
        # Zero gradients
        optimizer.zero_grad()

        # Make estimation
        joint3D, _ = model() # Forward pass to get the 3D joint

        joint2D_0 = model.project_point3(joint3D, intrin, dist, extrins[0][:3,:])
        joint2D_1 = model.project_point3(joint3D, intrin, dist, extrins[1][:3,:])
        joint2D_2 = model.project_point3(joint3D, intrin, dist, extrins[2][:3,:])

        # Compute loss
        loss_keypt  = torch.sum(keypt0[:,2].unsqueeze(dim=-1) * (joint2D_0 - keypt0[:,:2]) ** 2)
        loss_keypt += torch.sum(keypt1[:,2].unsqueeze(dim=-1) * (joint2D_1 - keypt1[:,:2]) ** 2)            
        loss_keypt += torch.sum(keypt2[:,2].unsqueeze(dim=-1) * (joint2D_2 - keypt2[:,:2]) ** 2)            
        loss_limit = model.hpose_limit() * 1e5
        loss = loss_keypt + loss_limit

        # Compute gradients
        loss.backward()
        
        # Updates parameters 
        optimizer.step()

    joint2D_0 = joint2D_0.detach().cpu().numpy()
    joint2D_1 = joint2D_1.detach().cpu().numpy()
    joint2D_2 = joint2D_2.detach().cpu().numpy()

    # Need to mirror image back the joint2D
    joint2D_1[:,0] = img_width - joint2D_1[:,0]
    joint2D_2[:,0] = img_width - joint2D_2[:,0]        

    print('Loop', loop_cnt)

    ################################
    ### Visualization of results ###
    ################################
    # Draw the skeleton
    model.draw_point(joint2D_0/2.0, img=img_opt, win_name='color', resize=False)
    model.draw_point(joint2D_1/2.0, img=img_opt, win_name='color', resize=False)
    model.draw_point(joint2D_2/2.0, img=img_opt, win_name='color', resize=False)

    cv2.imwrite(path+file+'/3views/2d_keypt/'+str(loop_cnt).zfill(5)+'.png', img)
    cv2.imwrite(path+file+'/3views/2d_skeleton/'+str(loop_cnt).zfill(5)+'.png', img_opt)

    ret, img = cap.read()
    if not ret:
        break
    img = cv2.resize(img, None, fx=0.25, fy=0.25) # Image for 2D keypoint detection
    img_opt = img.copy()



###################################################################
### Continue to create video from all the filenames with '.png' ###
###################################################################
def is_image(filename):
    return filename.endswith('.png')
def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])


######################################################
### Define the codec and create VideoWriter object ###
######################################################
fps = 30
width, height = 960*2, 540
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
video = cv2.VideoWriter(path+file+'/video.mp4', fourcc, fps, (width, height))


#################
### Main Loop ###
#################
folder = path+file+'/3views/2d_keypt/'
filenames = [image_basename(f) for f in os.listdir(folder) if is_image(f)]
filenames.sort()
print('Total number of files', len(filenames))

counter_log = 0

with tqdm(total=len(filenames)) as t:
    for f in filenames:
        ##########################
        ### Read in the images ###
        ##########################
        img_kp2 = cv2.imread(path+file+'/3views/2d_keypt/'+f+'.png')
        img_sk2 = cv2.imread(path+file+'/3views/2d_skeleton/'+f+'.png')

        ################
        ### Add Text ###
        ################
        cv2.putText(img_kp2, '2D Keypoint detection with bounding box', 
            (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(img_sk2, '3D Hand skeleton projected onto individual view ', 
            (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)

        combine4 = np.hstack((img_kp2, img_sk2))

        video.write(combine4)
        counter_log += 1
        t.update()

print("Video completed")    
video.release()      
