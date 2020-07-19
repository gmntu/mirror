###############################################################################
### Simplified version of 04_model_fitting.py                               ###
### To be called by 05_generate_result.py                                   ###
###############################################################################

import os
import cv2
import time
import yaml
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils_hand_skeleton import HandSkeletonPyTorch


######################
### Get user input ###
######################
parser = argparse.ArgumentParser()
parser.add_argument('--file', default='open_00') # Filename
parser.add_argument('--view', default=1) # Set the number of views (1 to 5)
parser.add_argument('--mode', default='auto') # Set the mode auto or manual keypoint labeling
args = parser.parse_args()

path = '../data/hp/' + str(args.view) + 'views/'
file = args.file
view = int(args.view)
mode = args.mode


########################
### Global variables ###
########################
n = 21*view # Total number of joints (each view has 21 joints)


#############################
### Load extrin parameter ###
#############################
device = 'cpu'
extrins = []
for i in range(view):
    filename = path + 'extrin/model2camera_matrix_' + str(i) + '.yaml'
    if os.path.exists(filename):
        param   = yaml.load(open(filename), Loader=yaml.FullLoader)
        extrin = np.asarray(param['model2camera_matrix'])
        extrin = np.linalg.inv(extrin)
        extrin = torch.from_numpy(extrin).float().to(device) # Convert to tensor
        extrins.append(extrin)      


#############################
### Load intrin parameter ###
#############################
filename = '../data/hp/intrin/camera_param_2016x1512.yaml'
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
model = HandSkeletonPyTorch(path=path, file=file, bone_calib=False)
model = model.to(device)
model.train() # Set to TRAIN mode
optimizer = optim.Adam(model.parameters(), lr=0.03) # Setup optimizer


#############################
### Load saved parameters ###
#############################
if os.path.exists(path+file+'.yaml'):
    # Store specific parameters such as hand pose and keypoint that is used for the specific file
    param = yaml.load(open(path+file+'.yaml'), Loader=yaml.FullLoader)

    keypts = []
    for i in range(view):
        if mode=='auto':
            keypts.append(np.asarray(param['keypt_auto'+str(i)]).reshape(-1,3))
        elif mode=='manual':
            keypts.append(np.asarray(param['keypt_manual'+str(i)]).reshape(-1,3))
    keypts = np.asarray(keypts)
    keypts = keypts.reshape(-1, 3) # (n*21,3)


#############################
### Optimize 3D hand pose ###
#############################
print('Optimizing ...')
start = time.time()
# Need to mirror image the keypoints x coordinates
img = cv2.imread(path+file+'.jpg')
img = cv2.resize(img, None, fx=0.5, fy=0.5) # Image for optimization
img_width = img.shape[1] # Half of 4032 = 2016
keypts_ = keypts.copy()
if view>1:
    keypts_[21*1:21*2,0] = img_width - keypts_[21*1:21*2,0] # Need to mirror image the keypoints
if view==3:
    keypts_[21*2:21*3,0] = img_width - keypts_[21*2:21*3,0]
if view==4:
    keypts_[21*3:21*4,0] = img_width - keypts_[21*3:21*4,0]
if view==5:
    keypts_[21*4:21*5,0] = img_width - keypts_[21*4:21*5,0]

keypts_[:,2] = keypts_[:,2]**2 # Square the confidence

# Convert numpy to torch.tensor
keypt0 = torch.from_numpy(keypts_[:21,:]).float().to(device)
if view>1:
    keypt1 = torch.from_numpy(keypts_[21*1:21*2,:]).float().to(device)
if view>2:
    keypt2 = torch.from_numpy(keypts_[21*2:21*3,:]).float().to(device)
if view>3:
    keypt3 = torch.from_numpy(keypts_[21*3:21*4,:]).float().to(device)
if view>4:
    keypt4 = torch.from_numpy(keypts_[21*4:21*5,:]).float().to(device)

for i in range(num_iter):
    # Zero gradients
    optimizer.zero_grad()

    # Make estimation
    joint3D, _ = model() # Forward pass to get the 3D joint

    # Project the 3D joints to individual 2D image plane
    joint2D_0 = model.project_point3(joint3D, intrin, dist, extrins[0][:3,:])
    if view>1:
        joint2D_1 = model.project_point3(joint3D, intrin, dist, extrins[1][:3,:])
    if view>2:
        joint2D_2 = model.project_point3(joint3D, intrin, dist, extrins[2][:3,:])
    if view>3:
        joint2D_3 = model.project_point3(joint3D, intrin, dist, extrins[3][:3,:])
    if view>4:
        joint2D_4 = model.project_point3(joint3D, intrin, dist, extrins[4][:3,:])

    # Compute loss
    loss_keypt  = torch.sum(keypt0[:,2].unsqueeze(dim=-1) * (joint2D_0 - keypt0[:,:2]) ** 2)
    if view>1:
        loss_keypt += torch.sum(keypt1[:,2].unsqueeze(dim=-1) * (joint2D_1 - keypt1[:,:2]) ** 2)            
    if view>2:
        loss_keypt += torch.sum(keypt2[:,2].unsqueeze(dim=-1) * (joint2D_2 - keypt2[:,:2]) ** 2)            
    if view>3:
        loss_keypt += torch.sum(keypt3[:,2].unsqueeze(dim=-1) * (joint2D_3 - keypt3[:,:2]) ** 2)            
    if view>4:
        loss_keypt += torch.sum(keypt4[:,2].unsqueeze(dim=-1) * (joint2D_4 - keypt4[:,:2]) ** 2)            
    loss_limit = model.hpose_limit() * 1e5
    loss = loss_keypt + loss_limit

    # Compute gradients
    loss.backward()
    
    # Updates parameters 
    optimizer.step()

joint2D_0 = joint2D_0.detach().cpu().numpy()
if view>1:
    joint2D_1 = joint2D_1.detach().cpu().numpy()
if view>2:
    joint2D_2 = joint2D_2.detach().cpu().numpy()
if view>3:
    joint2D_3 = joint2D_3.detach().cpu().numpy()
if view>4:
    joint2D_4 = joint2D_4.detach().cpu().numpy()
joint3D = joint3D[:,:3].detach().cpu().numpy()

# Need to mirror image back the joint2D
if view>1:
    joint2D_1[:,0] = img_width - joint2D_1[:,0]
if view==3:
    joint2D_2[:,0] = img_width - joint2D_2[:,0]        
if view==4:
    joint2D_3[:,0] = img_width - joint2D_3[:,0]        
if view==5:
    joint2D_4[:,0] = img_width - joint2D_4[:,0]     

end = time.time()   
print('Time elasped', end-start, 'seconds')

################################
### Visualization of results ###
################################
img = cv2.imread(path+file+'.jpg')
img = cv2.resize(img, None, fx=0.25, fy=0.25) # Manual label need to make image four times smaller to fit into screen
img_opt = img.copy()
# Draw the keypoints
color = [[0,0,0], 
         [255, 0, 0], [0, 255, 0], [0, 255, 0], [0, 0, 255], [0, 0, 255], # (All the 5 MCP)(knuckles)
         [255, 60, 0], [255, 120, 0], [255, 180, 0],                      # Thumb
         [60, 255, 0], [120, 255, 0], [180, 255, 0],                      # Index
         [0, 255, 60], [0, 255, 120], [0, 255, 180],                      # Middle
         [0, 60, 255], [0, 120, 255], [0, 180, 255],                      # Ring
         [60, 0, 255], [120, 0, 255], [180, 0, 255]]                      # Little    
for i in range(n):
    # Plot the keypoint as cross
    x = int(keypts[i,0]/2.0)
    y = int(keypts[i,1]/2.0)
    if keypts[i,2]>0.5:
        # Plot the keypoint as dot if confidence is high
        cv2.circle(img, (x,y), 6, color[i%21], -1)
    else:
        # Plot the keypoint as cross if confidence is low
        cv2.line(img, (x-6,y), (x+6,y), color[i%21], 3)
        cv2.line(img, (x,y-6), (x,y+6), color[i%21], 3)
cv2.imwrite(path+'/png/'+file+'_'+mode+'_keypt.png', img)     
# Draw the skeleton
model.draw_point(joint2D_0/2.0, img=img_opt, win_name='color', resize=False)
if view>1:
    model.draw_point(joint2D_1/2.0, img=img_opt, win_name='color', resize=False)
if view>2:
    model.draw_point(joint2D_2/2.0, img=img_opt, win_name='color', resize=False)
if view>3:
    model.draw_point(joint2D_3/2.0, img=img_opt, win_name='color', resize=False)
if view>4:
    model.draw_point(joint2D_4/2.0, img=img_opt, win_name='color', resize=False)            
cv2.imwrite(path+'/png/'+file+'_'+mode+'_skeleton.png', img_opt)     
# Get the joint angles
hpose = model.state_dict()['hpose_'].detach().cpu().numpy()
model.print_hpose(hpose)

# Save hpose parameters to .yaml file
if mode=='auto':
    param['hand_pose_auto'] = hpose.tolist()
elif mode=='manual':
    param['hand_pose_manual'] = hpose.tolist()
with open(path+file+'.yaml', 'w') as outfile:
    yaml.dump(param, outfile, default_flow_style=False)
    print('Saved', path+file+'.yaml')