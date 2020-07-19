###############################################################################
### Fit skeleton model to detected 2D hand keypoints                        ###
### Input : Color image                                                     ###
### Output: Save 3D hand pose                                               ###
###                                                                         ###
### Usage : Personalization of hand model needs to                          ###
###         set --bone_calib   (to allow optimization of bone parameter)    ###
###         use --mode manual  (detected keypoints are more accurate)       ###    
###         set --file open_00 (open hand pose has all joints visible)      ###
### python 04_model_fitting.py --bone_calib --mode manual --file open_00 --view 1
###                                                                         ###
### Usage : Measure hand rom                                                ###
###         set file to required type of hand pose                          ###
### python 04_model_fitting.py --mode auto --file fist_00 --view 1          ###
###                                                                         ###
### Note: Manual labeling of keypoint with mouse click is also allowed      ###
### Note: Press spacebar to start model fitting process                     ###
### Note: Press 's' to save optimized pose parameters                       ###
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

from utils_hand_skeleton import HandSkeletonPyTorch, Open3DHandSkeleton


######################
### Get user input ###
######################
parser = argparse.ArgumentParser()
# Possible file: 'open_00', 
#                'fist_00', 'hook_00', 'thumb_ip_00', 'thumb_mcp_00',
#                'fist_03', 'hook_03', 'thumb_ip_03', 'thumb_mcp_03'
parser.add_argument('--file', default='open_00')
parser.add_argument('--view', default=1) # Set the number of views (1 to 5)
parser.add_argument('--mode', default='auto') # Set the mode auto or manual keypoint labeling
parser.add_argument('--bone_calib', action='store_true') # Set to --bone_calib flag to enable bone calibration
args = parser.parse_args()

path = '../data/hp/'+str(args.view)+'views/'
file = args.file
view = int(args.view)
mode = args.mode
bone_calib = args.bone_calib


########################
### Global variables ###
########################
n = 21*view # Total number of joints (each view has 21 joints)

# Define the parameters for GUI window
gui_x = 90 # Increase this number to allow more space to display pt_name on gui window
gui_width = 200
gui_height = 500 # Increase this number to allow more space for more pt_name

index = 0 # To keep track of the keypoint index
mouse = [0,0] # To keep track of mouse (x,y) pose in image window
keypts = np.zeros((n,3)) # Contain an array of [x,y,confidence] data of the keypoints


############################################
### GUI for manual labeling of keypoints ###
############################################
def resetKeypointsIndex():
    global index
    global keypts
    
    index = 0
    keypts = np.zeros((n,3))


def mouseClick(event, x, y, flags, param):
    # Need to add global else will get local variable 'ptList' referenced before assignment
    global index
    global mouse
    global keypts 

    if event==cv2.EVENT_MOUSEMOVE:
        mouse = [x,y]    
    
    if event==cv2.EVENT_LBUTTONDOWN: # Left button click add new point with high confidence
        if index < n:
            keypts[index] = np.array([x*2.0, y*2.0, 1.0]) # Note: mouse x y are in the reference of image four times smaller, but keypts are in the reference of image two times smaller
            index += 1

    if event==cv2.EVENT_RBUTTONDOWN: # Right button click add new point wuth low confidence
        if index < n:
            keypts[index] = np.array([x*2.0, y*2.0, 0.0])
            index += 1
    
    elif event==cv2.EVENT_MBUTTONDOWN: # Middle button click remove point
        for i in range(n):
            # Reset those points that are < 5 pixels away from mouse cursor
            if abs(keypts[i,0]/2.0-x)<5 and abs(keypts[i,1]/2.0-y)<5: 
                keypts[i] = np.zeros(3)
                index = i


def mouseGUI(event, x, y, flags, param):
    global index 
    
    if event==cv2.EVENT_LBUTTONDOWN or event==cv2.EVENT_RBUTTONDOWN: # Left or middle button click select new index
        # Minus 0.5 to round down to nearest int
        # 45 is the starting y coordinate to draw rectangle on current index being selected
        # 20 is the width of the rectangle
        index = round((y-45)/20 - 0.5) 


# Define two windows and mousecallbacks
cv2.namedWindow('img')
cv2.namedWindow('gui')
cv2.setMouseCallback('img', mouseClick)
cv2.setMouseCallback('gui', mouseGUI)


pt_name = ['W', # Follow convention from BigHand\n",
           'T0','I0','M0','L0','P0',
           'T1','T2','T3',
           'I1','I2','I3',
           'M1','M2','M3',
           'L1','L2','L3',
           'P1','P2','P3']

color = [[0,0,0], 
         [255, 0, 0], [0, 255, 0], [0, 255, 0], [0, 0, 255], [0, 0, 255], # (All the 5 MCP)(knuckles)
         [255, 60, 0], [255, 120, 0], [255, 180, 0],                      # Thumb
         [60, 255, 0], [120, 255, 0], [180, 255, 0],                      # Index
         [0, 255, 60], [0, 255, 120], [0, 255, 180],                      # Middle
         [0, 60, 255], [0, 120, 255], [0, 180, 255],                      # Ring
         [60, 0, 255], [120, 0, 255], [180, 0, 255]]                      # Little    


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
    param  = yaml.load(open(filename), Loader=yaml.FullLoader)
    intrin = np.asarray(param['color_camera_matrix'])
    intrin = torch.from_numpy(intrin).float().to(device) # Convert to tensor
    dist   = np.asarray(param['color_distortion_coeffs'])
    dist   = torch.from_numpy(dist.flatten()).float().to(device) # Convert to tensor


###########################################
### Load 3D hand model for optimization ###
###########################################
num_iter = 300
model = HandSkeletonPyTorch(path=path, file=file, bone_calib=bone_calib)
model = model.to(device)
model.train() # Set to TRAIN mode
optimizer = optim.Adam(model.parameters(), lr=0.03) # Setup optimizer


############################
### Load 3D hand display ###
############################
disp = Open3DHandSkeleton(model, draw_hand_axes=False, enable_trackbar=False)
joint3D = np.zeros((21,3))


#############################
### Load saved parameters ###
#############################
if os.path.exists(path+'hand_param.yaml'):
    # Store general hand parameters such as bone length and rotation
    hand_param = yaml.load(open(path+'hand_param.yaml'), Loader=yaml.FullLoader)

if os.path.exists(path+file+'.yaml'):
    # Store specific parameters such as hand pose and keypoint that is used for the specific file
    param = yaml.load(open(path+file+'.yaml'), Loader=yaml.FullLoader)

    # Extract keypt
    keypts_auto = []
    keypts_manual = []
    for i in range(view):
        keypts_auto.append(np.asarray(param['keypt_auto'+str(i)]).reshape(-1,3))
        keypts_manual.append(np.asarray(param['keypt_manual'+str(i)]).reshape(-1,3))
    keypts_auto = np.asarray(keypts_auto)
    keypts_auto = keypts_auto.reshape(-1, 3) # (n*21,3)
    keypts_manual = np.asarray(keypts_manual)
    keypts_manual = keypts_manual.reshape(-1, 3) # (n*21,3)

    if mode=='manual':
        keypts = keypts_manual
    elif mode=='auto':
        keypts = keypts_auto


img = cv2.imread(path+file+'.jpg')
img = cv2.resize(img, None, fx=0.5, fy=0.5) # Image for optimization
img_width = img.shape[1] # Half of 4032 = 2016


#################
### Main loop ###
#################
while True:
    img = cv2.imread(path+file+'.jpg')
    img = cv2.resize(img, None, fx=0.25, fy=0.25) # Manual label need to make image four times smaller to fit into screen
    img_opt = img.copy()
    gui = np.zeros((gui_height,gui_width,3), np.uint8)
    gui.fill(100) # Set to dark grey background

    # Add text for image name
    cv2.putText(gui, file+'.png', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    # Add text for mouse (x,y) position
    cv2.putText(gui, str(mouse), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    # Add rectangle on current index being selected
    if index < n:
        cv2.rectangle(gui, (0,(index%21)*20+45), (gui_x,(index%21)*20+65), color[index%21])
    for i in range(21):
        # Add the name of keypoint
        cv2.putText(gui, pt_name[i%21], (0, i*20+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[i%21], 1, cv2.LINE_AA)
        # Add text on the (x,y) pose of keypoint
        cv2.putText(gui, str(keypts[i,:2]), (gui_x, i*20+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[i%21], 1, cv2.LINE_AA)
    for i in range(n):
        # Plot the keypoint as cross
        x = int(keypts[i,0]/2.0)
        y = int(keypts[i,1]/2.0)
        if keypts[i,2]>0.5:
            # Plot the keypoint as dot if confidence is high
            cv2.circle(img, (x,y), 4, color[i%21], -1)
        else:
            # Plot the keypoint as cross if confidence is low
            cv2.line(img, (x-4,y), (x+4,y), color[i%21], 2)
            cv2.line(img, (x,y-4), (x,y+4), color[i%21], 2)
    
    # Display image and GUI
    cv2.imshow('img', img)
    cv2.imshow('gui', gui)

    disp.update_display(joint3D)
    
    # Get user input
    key = cv2.waitKey(10)
    if key==27: # Press escape to end the program
        print('Quitting...')
        break 
    if key==ord('c'): # Press c to clear all list of keypoints and reset index
        resetKeypointsIndex()

    if key==ord('s'): # Press s to save
        # Save parameters to .yaml file
        if bone_calib and mode=='manual': # Only save the bone parameters when using manual mode
            opt_bolen = model.state_dict()['bolen_'].detach().cpu().numpy()
            opt_borot = model.state_dict()['borot_'].detach().cpu().numpy()
            hand_param['bone_length']   = opt_bolen.tolist()
            hand_param['bone_rotation'] = opt_borot.tolist()

        opt_hpose = model.state_dict()['hpose_'].detach().cpu().numpy()
        if mode=='auto':
            param['hand_pose_auto'] = opt_hpose.tolist()
            param['keypt_auto0'] = keypts_auto[:21,:].flatten().tolist()
            if view>1:
                param['keypt_auto1'] = keypts_auto[21*1:21*2,:].flatten().tolist()
            if view>2:
                param['keypt_auto2'] = keypts_auto[21*2:21*3,:].flatten().tolist()
            if view>3:
                param['keypt_auto3'] = keypts_auto[21*3:21*4,:].flatten().tolist()
            if view>4:
                param['keypt_auto4'] = keypts_auto[21*4:21*5,:].flatten().tolist()
        elif mode=='manual':
            param['hand_pose_manual'] = opt_hpose.tolist()
            param['keypt_manual0'] = keypts[:21,:].flatten().tolist()
            if view>1:
                param['keypt_manual1'] = keypts[21*1:21*2,:].flatten().tolist()
            if view>2:
                param['keypt_manual2'] = keypts[21*2:21*3,:].flatten().tolist()
            if view>3:
                param['keypt_manual3'] = keypts[21*3:21*4,:].flatten().tolist()
            if view>4:
                param['keypt_manual4'] = keypts[21*4:21*5,:].flatten().tolist()


        with open(path+file+'.yaml', 'w') as outfile:
            yaml.dump(param, outfile, default_flow_style=False)
            print('Saved', path+file+'.yaml' +'.yaml')

        if bone_calib:
            with open(path+'hand_param.yaml', 'w') as outfile:
                yaml.dump(hand_param, outfile, default_flow_style=False)
                print('Saved', path+'hand_param.yaml')            

        time.sleep(0.1)

    if key==32: # Press spacebar to optimize 3D hand pose
        print('Optimizing ...')
        start = time.time()
        # Convert keypt to tensor
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

        #############################
        ### Optimize 3D hand pose ###
        #############################
        for i in range(num_iter):
            # Zero gradients
            optimizer.zero_grad()

            # Make estimation
            joint3D, _ = model() # Forward pass to get the 3D joint

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
            # loss_keypt  = torch.sum((joint2D_0 - keypt0[:,:2]) ** 2)
            # if view>1:
            #     loss_keypt += torch.sum((joint2D_1 - keypt1[:,:2]) ** 2)            
            # if view>2:
            #     loss_keypt += torch.sum((joint2D_2 - keypt2[:,:2]) ** 2)            
            # if view>3:
            #     loss_keypt += torch.sum((joint2D_3 - keypt3[:,:2]) ** 2)            
            # if view>4:
            #     loss_keypt += torch.sum((joint2D_4 - keypt4[:,:2]) ** 2)            
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
        # Get the joint angles
        hpose = model.state_dict()['hpose_'].detach().cpu().numpy()
        model.print_hpose(hpose)


################
### Clean up ###
################
cv2.destroyAllWindows()
disp.vis.destroy_window()
print('End')