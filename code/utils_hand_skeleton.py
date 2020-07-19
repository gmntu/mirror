###############################################################################
### Sample program to articulate a hand skeleton                            ###
### By varying the hand pose, bone length, bone rotation                    ###
### The HandSkeleton class uses forward kinematics                          ###
### To compute the final joint pose in 3D                                   ###
### To project the 3D points into 2D image plane, use project_point()       ###
###                                                                         ###
### It is possible to run this program directly to articulate the model     ###
### Usage: python utils_hand_skeleton.py                                    ###
###############################################################################

import cv2
import yaml
import numpy as np
import open3d as o3d

import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
### Pytorch version of hand skeleton                                        ###
###############################################################################
class HandSkeletonPyTorch(nn.Module):
    def __init__(self, path, file, bone_calib=False):
        super(HandSkeletonPyTorch, self).__init__()

        self.bone_calib = bone_calib
        #################################
        ### Load parameters from file ###
        #################################
        param = yaml.load(open(path+file+'.yaml'), Loader=yaml.FullLoader)
        # DOF for hand pose 0:2-wrist trans, 3:5-wrist rot, 6:10-thumb, 11-27-index,middle,ring,litte add/abd and flex/ext
        self.hpose = np.asarray(param['hand_pose']) 

        if 'fist' in file:
            i = [12,13,14,16,17,18,20,21,22,24,25,26]
            self.hpose[i] = 1.57 # Set the DIP, PIP and MCP joints to 90 deg = 1.57 rad
        if 'hook' in file:
            i = [13,14,17,18,21,22,25,26]
            self.hpose[i] = 1.57 # Set the DIP and PIP joints to 90 deg = 1.57 rad


        ###########################
        ### Get the hand params ###
        ###########################
        hand_param = yaml.load(open(path+'hand_param.yaml'), Loader=yaml.FullLoader)
        if bone_calib:
            # Bone length total 20 values from 5 metacarpals then to 3 phalangeals for each finger
            self.bolen = np.asarray(hand_param['bone_length_init'])
            # Bone rotation about the wrist 0:Thumb CMC rot, 1:Index rot, 2:Ring rot, 3:Little rot        
            self.borot = np.asarray(hand_param['bone_rotation_init'])
        else:
            # Bone length total 20 values from 5 metacarpals then to 3 phalangeals for each finger
            self.bolen = np.asarray(hand_param['bone_length'])
            # Bone rotation about the wrist 0:Thumb CMC rot, 1:Index rot, 2:Ring rot, 3:Little rot        
            self.borot = np.asarray(hand_param['bone_rotation'])
        # Camera extrin and intrin
        self.camera_extrin = np.asarray(hand_param['camera_extrinsic'])
        self.camera_intrin = np.asarray(hand_param['camera_intrinsic'])
        # Get the min and max ROM for hand pose
        self.hpose_min = np.asarray(hand_param['hand_pose_min'])
        self.hpose_max = np.asarray(hand_param['hand_pose_max'])
        self.hpose_min_bone_calib = np.asarray(hand_param['hand_pose_min_bone_calib'])
        self.hpose_max_bone_calib = np.asarray(hand_param['hand_pose_max_bone_calib'])

        # Convert degree to radian
        if not self.bone_calib:
            self.hpose_min_rad = self.hpose_min.copy()
            self.hpose_max_rad = self.hpose_max.copy()
        else:
            self.hpose_min_rad = self.hpose_min_bone_calib.copy()
            self.hpose_max_rad = self.hpose_max_bone_calib.copy()
        self.hpose_min_rad[3:] = np.radians(self.hpose_min_rad[3:])
        self.hpose_max_rad[3:] = np.radians(self.hpose_max_rad[3:])  
        # Convert to torch tensor
        self.register_buffer('hpose_min_', torch.from_numpy(self.hpose_min_rad).float()) # [20, 3]
        self.register_buffer('hpose_max_', torch.from_numpy(self.hpose_max_rad).float()) # [20, 3]
        # For hpose_limit()
        self.register_buffer('zeros_', torch.from_numpy(np.zeros(27)).float()) # [27]

        # Initialize the rest joint locations
        joint = np.zeros((21,3))
        joint = np.hstack((joint, np.ones((21,1)))) # Create homo coordinate (x, y, z, 1)
        joint[1:,1] = -self.bolen # Set the y axis to negative of bone length
        self.register_buffer('joint_', torch.from_numpy(joint).float()) # [21, 4]


        #############################################
        ### Define the constants in tvec and rvec ###
        #############################################
        if not self.bone_calib: # Use the fixed bone rotation if not doing calibration
            rvecs_1_5 = np.zeros((5, 3))
            rvecs_1_5[[0,1,3,4], 0] = self.borot[[0,1,2,3]] # Rotation of metacarpals in the x direction
            self.register_buffer('rvecs_1_5_', torch.from_numpy(rvecs_1_5).float()) # [5, 3]

        rvecs_6_20 = np.zeros((21, 45)) # Create a sparse matrix to convert 21 values of hpose to (15, 3) array of rvec
        rvecs_index = [0,1,2,5,8,9,11,14,17,18,20,23,26,27,29,32,35,36,38,41,44]
        for i in range(21):
            rvecs_6_20[i, rvecs_index[i]] = 1
        self.register_buffer('rvecs_6_20_', torch.from_numpy(rvecs_6_20).float()) # [21, 45]

        if not self.bone_calib: # Use the fixed bone length if not doing calibration
            tvecs = np.zeros((20, 3))
            tvecs_index = [0,5,6,1,8,9,2,11,12,3,14,15,4,17,18]
            for i in range(5,20): 
                tvecs[i, 1] = -self.bolen[tvecs_index[i-5]]
            self.register_buffer('tvecs_', torch.from_numpy(tvecs).float()) # [20, 3]

        ###############################################
        ### Register the parameter for optimization ###
        ###############################################
        self.register_parameter('hpose_', 
            nn.Parameter(torch.from_numpy(self.hpose).float(), requires_grad=True)) # [27] Hand pose 27 DOF
        if self.bone_calib: # Allow the bone length and rotation to vary during calibration of the bone
            self.register_parameter('bolen_', 
                nn.Parameter(torch.from_numpy(self.bolen).float(), requires_grad=True)) # [20] 20 bones
            self.register_parameter('borot_', 
                nn.Parameter(torch.from_numpy(self.borot).float(), requires_grad=True)) # [4] 4 rotations


        # Kinematic tree linking the 21 joints together to form the skeleton
        self.ktree = [0,         # Wrist
                      0,0,0,0,0, # MCP
                      1,6,7,     # Thumb
                      2,9,10,    # Index finger
                      3,12,13,   # Middle finger
                      4,15,16,   # Ring finger
                      5,18,19]   # Little finger    

        self.color = [[0,0,0], 
                      [255, 0, 0], [0, 255, 0], [0, 255, 0], [0, 0, 255], [0, 0, 255], # (All the 5 MCP)(knuckles)
                      [255, 60, 0], [255, 120, 0], [255, 180, 0],                      # Thumb
                      [60, 255, 0], [120, 255, 0], [180, 255, 0],                      # Index
                      [0, 255, 60], [0, 255, 120], [0, 255, 180],                      # Middle
                      [0, 60, 255], [0, 120, 255], [0, 180, 255],                      # Ring
                      [60, 0, 255], [120, 0, 255], [180, 0, 255]]                      # Little                                  

        print('[HandSkeletonPyTorch] Loaded')


    def forward(self):
        # Set the device type ('cpu' or 'cuda')
        device = self.hpose_.device
        device = torch.device(device.type, device.index)

        # [1, 21] matmul [21, 45] -> [1, 45] -> [15, 3]
        rvecs_6_20 = torch.matmul(self.hpose_[6:].view(1,21), self.rvecs_6_20_).view(-1, 3)

        # Combine 3:6 values in hpose_ (global rotation) with rvecs_
        if self.bone_calib:
            rvecs_1_5 = torch.zeros((5, 3), device=device)
            rvecs_1_5[[0,1,3,4], 0] = self.borot_[[0,1,2,3]] # Rotation of metacarpals in the x direction
            rvecs = torch.cat([self.hpose_[3:6].view(1,3), rvecs_1_5, rvecs_6_20], dim=0) # [21, 3]
        else:
            rvecs = torch.cat([self.hpose_[3:6].view(1,3), self.rvecs_1_5_, rvecs_6_20], dim=0) # [21, 3]

        # Convert 3 by 1 axis angle to 3 by 3 rotation matrix
        rmats = self.batch_rodrigues(rvecs, device=device) # [21, 3, 3]

        # Combine first three values in hpose_ (global translation) with tvecs_
        if self.bone_calib:
            tvecs = torch.zeros((20, 3), device=device)
            tvecs_index = [0,5,6,1,8,9,2,11,12,3,14,15,4,17,18]
            for i in range(5,20): 
                tvecs[i, 1] = -self.bolen_[tvecs_index[i-5]]
            tvecs = torch.cat([self.hpose_[0:3].view(1,3), tvecs], dim=0) # [21, 3]
        else:
            tvecs = torch.cat([self.hpose_[0:3].view(1,3), self.tvecs_], dim=0) # [21, 3]

        # Create the transformation matrix
        transforms_mat = self.transform_mat(rmats, tvecs.unsqueeze(dim=-1)) # [21, 4, 4]

        transform_chain = [transforms_mat[0]]
        # Compute the global transformation matrices by multiplying local homo mat with the parent homo mat
        for i in range(1,len(self.ktree)): # Note: Start from 1 as the first transformation is global transformation and no parent
            # T[i] = torch.matmul(T[self.ktree[i]], T[i])
            curr_res = torch.matmul(transform_chain[self.ktree[i]], transforms_mat[i])
            transform_chain.append(curr_res)

        T = torch.stack(transform_chain)

        # if self.bone_calib:
        #     joint = torch.zeros((21,3), device=device)
        #     ones  = torch.ones((21,1), device=device)
        #     joint = torch.cat([joint, ones], dim=1) # Create homo coordinate (x, y, z, 1)
        #     joint[1:,1] = -self.bolen_ # Set the y axis to negative of bone length
        #     J = torch.matmul(T, joint.unsqueeze(dim=-1)).view(-1, 4)
        # self.joint_[1:,1] = -self.bolen_ # Set the y axis to negative of bone length
        
        J = torch.matmul(T, self.joint_.unsqueeze(dim=-1)).view(-1, 4)
        
        return J, T


    def batch_rodrigues(self, rvecs, device, epsilon=1e-8, dtype=torch.float32):
        # Calculates the rotation matrices for a batch of rotation vectors
        # Parameters
        # ----------
        # rvecs: torch.tensor Nx3
        #     array of N axis-angle vectors
        # Returns
        # -------
        # R: torch.tensor Nx3x3
        #     The rotation matrices for the given axis-angle parameters
    
        bs = rvecs.shape[0]

        angle = torch.norm(rvecs + 1e-8, dim=1, keepdim=True)
        rot_dir = rvecs / angle

        cos = torch.unsqueeze(torch.cos(angle), dim=1)
        sin = torch.unsqueeze(torch.sin(angle), dim=1)

        # Bx1 arrays
        rx, ry, rz = torch.split(rot_dir, 1, dim=1)
        K = torch.zeros((bs, 3, 3), dtype=dtype, device=device)

        zeros = torch.zeros((bs, 1), dtype=dtype, device=device)
        K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
            .view((bs, 3, 3))

        ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
        rmats = ident + sin * K + (1 - cos) * torch.bmm(K, K)
        return rmats


    def transform_mat(self, R, t):
        # Creates a batch of transformation matrices
        # Args:
        #     - R: Bx3x3 array of a batch of rotation matrices
        #     - t: Bx3x1 array of a batch of translation vectors
        # Returns:
        #     - T: Bx4x4 Transformation matrix

        # No padding left or right, only add an extra row
        return torch.cat([F.pad(R, [0, 0, 0, 1]),
                          F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


    def hpose_limit(self):
        max_ = torch.max((self.hpose_-self.hpose_max_), self.zeros_) # If hpose_ is more than hpose_max_ it will be penalized
        min_ = torch.max((self.hpose_min_-self.hpose_), self.zeros_) # If hpose_ is less than hpose_min_ it will be penalized

        return (torch.sum(max_) + torch.sum(min_))*(torch.sum(max_) + torch.sum(min_))


    def project_point(self, points, camera_intrinsic, camera_extrinsic):
        # Using OpenCV
        # rvec = cv2.Rodrigues(camera_extrinsic[0:3,0:3])[0]
        # tvec = camera_extrinsic[0:3,3]
        # point2D = cv2.projectPoints(np.float32(point3D), rvec, tvec, camera_intrinsic, distCoeffs=0)[0].reshape(-1, 2)
        
        # My own calculation method
        # points [n, 4]
        # camera_matrix [3,4] = camera_intrinsic [3,3] matmul camera_extrinsic [3,4]
        camera_matrix = torch.matmul(camera_intrinsic, camera_extrinsic[:3,:])
        # [3,4] matmul [4, n] -> [3, n] -> [n, 3]
        uv = torch.matmul(camera_matrix, points.transpose(dim0=0, dim1=1)).transpose(dim0=0, dim1=1) 
        # u = (X.fx + Z.cx) / Z
        # v = (Y.fy + Z.cy) / Z
        uv = uv[:, :2] / uv[:, -1:] # [n, 2]        

        return uv


    def project_point2(self, points, camera_matrix):
        # Premultiply the intrinsic with extrinsic to get the camera matrix
        # To help save some time in doing matmul

        # My own calculation method
        # points [n, 4]
        # [3,4] matmul [4, n] -> [3, n] -> [n, 3]
        uv = torch.matmul(camera_matrix, points.transpose(dim0=0, dim1=1)).transpose(dim0=0, dim1=1) 
        # u = (X.fx + Z.cx) / Z
        # v = (Y.fy + Z.cy) / Z
        uv = uv[:, :2] / uv[:, -1:] # [n, 2]        

        return uv      


    def project_point3(self, points, camera_intrinsic, camera_dist, camera_extrinsic):
        # My own calculation method with consideration of distortion
        # points [n, 4]
        # [3,4] matmul [4, n] -> [3, n] -> [n, 3]
        xyz = torch.matmul(camera_extrinsic, points.transpose(dim0=0, dim1=1)).transpose(dim0=0, dim1=1)
        # x = x/z
        # y = y/z
        xy = xyz[:, :2] / xyz[:, -1:] # [n, 2]
        # Extract camera distortion into individual parameter
        k1, k2, p1, p2, k3 = camera_dist
        # r^2 = x^2 + y^2
        r2 = torch.sum(xy**2, dim=1) # [n]
        x = xy[:,0] * (1 + k1*r2 + k2*r2**2 + k3*r2**4) + 2*p1*xy[:,0]*xy[:,1] + p2*(r2 + 2*xy[:,0]**2) # n
        y = xy[:,1] * (1 + k1*r2 + k2*r2**2 + k3*r2**4) + p1*(r2 + 2*xy[:,1]**2) + 2*p2*xy[:,0]*xy[:,1] # n
        fx = camera_intrinsic[0,0]
        fy = camera_intrinsic[1,1]
        cx = camera_intrinsic[0,2]
        cy = camera_intrinsic[1,2]
        u = fx*x + cx # n
        v = fy*y + cy # n
        uv = torch.cat((u.unsqueeze(-1),v.unsqueeze(-1)), dim=1)

        return uv          


    def draw_point(self, point2D, img=None, color=(0,255,0), win_name='0', resize=False):
        if img is None:
            img = np.zeros((480,640,3), dtype=np.uint8)

        if isinstance(point2D, torch.Tensor):
            point2D = point2D.detach().cpu().numpy()
        for i, p in enumerate(point2D):
            if np.isfinite(p[0]) and np.isfinite(p[1]):
                # Draw skeleton
                start = point2D[self.ktree[i],:]
                if start[0]>0 and start[1]>0:
                    cv2.line(img, (int(start[0]), int(start[1])), (int(p[0]), int(p[1])), self.color[i], 3)
        # Note: this is split out from the above loop to get better display
        for i, p in enumerate(point2D):
            if np.isfinite(p[0]) and np.isfinite(p[1]):
                # Draw the joints
                cv2.circle(img, (int(p[0]), int(p[1])), 6, self.color[i], -1)

        if resize: 
            img = cv2.resize(img, None, fx=0.5, fy=0.5)
        cv2.imshow('img' + win_name, img)

        return img


    def print_hpose(self, hpose):
        # Extract the hpose
        hpose = np.degrees(hpose)
        digit1 = hpose[6:11]  # Thumb
        digit2 = hpose[11:15] # Index
        digit3 = hpose[15:19] # Middle
        digit4 = hpose[19:23] # Ring
        digit5 = hpose[23:]   # Little
        # Print out the joint angle
        print('Thumb   MCP:{:.1f} \tIP:{:.1f} \tRx:{:.1f} \tRy:{:.1f} \tRz:{:.1f}'.format(digit1[3], digit1[4], digit1[0], digit1[1], digit1[2]))
        print('Index   MCP:{:.1f} \tPIP:{:.1f} \tDIP:{:.1f} \tADD:{:.1f}'.format(digit2[1], digit2[2], digit2[3], digit2[0]))
        print('Middle  MCP:{:.1f} \tPIP:{:.1f} \tDIP:{:.1f} \tADD:{:.1f}'.format(digit3[1], digit3[2], digit3[3], digit3[0]))
        print('Ring    MCP:{:.1f} \tPIP:{:.1f} \tDIP:{:.1f} \tADD:{:.1f}'.format(digit4[1], digit4[2], digit4[3], digit4[0]))
        print('Little  MCP:{:.1f} \tPIP:{:.1f} \tDIP:{:.1f} \tADD:{:.1f}'.format(digit5[1], digit5[2], digit5[3], digit5[0]))         


###############################################################################
### Using Open3D to visualize the hand skeleton                             ###
###############################################################################
class Open3DHandSkeleton:
    def __init__(self, hand_model, draw_hand_axes=True, enable_trackbar=True):

        joint, _ = hand_model.forward() # Get the first set of joint for Opn3D to visualize
        joint = joint[:,:3].detach().cpu().numpy() # Extract first three columns xyz from the homo coordinate

        self.joint = joint
        self.hpose = hand_model.hpose
        self.bolen = hand_model.bolen
        self.borot = hand_model.borot
        self.camera_intrin = hand_model.camera_intrin
        self.camera_extrin = hand_model.camera_extrin
        self.draw_hand_axes = draw_hand_axes

        # 21 colors for joints
        self.color = [[0,0,0], 
                      [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], # MCP
                      [1, 0.235, 0], [1, 0.471, 0], [1, 0.706, 0], # Thumb red orange
                      [0.235, 1, 0], [0.471, 1, 0], [0.706, 1, 0], # Index green 
                      [0, 1, 0.235], [0, 1, 0.471], [0, 1, 0.706], # Middle cyan blue
                      [0, 0.235, 1], [0, 0.471, 1], [0, 0.706, 1], # Ring blue
                      [0.235, 0, 1], [0.471, 0, 1], [0.706, 0, 1]] # Little blue magenta)
        self.color = [[c[2], c[1], c[0]] for c in self.color] # Convert between BGR and RGB order                                                 

        ############################
        ### Open3D visualization ###
        ############################
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=640, height=480)
        self.vis.get_render_option().point_size = 8.0

        # Draw 21 joints
        self.pcdj = o3d.geometry.PointCloud()
        self.pcdj.points = o3d.utility.Vector3dVector(self.joint)
        self.pcdj.colors = o3d.utility.Vector3dVector(self.color)
        
        # Draw 20 bones
        self.bone = o3d.geometry.LineSet()
        self.bone.points = o3d.utility.Vector3dVector(self.joint)
        self.bone.colors = o3d.utility.Vector3dVector(self.color[1:])
        self.bone.lines  = o3d.utility.Vector2iVector(
                                    [[0,1],[0,2],[0,3],[0,4],[0,5], # MCP
                                     [1,6],[6,7],[7,8],             # Thumb
                                     [2,9],[9,10],[10,11],          # Index
                                     [3,12],[12,13],[13,14],        # Middle
                                     [4,15],[15,16],[16,17],        # Ring
                                     [5,18],[18,19],[19,20]])       # Little

        # Draw 21 axes
        self.axes = []
        # Root axis for camera frame (global reference frame)
        self.axes.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])) 
        if self.draw_hand_axes:
            # Remaining 20 axes on the hand
            for i in range(20):
                self.axes.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0,0,0]))

        # Add the geometry to visualize
        self.vis.add_geometry(self.pcdj)
        self.vis.add_geometry(self.bone)
        for axis in self.axes: 
            self.vis.add_geometry(axis)
        

        ##########################################
        ### OpenCV GUI to articulate the model ###
        ##########################################
        if enable_trackbar:
            cv2.namedWindow('hand_pose', cv2.WINDOW_NORMAL)
            cv2.namedWindow('bone_length', cv2.WINDOW_NORMAL)
            cv2.namedWindow('bone_rotation', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('hand_pose', 640,480)
            cv2.resizeWindow('bone_length', 640,480)
            cv2.resizeWindow('bone_rotation', 640,480)

            for i in range(27): 
                cv2.createTrackbar(str(i), 'hand_pose', 100, 200, self.nothing)
            for i in range(20): 
                cv2.createTrackbar(str(i), 'bone_length', 100, 200, self.nothing)
            for i in range(4) : 
                cv2.createTrackbar(str(i), 'bone_rotation', 100, 200, self.nothing)

            self.reset_trackbar()

            self.jump_to_camera_view()


    def nothing(self, x): pass # For cv2.createTrackbar


    def update_trackbar(self, hpose, bolen, borot):
        for i in range(3): 
            hpose[i] = (cv2.getTrackbarPos(str(i), 'hand_pose')-100)/200 # Scale from [0,200] to [-0.5,0.5]
        for i in range(3,27): 
            hpose[i] = (cv2.getTrackbarPos(str(i), 'hand_pose')-100)/31.8 # Scale from [0,200] to [-3.14,3.14]
        for i in range(20):   
            bolen[i] = (cv2.getTrackbarPos(str(i), 'bone_length'))/1000.0 # Scale from [0,200] to [0,0.2]
        for i in range(4):    
            borot[i] = (cv2.getTrackbarPos(str(i), 'bone_rotation')-100)/31.8 # Scale from [0,200] to [-3.14,3.14]
        
        return hpose, bolen, borot


    def reset_trackbar(self):
        for i in range(3): 
            cv2.setTrackbarPos(str(i), 'hand_pose', int(self.hpose[i]*200+100))
        for i in range(3,27): 
            cv2.setTrackbarPos(str(i), 'hand_pose', int(self.hpose[i]*31.8+100))
        for i in range(20):   
            cv2.setTrackbarPos(str(i), 'bone_length', int(self.bolen[i]*1000))
        for i in range(4) :   
            cv2.setTrackbarPos(str(i), 'bone_rotation', int(self.borot[i]*31.8+100))            


    def jump_to_camera_view(self):
        p = o3d.camera.PinholeCameraParameters()
        p.extrinsic = self.camera_extrin
        p.intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, 
            self.camera_intrin[0,0], self.camera_intrin[1,1], 
            self.camera_intrin[0,2], self.camera_intrin[1,2])
        self.vis.get_view_control().convert_from_pinhole_camera_parameters(p)        


    def update_display(self, joint, trans=None):
        self.pcdj.points = o3d.utility.Vector3dVector(joint)
        self.bone.points = o3d.utility.Vector3dVector(joint)

        if self.draw_hand_axes:
            for i in range(1,21): 
                self.axes[i].transform(trans[i]) 

        self.vis.update_geometry(None)
        self.vis.poll_events()
        self.vis.update_renderer()       

        if self.draw_hand_axes:
            for i in range(1,21): 
                self.axes[i].transform(np.linalg.inv(trans[i]))  


###############################################################################
### Test program                                                            ###
###############################################################################
if __name__ == '__main__':
    import time

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    path = '../data/hp/1views/'
    file = 'open_00'
    hand = HandSkeletonPyTorch(path, file, bone_calib=True)

    disp = Open3DHandSkeleton(hand)
    hpose = np.zeros(27)
    bolen = np.zeros(20)
    borot = np.zeros(4)

    while True:
        hpose, bolen, borot = disp.update_trackbar(hpose, bolen, borot)

        ############################
        ### Update the 3D joints ###
        ############################
        xhpose = torch.from_numpy(hpose).float().to(device)
        xbolen = torch.from_numpy(bolen).float().to(device)
        xborot = torch.from_numpy(borot).float().to(device)
        hand.hpose_ = nn.Parameter(xhpose)
        hand.bolen_ = nn.Parameter(xbolen)
        hand.borot_ = nn.Parameter(xborot)
        joint, trans = hand.forward()

        ############################
        ### Get projected points ###
        ############################
        camera = disp.vis.get_view_control().convert_to_pinhole_camera_parameters()
        intrin = torch.from_numpy(camera.intrinsic.intrinsic_matrix).float() # (4, 4)
        extrin = torch.from_numpy(camera.extrinsic).float() # (4, 4)
        hand.draw_point(hand.project_point(joint, intrin, extrin))        

        # Update joints for Open3D display
        joint = joint[:,:3].detach().cpu().numpy()
        trans = trans.detach().cpu().numpy()
        disp.update_display(joint, trans)

            
        key = cv2.waitKey(10)
        if key==27: # Press escape to end the program
            print('Quitting...')
            break
        if key==ord('r'): # Press 'r' to reset the trackbar
            disp.reset_trackbar()
        if key==ord('j'): # Press 'j' to jump to camera view
            disp.jump_to_camera_view()
            time.sleep(0.1)                     

    # Clean up
    disp.vis.destroy_window()
    cv2.destroyAllWindows()
    
    print('End')
