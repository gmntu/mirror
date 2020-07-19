###############################################################################
### Calibrate camera intrinsics and extrinsics                              ###
### Input : Color images                                                    ###
### Output: Get and save camera intrinsic and extrinsic                     ###
###############################################################################

import cv2
import yaml
import numpy as np


class Calibration:
    def __init__(self, chessboard_size=(6,5), chessboard_sq_size=0.015):
        self.chessboard_size    = chessboard_size
        self.chessboard_sq_size = chessboard_sq_size
        
        # Prepare 3D object points in real world space
        self.obj_pts = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
        # [[0,0,0], [1,0,0], [2,0,0] ....,[9,6,0]]
        self.obj_pts[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2) 
        self.obj_pts *=  chessboard_sq_size # Convert length of each black square to units in meter
        
        # Termination criteria for cornerSubPix
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-3)

        # Flag for findChessboardCorners
        self.flags_findChessboard = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

        # Termination criteria for stereoCalibrate
        self.criteria_stereoCalibrate = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        # Flag for stereoCalibrate
        self.flags_stereoCalibrate = cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_CB_FAST_CHECK


    def get_intrin(self, img_list, filepath=None):
        # Use chessboard pattern to calib intrinsic of color camera
        
        # List to store object points and image points from all the images
        objpt = [] # 3d point in real world space
        imgpt = [] # 2d points in image plane
                
        for i, img in enumerate(img_list):
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, self.flags_findChessboard)

            # If found, add object points, image points (after refining them)
            if ret==True:
                corners2 = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), self.criteria) # (chessboard_size[0]*chessboard_size[1], 1, 2}
                imgpt.append(corners2)
                objpt.append(self.obj_pts)
                # Draw the corners
                img = cv2.drawChessboardCorners(img, self.chessboard_size, corners2, ret)
                print('Found ChessboardCorners', i)
            else:
                print('Cannot find ChessboardCorners', i)
                
        # Calibration
        if len(objpt)>0 and len(imgpt)>0:
            print('Calibrating ...')
            ret, mat, dist, rvec, tvec = cv2.calibrateCamera(objpt, imgpt, gray.shape[::-1], None, None)
            print('Calibrating done')

            # Draw the projected xyz axis on the image
            if filepath is not None:
                for i, img in enumerate(img_list):
                    ret, rvec, tvec = cv2.solvePnP(objpt[i], imgpt[i], mat, dist)     
                    self.project_3Daxis_to_2Dimage(img, mat, dist, rvec, tvec) 
                    cv2.imwrite(filepath+str(i).zfill(2)+'_.png', img)

                    # Get reprojection error
                    error = self.get_reprojection_error(
                                np.asarray(objpt[i]).reshape(-1, 3), # To convert from m list of (n, 3) to (m*n, 3)
                                np.asarray(imgpt[i]).reshape(-1, 2), # To convert from m list of (n, 1, 2) to (m*n, 2)
                                mat, dist, rvec, tvec)
                    print(i, '[Calibration] Reprojection error', error)

            return mat, dist


    def get_extrin_manual(self, img_list, mat, dist, filepath, num_views):
        # Use chessboard to calib extrin of color camera and plane mirrors
        # Need manual input from user to guide the process

        # List to store object points and image points
        objpt = [] # 3d point in real world space
        imgpt0 = [] # 2d points in image plane        
        imgpt1 = [] # 2d points in image plane
        homo_matrix = np.eye(4)  

        total_view = int(num_views)
        print('Total number of views:', total_view, ' and number of images to process:', len(img_list))
        print('For each image, user is required to manually determine:')
        print('1) Press Y if the axis needs to be flipped (have to ensure projected axis follows mirror reflection properties')
        print('2) Label the view number: 0 for actual camera view and 1, 2, ... for virtual camera views (in clockwise direction starting from actual camera view')

        for i, img in enumerate(img_list):
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            for j in range(total_view):
                #########################################
                ### Find the first chessboard corners ###
                #########################################
                ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, self.flags_findChessboard)

                if ret==True:
                    # Refine the corners
                    corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), self.criteria) # (chessboard_size[0]*chessboard_size[1], 1, 2}
                    # Draw the corners
                    dis = img.copy()
                    dis = cv2.drawChessboardCorners(dis, self.chessboard_size, corners, ret)
                    # Get the transformations
                    ret, rvec, tvec = cv2.solvePnP(self.obj_pts, corners, mat, dist)     
                    # Draw the axis
                    self.project_3Daxis_to_2Dimage(dis, mat, dist, rvec, tvec)                     
                    # Display the image
                    dis = cv2.resize(dis, None, fx=0.5, fy=0.5)
                    cv2.imshow('dis', dis)
                    # Let user check if need to flip the corners
                    print('Press y to flip, press enter to continue, or press c to cancel this view:')
                    key = cv2.waitKey(0)
                    if key==ord('c'):
                        print('Cancel this view')
                        continue
                    else:
                        if key==ord('y'):
                            corners = self.flip_corners(corners)
                            corners[:,:,0] = img.shape[1] - corners[:,:,0] # Need to mirror reflect about the x axis also
                            img = cv2.flip(img, flipCode=1)
                            gray = cv2.flip(gray, flipCode=1)
                        # Redraw the corners
                        img = cv2.drawChessboardCorners(img, self.chessboard_size, corners, ret)
                        # Recompute the transformations
                        ret, rvec, tvec = cv2.solvePnP(self.obj_pts, corners, mat, dist)     
                        # Redraw the axis
                        self.project_3Daxis_to_2Dimage(img, mat, dist, rvec, tvec) 
                        # Mask out the first chessboard
                        gray = self.mask_chessboard(gray, corners)  
                        # cv2.imwrite('../data/hp/calib_'+str(j)+'.png', img) # For writing paper create this figure to show calibration process
                        # img = self.mask_chessboard(img, corners)

                        if key==ord('y'):
                            img = cv2.flip(img, flipCode=1)
                            gray = cv2.flip(gray, flipCode=1)
                        # Display the image
                        dis = cv2.resize(img, None, fx=0.5, fy=0.5)
                        cv2.imshow('dis', dis)
                        # Let user decide the view
                        print('Enter the view number')
                        view = cv2.waitKey(0)-48 # Subtract 48 to convert from ASCII code to number
                        print('view', view)

                        # Get reprojection error
                        error = self.get_reprojection_error(
                                    self.obj_pts,
                                    corners.reshape(-1, 2), # To convert from (n, 1, 2) to (n, 2)
                                    mat, dist, rvec, tvec)

                        # Create the 4 by 4 homo matrix [R|T] to transform 3D model coordinate to 2D camera coordinate 
                        homo_matrix = np.hstack((cv2.Rodrigues(rvec)[0], tvec)) # 3 by 4 matrix
                        homo_matrix = np.vstack((homo_matrix, np.array([0,0,0,1]))) # 4 by 4 matrix
                        # if flip==ord('y'):
                        #     # homo_matrix = np.matmul(homo_matrix, np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])) # Need to flip the z axis as during opencv detection the z axis is pointing up
                        #     # homo_matrix = np.matmul(np.array([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), homo_matrix) # Need to do mirror reflection about x axis
                        #     corners[:,:,0] = img.shape[1] - corners[:,:,0] # Need to mirror reflect about the x axis also
                        
                        # Proceed to save all the parameters for multi-camera calibration
                        results = dict( 
                            relative_transformation_matrix=homo_matrix.tolist(), 
                            detected_2D_corners=corners.reshape(-1, 2).tolist(), 
                            reprojection_error=error, 
                        )

                        results_camera = dict(
                            # camera_intrinsics_matrix=c.camera_matrix_infra.tolist(),
                            camera_intrinsics_matrix=mat.tolist(),
                            camera_dist_coeffs=dist.flatten().tolist(),
                            camera_height=img.shape[0],
                            camera_width=img.shape[1],
                            camera_serial_number=view,
                        )

                        results_pattern = dict(
                            checkerboard_col=self.chessboard_size[0],
                            checkerboard_row=self.chessboard_size[1],
                            checkerboard_sq_size=self.chessboard_sq_size,
                        )

                        with open(filepath+'multicam/C'+str(view)+'_P'+str(i)+'_.yaml', 'w') as outfile:
                            yaml.dump(results, outfile, default_flow_style=False)

                        with open(filepath+'multicam/C'+str(view)+'.yaml', 'w') as outfile:
                            yaml.dump(results_camera, outfile, default_flow_style=False)

                        with open(filepath+'multicam/P'+str(i)+'.yaml', 'w') as outfile:
                            yaml.dump(results_pattern, outfile, default_flow_style=False)                    

            cv2.imwrite(filepath+str(i).zfill(2)+'_.png', img)


    def mask_chessboard(self, img, corners):
        # Get the four corners
        c0 = corners[0,:,:].astype(np.int32)
        c1 = corners[self.chessboard_size[0]-1,:,:].astype(np.int32)
        c2 = corners[-self.chessboard_size[0],:,:].astype(np.int32)
        c3 = corners[-1,:,:].astype(np.int32)
        c_ = np.array([c0,c1,c3,c2])

        img_ = img.copy()
        cv2.fillPoly(img_, [c_], color=(255,255,255)) # Mask out the corners with white polygon

        return img_   


    def project_3Daxis_to_2Dimage(self, img, mat, dist, rvec, tvec):
        axis_3D = np.float32([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]) * 2 * self.chessboard_sq_size # Create a 3D axis that is twice the length of chessboard_sq_size
        axis_2D = cv2.projectPoints(axis_3D, rvec, tvec, mat, dist)[0].reshape(-1, 2)
        if axis_2D.shape[0]==4:
            colours = [(0,0,255),(0,255,0),(255,0,0)] # BGR
            for i in range(1,4):
                (x0, y0), (x1, y1) = axis_2D[0], axis_2D[i]
                cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), colours[i-1], 5)    


    def get_reprojection_error(self, p3D, p2D, mat, dist, rvec, tvec):
        p2D_reproj = cv2.projectPoints(p3D, rvec, tvec, mat, dist)[0].reshape(-1, 2)
        error = cv2.norm(p2D, p2D_reproj, cv2.NORM_L2) / len(p2D)

        return error # https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html


    def flip_corners(self, corners):
        col, row = self.chessboard_size[0], self.chessboard_size[1]
        temp = corners.copy()
        # for r in range(row):
        #   temp[r*col:r*col+col, :, :] = corners[(row-r-1)*col:(row-r-1)*col+col:, :, :] # Note: For (7,4) checkerboard
        for r in range(row):
            for c in range(col):
                temp[r*col+c, :, :] = corners[r*col+(col-c-1), :, :] # Note: For (10,7) checkerboard

        return temp
