###############################################################################
### Set of utility functions for multicam setup and cam pose optimization   ###
###############################################################################

import os
import cv2
import yaml
import numpy as np
import open3d as o3d # 0.9.0.0
import networkx as nx # pip install networkx==2.4
import matplotlib.pyplot as plt

from lmfit import Minimizer, Parameters, report_fit # pip install lmfit==1.0.1


def get_homo_matrix(tvec, rvec): 
    # Simple function to create 4x4 homo matrix from translation and rotation vector
    homo_matrix = np.hstack((cv2.Rodrigues(rvec)[0], tvec)) # 3 by 4 matrix
    homo_matrix = np.vstack((homo_matrix, np.array([0,0,0,1]))) # 4 by 4 matrix

    return homo_matrix


def get_tvec_rvec(homo): 
    # Reverse process of the above function to get back 
    # translation and rotation vector from the homo matrix
    tvec = homo[0:3,3]
    rvec = cv2.Rodrigues(homo[0:3,0:3])[0]

    return tvec, rvec


###################################################
### Simluate virtual camera pose and parameters ###
###################################################
class camera_intrinsics: 
    def __init__(self, fx, fy, px, py, width, height):
        self.intrinsic_matrix = np.array([[fx, 0 , px],
                                         [0  , fy, py],
                                         [0  , 0 , 1]], dtype=np.float32)
        self.width  = width
        self.height = height


class camera_sim:
    def __init__(self, tvec, rvec, fx, fy, px, py, width, height):
        self.tvec   = np.asarray(tvec, dtype=np.float32).reshape(3,1) # Vector size 3
        self.rvec   = np.asarray(rvec, dtype=np.float32).reshape(3,1) # Vector size 3
        self.fx     = fx
        self.fy     = fy
        self.px     = px
        self.py     = py
        self.intrinsics = camera_intrinsics(fx, fy, px, py, width, height)
        self.dist_coeffs = np.zeros(4)
        self.homo_matrix = get_homo_matrix(self.tvec, self.rvec) # Matrix 4x4


##################################################
### Simulate the checkerboard pose and pattern ###
##################################################
class checkerboard_sim:
    def __init__(self, tvec, rvec, row, col, sq_size):
        self.tvec   = np.asarray(tvec, dtype=np.float32).reshape(3,1) # Vector size 3
        self.rvec   = np.asarray(rvec, dtype=np.float32).reshape(3,1) # Vector size 3
        self.row    = row+1
        self.col    = col+1
        self.size   = sq_size
        self.mesh   = self.create_mesh()
        self.line_set = self.create_line_set()
        self.homo_matrix = get_homo_matrix(self.tvec, self.rvec) # Matrix 4x4


    def create_mesh(self):
        # Origin at top left of the checkerboard
        # Axis is +ve X points to right, +ve Y points downwards
        # o-->X
        # |
        # Y
        # Vertex order is anti clockwise
        # 0 ---- 3
        # |      |
        # 1 ---- 2
        vertices    = []
        triangles   = []    # Must be anti clockwise order when view from outside of the mesh
        black       = True  # Use to alternate between black and white square when loop across row and col
        index       = 0     # To keep track of the number of vertices
        for i in range(self.row): # In +ve Y axis direction
            for j in range(self.col): # In +ve X axis direction
                    if black:
                        x0, y0 = j*self.size, i*self.size                       # In anti clockwise order from top left
                        x1, y1 = j*self.size, i*self.size+self.size             # bottom left
                        x2, y2 = j*self.size+self.size, i*self.size+self.size   # bottom right
                        x3, y3 = j*self.size+self.size, i*self.size             # top right
                        vertices.append([x0, y0, 0])
                        vertices.append([x1, y1, 0])
                        vertices.append([x2, y2, 0])
                        vertices.append([x3, y3, 0])
                        triangles.append([index, index+1, index+2])
                        triangles.append([index, index+2, index+3])
                        index += 4

                    black = not black # Toggle the flag for next square
            
            if self.col%2 == 0: # Important: Need to check if col is even else will get parallel black strips as for even col the sq in the next row follw the same color
                black = not black

        # To shift the origin to the bottom right of first top left black square
        vertices = np.asarray(vertices) - np.array([self.size, self.size, 0])

        mesh            = o3d.geometry.TriangleMesh()
        mesh.vertices   = o3d.utility.Vector3dVector(vertices)
        mesh.triangles  = o3d.utility.Vector3iVector(triangles)
        mesh.paint_uniform_color([0,0,0]) # Black color

        return mesh


    def create_line_set(self):
        # Draw 4 lines to enclose the checkerboard
        points  = [[0,0,0],                                     # Top left
                  [0,self.row*self.size,0],                     # Bottom left
                  [self.col*self.size, self.row*self.size,0],   # Bottom right
                  [self.col*self.size,0,0]]                     # Top right
        lines   = [[0,1], [1,2], [2,3], [3,0]]
        colors  = [[0,0,0] for i in range(len(lines))] # Set to uniform black color
        
        # To shift the origin to the bottom right of first top left black square
        points = np.asarray(points) - np.array([self.size, self.size, 0])
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines  = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set         


#######################################################################
### Use graph data struct to initalize the global pose of all nodes ###
#######################################################################
def init_global_pose(filepath):
    # Get camera and pattern index
    filenames = os.listdir(filepath) # Get the filenames of all results

    camera_index = []
    pattern_index = []
    for f in filenames: 
        if not '_' in f:
            if f[0] == 'C':
                camera_index.append(f.split('.')[0]) # Only get the index (C0) before . e.g C0.yaml
            elif f[0] == 'P':
                pattern_index.append(f.split('.')[0]) # Only get the index (P0) before . e.g P0.yaml

    # print('camera_index', camera_index)
    # print('pattern_index', pattern_index)

    # Create the nodes of the graph based on the total number of cameras and patterns
    # Camera nodes are labelled as CX, where X is the camera index
    # Pattern nodes are labelled as PX, where X is the pattern index
    # E.g. C0 C1 C2 P0 P1 P2 ...
    # Fortunately the graph will only add unique nodes so we dont need to check for duplicate just let it loop through all the index
    G = nx.Graph()
    for c in camera_index:
        params = yaml.load(open(filepath+c+'.yaml'), Loader=yaml.FullLoader)
        intrinsics = np.asarray(params['camera_intrinsics_matrix'])
        width = np.asarray(params['camera_width'])
        height = np.asarray(params['camera_height'])
        serial_number = np.asarray(params['camera_serial_number'])
        G.add_node(c, global_pose=np.identity(4), intrinsics=intrinsics, width=width, height=height, serial_number=serial_number) # Also init the global pose of camara node as attribute of the node

    for p in pattern_index:
        params = yaml.load(open(filepath+p+'.yaml'), Loader=yaml.FullLoader)
        row = np.asarray(params['checkerboard_row'])
        col = np.asarray(params['checkerboard_col'])
        size = np.asarray(params['checkerboard_sq_size'])
        G.add_node(p, global_pose=np.identity(4), row=row, col=col, size=size) # Also init the global pose of pattern node as attribute of the node

    # Create the edges of the graph based on the number of calibrated images
    for f in filenames: # Files are named in CX_PY_.yaml (X is camera index, Y is pattern index)
        if '_' in f:
            c, p, _ = f.split('_') # Get the camera and pattern index
            params = yaml.load(open(filepath+c+'_'+p+'_.yaml'), Loader=yaml.FullLoader)
            matrix = np.asarray(params['relative_transformation_matrix'])

            G.add_edge(c, p, relative_pose=matrix)

    # print('Num of nodes', G.number_of_nodes())
    # print('Num of edges', G.number_of_edges())

    # Perform breadth first search starting from the first camera node C0 (Hardcoded)
    bfs = list(nx.bfs_edges(G,'C0'))
    
    # Initialize the global pose of all the nodes
    for parent, child in bfs:
        # print(parent, child)
        # print(G[parent][child]['relative_pose'])
        # print(parent, G.nodes[parent]['global_pose'])
        # print(child, G.nodes[child]['global_pose'])
        
        if child[0] == 'C': # Compute global pose for camera node
            G.nodes[child]['global_pose'] = np.matmul(G.nodes[parent]['global_pose'], np.linalg.inv(G[parent][child]['relative_pose']))
            # print('Camera node', child, '\n', get_tvec_rvec(G.nodes[child]['global_pose']))
        elif child[0] == 'P': # Compute global pose for pattern node
            G.nodes[child]['global_pose'] = np.matmul(G.nodes[parent]['global_pose'], G[parent][child]['relative_pose'])
            # print('Pattern node', child, '\n', get_tvec_rvec(G.nodes[child]['global_pose']))

    # nx.draw(G, with_labels=True)
    # plt.show()

    return G


################################
### Optimize the global pose ###
################################
def fcn2min(parameters, G, filepath, obj_pts): # Define objective function
    error = []
    edges = G.edges()
    for e in edges:
        # Not sure if need to check will the edge returned in random order
        # if e[0][0] == 'C':
        #   c, p = e # Camera index first followed by pattern index
        # elif e[0][0] == 'P':
        #   p, c = e # Pattern index first followed by camera index

        # Extract the translation and rotation vector to be optimized from parameters
        c, p = e # Assume camera index first followed by pattern index
        if c == 'C0': # Dont need to optimize the first camera
            tvec = np.zeros(3).reshape(3,1)
            rvec = np.zeros(3)
        else:
            tvec = np.array([parameters[c+'_tx'],parameters[c+'_ty'],parameters[c+'_tz']]).reshape(3,1)
            rvec = np.array([parameters[c+'_rx'],parameters[c+'_ry'],parameters[c+'_rz']])

        # Create the global camera pose (homo matrix) from the translation and rotation vector
        global_camera_pose = get_homo_matrix(tvec, rvec)

        # Similarly create the global pattern pose (homo matrix) from the translation and rotation vector
        tvec = np.array([parameters[p+'_tx'],parameters[p+'_ty'],parameters[p+'_tz']]).reshape(3,1)
        rvec = np.array([parameters[p+'_rx'],parameters[p+'_ry'],parameters[p+'_rz']])
        global_pattern_pose = get_homo_matrix(tvec, rvec)

        # Now given the global camera and pattern pose we can compute a relative transformation from pattern to camera
        # Hij = inv(Hi) mul Hj
        # Where Hij is the relative pose from pattern to camera
        # Hi is the camera global pose
        # Hj is the pattern global pose
        relative_pose   = np.matmul(np.linalg.inv(global_camera_pose), global_pattern_pose)
        tvec, rvec      = get_tvec_rvec(relative_pose)
        
        # Read back the camera parameters
        params          = yaml.load(open(filepath+c+'.yaml'), Loader=yaml.FullLoader)
        camera_matrix   = np.asarray(params['camera_intrinsics_matrix'])
        dist_coeffs     = np.asarray(params['camera_dist_coeffs'])

        # Read back the original 2D corner detections that were precomputed during the logging of data
        params          = yaml.load(open(filepath+c+'_'+p+'_.yaml'), Loader=yaml.FullLoader)
        corner_ori      = np.asarray(params['detected_2D_corners']) 

        # Compute the new 2D corners that were projected using the updated relative pose
        corner_new      = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)[0].reshape(-1, 2)
        corner_new      = corner_new.astype(np.float64)

        # Compute the difference in reprojection error between the 
        error.append(corner_ori-corner_new)

    error = np.asarray(error) # If use LM must return the array to be minimized

    return error 


def optimize_global_pose(filepath, G):
    # Note: Hardcode the pattern index to start from P0 and assume all patterns have the same size
    checkerboard_row = G.nodes['P0']['row']
    checkerboard_col = G.nodes['P0']['col']
    checkerboard_sq_size = G.nodes['P0']['size']

    # Prepare 3D object points in real world space
    obj_pts = np.zeros((checkerboard_row*checkerboard_col,3), np.float64)
    obj_pts[:,:2] = np.mgrid[0:checkerboard_col,0:checkerboard_row].T.reshape(-1,2) # [[0,0,0], [1,0,0], [2,0,0] ....,[9,6,0]]
    obj_pts *=  checkerboard_sq_size # Convert length of each black square to units in meter

    # Create a set of Parameters to be optimised
    params = Parameters()
    nodes = G.nodes()
    for n in nodes: 
        if n != 'C0': # Dont need to optimize the first camera
            tvec, rvec = get_tvec_rvec(G.nodes[n]['global_pose'])
            params.add(n+'_tx', value=tvec[0])
            params.add(n+'_ty', value=tvec[1])
            params.add(n+'_tz', value=tvec[2])
            params.add(n+'_rx', value=rvec[0,0]) # Note: rvec is (3,1)!
            params.add(n+'_ry', value=rvec[1,0])
            params.add(n+'_rz', value=rvec[2,0])

    # Optimize with leastsq model
    leastsq_kws={'max_nfev': 100} # Set the maximum number of iterations
    lmmin = Minimizer(fcn2min, params, fcn_args=(G, filepath, obj_pts), **leastsq_kws)
    print('[utils_multicam] Optimizing ......')
    result = lmmin.minimize()

    # Print out detailed error report
    report_fit(result)

    # Update the new global pose
    for n in nodes: 
        if n != 'C0': # Dont need to optimize the first camera
            tvec, rvec = np.zeros(3), np.zeros(3)
            tvec[0] = result.params[n+'_tx'].value
            tvec[1] = result.params[n+'_ty'].value
            tvec[2] = result.params[n+'_tz'].value
            rvec[0] = result.params[n+'_rx'].value
            rvec[1] = result.params[n+'_ry'].value
            rvec[2] = result.params[n+'_rz'].value
            G.nodes[n]['global_pose'] = get_homo_matrix(tvec.reshape(3,1), rvec)


    # error = []
    # edges = G.edges()
    # for e in edges:
    #   # Not sure if need to check will the edge returned in random order
    #   # if e[0][0] == 'C':
    #   #   c, p = e # Camera index first followed by pattern index
    #   # elif e[0][0] == 'P':
    #   #   p, c = e # Pattern index first followed by camera index

    #   c, p = e # Assume camera index first followed by pattern index
    #   global_camera_pose = G.nodes[c]['global_pose']
    #   global_pattern_pose = G.nodes[p]['global_pose']

    #   relative_pose   = np.matmul(np.linalg.inv(global_camera_pose), global_pattern_pose)
    #   tvec, rvec      = get_tvec_rvec(relative_pose)
        
    #   params          = yaml.load(open(filepath+c+'.yaml'), Loader=yaml.FullLoader)
    #   camera_matrix   = np.asarray(params['camera_intrinsics_matrix'])
    #   dist_coeffs     = np.asarray(params['camera_dist_coeffs'])

    #   params          = yaml.load(open(filepath+c+'_'+p+'_.yaml'), Loader=yaml.FullLoader)
    #   corner_ori      = np.asarray(params['detected_2D_corners'], dtype=np.float32)   
    #   # error_ori         = np.asarray(params['reprojection_error'])  
        
    #   # Compute the reprojection error
    #   corner_new      = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)[0].reshape(-1, 2)
    #   # error             = cv2.norm(corner_ori, corner_new, cv2.NORM_L2) / len(corner_ori)
    #   # error             = cv2.norm(corner_ori, corner_new, cv2.NORM_L2)
    #   # print(c,p,error_ori,error_new)
    #   # print(c,p,error)

    #   error.append(corner_ori-corner_new)

    # error = np.asarray(error) # 3 dimemsional array e.g. (4,70,2) 4 cam-pat, 70 corners, 2 coordinates

    return G


