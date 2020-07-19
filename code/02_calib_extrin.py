###############################################################################
### Calibrate extrinsics of camera                                          ###
### Input : Multicam data that is prepared by 01_prep_extrin.py             ###
### Output: Save camera extrinsics                                          ###
### Usage : python 02_calib_extrin.py --view 2                              ###
### Usage : python 02_calib_extrin.py --view 3                              ###
### Usage : python 02_calib_extrin.py --view 4                              ###
### Usage : python 02_calib_extrin.py --view 5                              ###
### Note  : The optimization process will take a few minutes                ###
###############################################################################

import argparse
from utils_multicam import *


# Get user input
parser = argparse.ArgumentParser()
parser.add_argument('--view', default=2) # Set the number of views (2 to 5)
args = parser.parse_args()

# Folder multicam that contains data prepared by 01_prep_extrin.py
filepath = '../data/hp/'+str(args.view)+'views/extrin/'

##################################
### Initialize the global pose ###
##################################
graph = init_global_pose(filepath+'multicam/')


############################################################################
### Setup the camera and checkboard pose for displaying the initial pose ###
############################################################################
cameras = []
checkerboards = []
nodes = graph.nodes()
for n in nodes:
    print(n)
    if n[0] == 'C': # Camera node
        tvec, rvec = get_tvec_rvec(graph.nodes[n]['global_pose'])
        fx = graph.nodes[n]['intrinsics'][0][0]
        fy = graph.nodes[n]['intrinsics'][1][1]
        px = graph.nodes[n]['intrinsics'][0][2]
        py = graph.nodes[n]['intrinsics'][1][2]
        width = graph.nodes[n]['width']
        height = graph.nodes[n]['height']
        cameras.append(camera_sim(tvec,rvec,fx,fy,px,py,width,height))

        # matrix = dict(model2camera_matrix=graph.nodes[n]['global_pose'].tolist())
        # # Save in human readable format
        # with open(filepath+'model2camera_matrix_' + str(graph.nodes[n]['serial_number']) + '.yaml', 'w') as outfile:
        #     yaml.dump(matrix, outfile, default_flow_style=False)
        # print('Saved model2camera_matrix', str(graph.nodes[n]['serial_number']))        

    elif n[0] == 'P': # Pattern node
        tvec, rvec = get_tvec_rvec(graph.nodes[n]['global_pose'])
        row = graph.nodes[n]['row']
        col = graph.nodes[n]['col']
        size = graph.nodes[n]['size']
        checkerboards.append(checkerboard_sim(tvec,rvec,row,col,size))


################################
### Optimize the global pose ###
################################
graph = optimize_global_pose(filepath+'multicam/', graph)


##############################################################################
### Setup the camera and checkboard pose for displaying the optimized pose ###
##############################################################################
cameras2 = []
checkerboards2 = []
nodes = graph.nodes()
for n in nodes:
    if n[0] == 'C': # Camera node
        tvec, rvec = get_tvec_rvec(graph.nodes[n]['global_pose'])
        fx = graph.nodes[n]['intrinsics'][0][0]
        fy = graph.nodes[n]['intrinsics'][1][1]
        px = graph.nodes[n]['intrinsics'][0][2]
        py = graph.nodes[n]['intrinsics'][1][2]
        width = graph.nodes[n]['width']
        height = graph.nodes[n]['height']
        cameras2.append(camera_sim(tvec,rvec,fx,fy,px,py,width,height))

        matrix = dict(model2camera_matrix=graph.nodes[n]['global_pose'].tolist())
        # Save in human readable format
        filename = 'model2camera_matrix_' + str(graph.nodes[n]['serial_number']) + '.yaml' 
        with open(filepath+filename, 'w') as outfile:
            yaml.dump(matrix, outfile, default_flow_style=False)
        print('Saved model2camera_matrix', str(graph.nodes[n]['serial_number']))

    elif n[0] == 'P': # Pattern node
        tvec, rvec = get_tvec_rvec(graph.nodes[n]['global_pose'])
        row = graph.nodes[n]['row']
        col = graph.nodes[n]['col']
        size = graph.nodes[n]['size']
        checkerboards2.append(checkerboard_sim(tvec,rvec,row,col,size))

        # matrix = dict(pattern_pose=graph.nodes[n]['global_pose'].tolist())
        # # Save in human readable format
        # with open(filepath+str(n)+'_pose.yaml', 'w') as outfile:
        #     yaml.dump(matrix, outfile, default_flow_style=False)
        # print('Saved pattern pose', str(n))        


############################
### Open3D visualization ###
############################
vis = o3d.visualization.Visualizer()
# vis.create_window(width=width, height=height)
vis.create_window(width=640, height=480) # Reduce window size so that it can fit to my display

# Global reference coordinate at the origin
mesh_global = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0,0,0]) 
vis.add_geometry(mesh_global)

# Camera pose (red, green, blue, cyan, magenta, yellow)
colors = [[1,0,0],[0,1,0],[0,0,1],[0,1,1],[1,0,1],[1,1,0]] 
mesh_camera = []
for i, c in enumerate(cameras):
    mesh_camera.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0,0,0]))
    # mesh_camera[i].paint_uniform_color(colors[i])
    # if i==1 or i==4:
    #     c.homo_matrix = np.matmul(c.homo_matrix, np.array([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
    mesh_camera[i].transform(c.homo_matrix)
    print('Original', i)
    print(c.homo_matrix)
    vis.add_geometry(mesh_camera[i]) # Draw the camera pose

mesh_camera2 = []
for i, c in enumerate(cameras2):
    mesh_camera2.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0,0,0]))
    mesh_camera2[i].paint_uniform_color(colors[i])
    mesh_camera2[i].transform(c.homo_matrix)
    print('New')
    print(c.homo_matrix)
    vis.add_geometry(mesh_camera2[i]) # Draw the camera pose

# Checkerboard pose
mesh_checkerboard = []
line_set_checkerboard = []
for i, c in enumerate(checkerboards):
    mesh_checkerboard.append(c.mesh)
    mesh_checkerboard[i].transform(c.homo_matrix)
    vis.add_geometry(mesh_checkerboard[i]) # Draw the checkerboard pattern
    
    line_set_checkerboard.append(c.line_set)
    line_set_checkerboard[i].transform(c.homo_matrix)
    vis.add_geometry(line_set_checkerboard[i]) # Draw the frame enclosing the checkerboard

mesh_checkerboard2 = []
line_set_checkerboard2 = []
for i, c in enumerate(checkerboards2):
    mesh_checkerboard2.append(c.mesh)
    mesh_checkerboard2[i].transform(c.homo_matrix)
    # vis.add_geometry(mesh_checkerboard2[i]) # Draw the checkerboard pattern
    
    line_set_checkerboard2.append(c.line_set)
    line_set_checkerboard2[i].transform(c.homo_matrix)
    vis.add_geometry(line_set_checkerboard2[i]) # Draw the frame enclosing the checkerboard 

print('Press escape to end the program')
vis.run() 


################
### Clean up ###
################
vis.destroy_window()
print('End')
