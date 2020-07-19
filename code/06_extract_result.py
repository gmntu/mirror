###############################################################################
### Extract hand pose generated results                                     ###
###############################################################################

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_hpose(filename, mode):
    
    param = yaml.load(open(filename), Loader=yaml.FullLoader)

    # Extract the hpose
    hpose = np.asarray(param['hand_pose_'+mode])
    hpose = np.degrees(hpose)
    digit1 = hpose[6:11] # Thumb
    digit2 = hpose[11:15] # Index
    digit3 = hpose[15:19] # Middle
    digit4 = hpose[19:23] # Ring
    digit5 = hpose[23:] # Little

    # Print out the joint angle
    # print('Thumb   MCP:{:.1f} \tIP:{:.1f}'.format(digit1[3], digit1[4]))
    # print('Index   MCP:{:.1f} \tPIP:{:.1f} \tDIP:{:.1f}'.format(digit2[1], digit2[2], digit2[3]))
    # print('Middle  MCP:{:.1f} \tPIP:{:.1f} \tDIP:{:.1f}'.format(digit3[1], digit3[2], digit3[3]))
    # print('Ring    MCP:{:.1f} \tPIP:{:.1f} \tDIP:{:.1f}'.format(digit4[1], digit4[2], digit4[3]))
    # print('Little  MCP:{:.1f} \tPIP:{:.1f} \tDIP:{:.1f}'.format(digit5[1], digit5[2], digit5[3])) 

    if 'fist' in filename: # Measure mcp 
        data = [digit2[1],digit3[1],digit4[1],digit5[1]]
    elif 'hook' in filename: # Measure pip and dip
        data = [digit2[2],digit3[2],digit4[2],digit5[2],digit2[3],digit3[3],digit4[3],digit5[3]]
    elif 'thumb_mcp' in filename: # Measure thumb mcp
        data = [digit1[3]]
    elif 'thumb_ip' in filename: # Measure thumb ip
        data = [digit1[4]]

    return data # Return a list so easier to combine later

data = []
for view in range(1,6):
    data += extract_hpose(filename='../data/hp/'+str(view)+'views/fist_00.yaml',      mode='auto')
    data += extract_hpose(filename='../data/hp/'+str(view)+'views/hook_00.yaml',      mode='auto')
    data += extract_hpose(filename='../data/hp/'+str(view)+'views/thumb_mcp_00.yaml', mode='auto')
    data += extract_hpose(filename='../data/hp/'+str(view)+'views/thumb_ip_00.yaml',  mode='auto')

    data += extract_hpose(filename='../data/hp/'+str(view)+'views/fist_03.yaml',      mode='auto')
    data += extract_hpose(filename='../data/hp/'+str(view)+'views/hook_03.yaml',      mode='auto')
    data += extract_hpose(filename='../data/hp/'+str(view)+'views/thumb_mcp_03.yaml', mode='auto')
    data += extract_hpose(filename='../data/hp/'+str(view)+'views/thumb_ip_03.yaml',  mode='auto')


for view in range(1,6):
    data += extract_hpose(filename='../data/hp/'+str(view)+'views/fist_00.yaml',      mode='manual')
    data += extract_hpose(filename='../data/hp/'+str(view)+'views/hook_00.yaml',      mode='manual')
    data += extract_hpose(filename='../data/hp/'+str(view)+'views/thumb_mcp_00.yaml', mode='manual')
    data += extract_hpose(filename='../data/hp/'+str(view)+'views/thumb_ip_00.yaml',  mode='manual')

    data += extract_hpose(filename='../data/hp/'+str(view)+'views/fist_03.yaml',      mode='manual')
    data += extract_hpose(filename='../data/hp/'+str(view)+'views/hook_03.yaml',      mode='manual')
    data += extract_hpose(filename='../data/hp/'+str(view)+'views/thumb_mcp_03.yaml', mode='manual')
    data += extract_hpose(filename='../data/hp/'+str(view)+'views/thumb_ip_03.yaml',  mode='manual')


data = np.asarray(data).reshape(-1, 14)
print(data.shape)

np.savetxt('../data/hp/extract_result.csv', data, delimiter=",")