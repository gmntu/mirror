###############################################################################
### Extract 2D hand keypoints from color image                              ###
### Input: Color image                                                      ###
### Output: 21 2D hand keypoints                                            ###
###############################################################################

import cv2
import time
import numpy as np
import tensorflow as tf # For TF1.0
# import tensorflow.compat.v1 as tf # A quick hack to run 1.X code unmodified in TF2.0
# tf.disable_v2_behavior()


###############################################################################
### For TensorFlow to load graph                                            ###
###############################################################################
class Estimator(object):
    def __init__(self, model_file, input_layer='input_1', output_layer='k2tfout_0'):
        
        input_name = 'import/' + input_layer
        output_name = 'import/' + output_layer

        self.graph = self.load_graph(model_file)
        self.input_operation = self.graph.get_operation_by_name(input_name)
        self.output_operation = self.graph.get_operation_by_name(output_name)
        self.sess = tf.Session(graph=self.graph)


    def load_graph(self, model_file):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, 'rb') as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph        


    def predict(self, img):
        results = self.sess.run(self.output_operation.outputs[0], feed_dict={self.input_operation.outputs[0]: img})
        return np.squeeze(results)


###############################################################################
### Use FORTH Gouidis to get hand keypoints                                 ###
### Adapted from                                                            ###
### https://github.com/FORTH-ModelBasedTracker/MonocularRGB_2D_Handjoints_MVA19
###############################################################################
class MobileNetHand:
    def __init__(self, top_left_x=208, top_left_y=128):
        
        ##########################
        ### Load trained model ###
        ##########################
        model_file   = '../model/mobilenet/mobnet4f_cmu_adadelta_t1_model.pb'
        input_layer  = 'input_1'
        output_layer = 'k2tfout_0'
        self.net = Estimator(model_file, input_layer, output_layer)
        
        print('[MobileNetHand] Loaded model from', MobileNetHand)

        # User parameters
        self.stride = 4            
        self.bbsize = 224 # Bounding box size

        self.keypt = np.zeros((21,3), dtype=np.float32) # Note: (x_coor, y_coor, confidence)
        self.keypt_bb = np.zeros((21,3), dtype=np.float32) # Note: (x_coor, y_coor, confidence) with respect to small bounding box dimension
        
        # Initialize top left corners of the bounding box
        self.top_left_x = top_left_x # 320 - 112 = 208
        self.top_left_y = top_left_y # 240 - 112 = 128

        # Joint labeling (Thumb, Index, Middle, Ring, Little) 
        # T  I  M  R  L
        # 4  8 12  16 20 (Tip)
        # 3  7 11  15 19 (DIP)
        # 2  6 10  14 18 (PIP)
        # 1  5  9  13 17 (MCP)
        #       0 (Wrist)
        
        # Convert the above Mobilenet joint convention to
        # BigHand convention:
        #  0:Wrist, 
        #  1:TMCP, 2:IMCP, 3:MMCP, 4:RMCP, 5:PMCP, (All the 5 MCP)(knuckles)
        #  6:TPIP,  7:TDIP,  8:TTIP (Thumb)
        #  9:IPIP, 10:IDIP, 11:ITIP (Index)
        # 12:MPIP, 13:MDIP, 14:MTIP (Middle)
        # 15:RPIP, 16:RDIP, 17:RTIP (Ring)
        # 18:PPIP, 19:PDIP, 20:PTIP (Pinky)
        self.index = [0,           # Wrist  
                      1,5,9,13,17, # All the 5 MCP (knuckles)
                      2,3,4,       # Thumb
                      6,7,8,       # Index
                      10,11,12,    # Middle
                      14,15,16,    # Ring
                      18,19,20]    # Little

        # Define kinematic tree based on BigHand convention linking the keypt together to form the skeleton
        self.ktree = [0,        # Wrist
                      0,0,0,0,0,# (All the 5 MCP)(knuckles)
                      1,6,7,    # Thumb
                      2,9,10,   # Index
                      3,12,13,  # Middle
                      4,15,16,  # Ring
                      5,18,19]  # Little

        # Define the joint color based on BigHand convention
        self.color = [[0,0,0], 
                      [255, 0, 0], [0, 255, 0], [0, 255, 0], [0, 0, 255], [0, 0, 255], # (All the 5 MCP)(knuckles)
                      [255, 60, 0], [255, 120, 0], [255, 180, 0],                      # Thumb
                      [60, 255, 0], [120, 255, 0], [180, 255, 0],                      # Index
                      [0, 255, 60], [0, 255, 120], [0, 255, 180],                      # Middle
                      [0, 60, 255], [0, 120, 255], [0, 180, 255],                      # Ring
                      [60, 0, 255], [120, 0, 255], [180, 0, 255]]                      # Little


    def get_hand_keypoints(self, img, crop=True):
        #########################
        ### Crop out hand ROI ###
        #########################
        if crop:
            cnt, xx, yy = 0, 0, 0
            # Get mean position of the hand keypoints
            for i, kp in enumerate(self.keypt_bb):
                if kp[0]>0 and kp[1]>0:
                    xx += kp[0]
                    yy += kp[1]
                    cnt += 1
            if cnt > 0:
                self.top_left_x += int(xx/cnt - self.bbsize/2)
                self.top_left_y += int(yy/cnt - self.bbsize/2)

        row, col, _ = img.shape # (480, 640)
        x1 = max(self.top_left_x, 0)
        y1 = max(self.top_left_y, 0)
        x2 = min(x1 + self.bbsize, col)
        y2 = min(y1 + self.bbsize, row)
        img_bb = img[y1:y2, x1:x2, :] # Crop out a bounding box 224 by 224
        # Draw the bounding box on the input image
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0))

        
        ######################################
        ### Preprocess the raw input image ###
        ######################################
        if img_bb.shape[0]==224 and img_bb.shape[1]==224:
            img_preprocess = img_bb[np.newaxis,:,:,:] # Required shape (1, height, width, channels)
        else:
            # Seems like the below step is not necessary unless the hand img crop is smaller than bounding box
            img_preprocess = self.preprocess(img_bb, self.bbsize, self.stride) # (1, 224, 224, 3)


        ##################################
        ### Feedforward to the network ###
        ##################################
        heatmap = self.net.predict(img_preprocess)
        heatmap = cv2.resize(heatmap, (0, 0), fx=self.stride, fy=self.stride) # (224, 224, 22)


        ################################
        ### Extract the 21 keypoints ###
        ################################
        # heatmap_comb = np.zeros((224,224))
        for i in range(21):
            # Find global maxima of the probMap
            minVal, prob, minLoc, point = cv2.minMaxLoc(heatmap[:,:,i])
            self.keypt[i,:] = (point[0]+x1, point[1]+y1, prob) # With reference to the original image
            self.keypt_bb[i,:] = (point[0], point[1], prob) # With reference to the bounding box

            # print(i, np.min(heatmap[:,:,i]), np.max(heatmap[:,:,i]), prob)
            # heatmap_comb += heatmap[:,:,i]
        
        # cv2.imwrite('heatmap_comb.png', heatmap_comb*255)
        # cv2.imshow('heatmap_comb', heatmap_comb)

        # Reorder to bighand keypoints
        self.keypt = self.keypt[self.index, :]
        self.keypt_bb = self.keypt_bb[self.index, :]

        return self.keypt_bb, self.keypt, img_bb


    def draw_hand_keypoints(self, img, keypoints):
        for i, kp in enumerate(keypoints):
            if kp[0]>0 and kp[1]>0:
                # Draw the joints
                cv2.circle(img, (int(kp[0]), int(kp[1])), 6, self.color[i], -1) 
                # Number the keypoints
                # cv2.putText(img, "{}".format(i), (kp[0], kp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color[i])
                # Draw skeleton
                start = keypoints[self.ktree[i],:]
                if start[0]>0 and start[1]>0:
                    # cv2.line(img, (int(start[0]), int(start[1])), (int(kp[0]), int(kp[1])), self.color[i], 3)
                    cv2.line(img, (int(start[0]), int(start[1])), (int(kp[0]), int(kp[1])), self.color[i], 1)

        return img


    def get_hand_keypoints_single(self, img, threshold=0.1):
        crop    = cv2.resize(img, (self.bbsize, self.bbsize))
        img_new = self.preprocess(crop, self.bbsize , self.stride)


        ##################################
        ### Feedforward to the network ###
        ##################################
        heatmap = self.net.predict(img_new)
        heatmap = cv2.resize(heatmap, (0, 0), fx=self.stride, fy=self.stride) # (224, 224, 22)


        ################################
        ### Extract the 21 keypoints ###
        ################################
        keypoints = []
        for i in range(21):
            # Find global maxima of the probMap
            minVal, prob, minLoc, point = cv2.minMaxLoc(heatmap[:,:,i])
            # Append to keypoints
            self.keypt_bb[i,:] = (point[0], point[1], prob)

        # Reorder to bighand keypoints
        self.keypt_bb = self.keypt_bb[self.index, :]            

        return self.keypt_bb, crop


    def draw_hand_keypoints_single(self, img, keypoints):
        for i, kp in enumerate(keypoints):
            if not kp==None:
                # Draw the joints
                cv2.circle(img, (kp[0], kp[1]), 4, self.color[i], -1) 
                # Number the keypoints
                # cv2.putText(img, "{}".format(i), (kp[0], kp[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color[i])
                # Draw skeleton
                start = keypoints[self.ktree[i]]
                if not start==None:
                    cv2.line(img, (start[0], start[1]), (kp[0], kp[1]), self.color[i], 3)

        return img      


    def preprocess(self, oriImg, bbsize=224, stride=4, padValue=128):
        scale = float(bbsize) / float(oriImg.shape[0])

        imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = self.pad_right_down_corner(imageToTest, stride, padValue)
        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # Required shape (1, height, width, channels)

        return input_img


    def pad_right_down_corner(self, img, stride, padValue):
        h = img.shape[0]
        w = img.shape[1]

        pad = 4 * [None]
        pad[0] = 0 # up
        pad[1] = 0 # left
        pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
        pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

        img_padded = img
        pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
        img_padded = np.concatenate((pad_up, img_padded), axis=0)
        pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
        img_padded = np.concatenate((pad_left, img_padded), axis=1)
        pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
        img_padded = np.concatenate((img_padded, pad_down), axis=0)
        pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
        img_padded = np.concatenate((img_padded, pad_right), axis=1)

        return img_padded, pad
