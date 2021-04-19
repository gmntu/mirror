###############################################################################
### GUI to display hand rom                                                 ###
### Input: Hand rom                                                         ###
### Output: Realtime display of hand rom                                    ###
###############################################################################

import numpy as np
import matplotlib.pyplot as plt


class GuiHand:
    def __init__(self, max_len=50):
        # Use ggplot style for more sophisticated visuals
        plt.style.use('ggplot')
        
        # Line plot for MCP joint of five digits
        self.line_mcp = [[] for i in range(5)] 
        self.line_pip = [[] for i in range(5)]
        self.line_dip = [[] for i in range(5)]
        # To store the joint angles for each digit
        self.y_mcp = [np.zeros(max_len) for i in range(5)]
        self.y_pip = [np.zeros(max_len) for i in range(5)]
        self.y_dip = [np.zeros(max_len) for i in range(5)]
        # x axis value [0, 0.01, ..., 0.99]
        self.x_vec = np.linspace(0,1,max_len+1)[0:-1] 

        # Call to matplotlib to allow dynamic plotting
        plt.ion() 
        
        fig = plt.figure(figsize=(6,3.7))
        ax = fig.add_subplot(111)
        for i in range(5): 
            self.line_mcp[i], = ax.plot(self.x_vec, self.y_mcp[i], '-o', markersize=2)        
            self.line_pip[i], = ax.plot(self.x_vec, self.y_pip[i], '-o', markersize=2)        
            self.line_dip[i], = ax.plot(self.x_vec, self.y_dip[i], '-o', markersize=2)
        
        plt.show()


    def live_plotter(self, data, mode=0, pause_time=0.001):
        # Extract the data into different types
        data_mcp = [data[9],data[12],data[16],data[20],data[24]]
        data_pip = [data[10],data[13],data[17],data[21],data[25]]
        data_dip = [data[10],data[14],data[18],data[22],data[26]] # Note data[10] is thumb IP so considered dip and pip

        ###########
        ### MCP ###
        ###########
        label = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
        for i, d in enumerate(data_mcp):
            self.y_mcp[i][-1] = d
            # Update the y-data
            self.line_mcp[i].set_ydata(self.y_mcp[i])
            if mode==0: 
                self.line_mcp[i].set_label('%s %.2f'%(label[i], d))
            else:
                self.line_mcp[i].set_label('')
            self.y_mcp[i] = np.append(self.y_mcp[i][1:], 0.0)             

        ###########
        ### PIP ###
        ###########
        for i, d in enumerate(data_pip):
            self.y_pip[i][-1] = d
            # Update the y-data
            self.line_pip[i].set_ydata(self.y_pip[i])
            if mode==1: 
                self.line_pip[i].set_label('%s %.2f'%(label[i], d))
            else:
                self.line_pip[i].set_label('')
            self.y_pip[i] = np.append(self.y_pip[i][1:], 0.0)           


        ###########
        ### DIP ###
        ###########
        for i, d in enumerate(data_dip):
            self.y_dip[i][-1] = d
            # Update the y-data
            self.line_dip[i].set_ydata(self.y_dip[i])
            if mode==2: 
                self.line_dip[i].set_label('%s %.2f'%(label[i], d))
            else: 
                self.line_dip[i].set_label('')
            self.y_dip[i] = np.append(self.y_dip[i][1:], 0.0)           

        # Adjust the plot visibility according 
        if mode==0:
            for i in range(5):
                self.line_mcp[i].set_visible(True)
                self.line_pip[i].set_visible(False)
                self.line_dip[i].set_visible(False)
                # Adjust limits if new data goes beyond bounds
                ylim_min = self.line_mcp[i].axes.get_ylim()[0] 
                ylim_max = self.line_mcp[i].axes.get_ylim()[1] 
                if np.min(self.y_mcp[i])<=ylim_min or np.max(self.y_mcp[i])>=ylim_max:
                    plt.ylim([np.min(self.y_mcp[i])-np.std(self.y_mcp[i]),np.max(self.y_mcp[i])+np.std(self.y_mcp[i])])                    
            plt.ylabel('MCP angle (deg)')
        elif mode==1:
            for i in range(5):
                self.line_mcp[i].set_visible(False)
                self.line_pip[i].set_visible(True)
                self.line_dip[i].set_visible(False)   
                # Adjust limits if new data goes beyond bounds
                ylim_min = self.line_pip[i].axes.get_ylim()[0] 
                ylim_max = self.line_pip[i].axes.get_ylim()[1] 
                if np.min(self.y_pip[i])<=ylim_min or np.max(self.y_pip[i])>=ylim_max:
                    plt.ylim([np.min(self.y_pip[i])-np.std(self.y_pip[i]),np.max(self.y_pip[i])+np.std(self.y_pip[i])])                                   
            plt.ylabel('PIP angle (deg)')
        elif mode==2:
            for i in range(5):
                self.line_mcp[i].set_visible(False)
                self.line_pip[i].set_visible(False)
                self.line_dip[i].set_visible(True)   
                # Adjust limits if new data goes beyond bounds
                ylim_min = self.line_dip[i].axes.get_ylim()[0] 
                ylim_max = self.line_dip[i].axes.get_ylim()[1] 
                if np.min(self.y_dip[i])<=ylim_min or np.max(self.y_dip[i])>=ylim_max:
                    plt.ylim([np.min(self.y_dip[i])-np.std(self.y_dip[i]),np.max(self.y_dip[i])+np.std(self.y_dip[i])])                                   
            plt.ylabel('DIP angle (deg)')

        # Replot the legend to update the data
        plt.legend(loc='upper left')
        
        # Pauses the data so the figure/axis can catch up
        plt.pause(pause_time)
        

###############################################################################
### Simple example to test the program                                      ###
###############################################################################
if __name__ == '__main__':
    import time
    import keyboard
    gui = GuiHand(max_len=100)

    mode = 0
    while True:
        data = np.random.randn(27)
        print(data)
        gui.live_plotter(data, mode)

        if keyboard.is_pressed('esc'): # Press escape to end the program
            print('Quitting...')
            break
        if keyboard.is_pressed('m'):
            mode = (mode+1)%3
            print('mode', mode)
            time.sleep(0.3)
