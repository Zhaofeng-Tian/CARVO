import math
from math import atan2, cos, sin, tan, pi, sqrt
import sys
sys.path.append('C:\\Users\\61602\\Desktop\\Coding\\python')
from mobile_robot_simulator.world.world import World
from mobile_robot_simulator.world.param import CarParam
from mobile_robot_simulator.world.utils import plot_car
from collections import deque
import time

import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import random
from skimage.draw import disk, polygon

class CarRobot:
    def __init__(self, world, state, param):
        # Initiate 
        self.world = world
        self.state = state # [x,y,theta,v,phi]
        self.vertices = None
        # Global variables
        self.sequence = deque(maxlen=100) # History of states
        self.control_signal = [0,0] # Control signal 
        # self.execut_signal = [0,0] # Actually executed signal by actuators
        # Constant parameters
        self.dt = param.dt
        self.v_limits = param.kinematic_constraints
        self.a_limits = param.dynamic_constraints
        self.body_structure = param.body_structure

        self.x_fd = self.state[0] + (3*self.body_structure[0]+3*self.body_structure[1]-self.body_structure[2])*cos(self.state[2]/4)
        self.y_fd = self.state[1] + (3*self.body_structure[0]+3*self.body_structure[1]-self.body_structure[2])*sin(self.state[2]/4)
        self.x_fd = self.state[0] + (self.body_structure[0]+self.body_structure[1]-3*self.body_structure[2])*cos(self.state[2]/4)
        self.y_fd = self.state[1] + (self.body_structure[0]+self.body_structure[1]-3*self.body_structure[2])*sin(self.state[2]/4)
        self.disc_r = 0.5*sqrt((np.sum(self.body_structure[:4])**2/4 + self.body_structure[3]**2))
    
    def move_base(self, u):
        self. control_signal = [u[0], u[1]]
    def clip_control_signal(self):
        """ 
        Clif the control signal so that the real executed 
        signal is within acceleration limits.
        """
        s = self.state.copy(); dt = self.dt; clipped = [0,0]
        if self.control_signal[0]-s[3] >= 0: # Throttle
            clipped[0] = min(self.control_signal[0]-s[3], self.a_limits[1][0] * dt, self.v_limits[1][0]-s[3]) 
        else:    # Brake
            clipped[0] = max(self.control_signal[0]-s[3], self.a_limits[0][0] * dt, self.v_limits[0][0]-s[3]) 

        if self.control_signal[1]-s[4] >= 0: # steer
            clipped[1] = min(self.control_signal[1]-s[4], self.a_limits[1][1] * dt, self.v_limits[1][1]-s[4]) 
        else:    # Brake
            clipped[1] = max(self.control_signal[1]-s[4], self.a_limits[0][1] * dt, self.v_limits[0][1]-s[4])
        return clipped # clipped = [delta v, delta w]
    
    def state_update(self):
        # First store the old state
        old_state = self.state.copy()
        self.sequence.append(old_state)
        # Update actuator state
        clipped = self.clip_control_signal()
        self.state[3:] += clipped
        # Update pose state
        self.state[:3] = old_state[:3] + np.array([cos(old_state[2]),sin(old_state[2]),tan(self.state[4])/self.body_structure[0]])*self.state[3]*self.dt
        self.calc_rect_vertices()
    def calc_rect_vertices(self):
        """ First calculated four vertices' coordinates (x, y),
            then coverted them into indices on the map (ir, ic),
            where row index = y /resolution,
                  col index = x /resolution.
        """
        x = self.state[0]
        y = self.state[1]
        theta = self.state[2]
        fdiag = sqrt((self.body_structure[0]+self.body_structure[1])**2 
                           + (self.body_structure[3]/2)**2)
        rdiag = sqrt((self.body_structure[2])**2 
                           + (self.body_structure[3]/2)**2)
        temp1 = atan2(self.body_structure[3]/2,self.body_structure[0]+self.body_structure[1])
        temp2 = atan2(self.body_structure[3]/2,self.body_structure[2])
        #print(temp)

        ph1= theta + temp1
        ph2 = theta + pi-temp2
        ph3 = theta + pi + temp2
        ph4 = theta + 2*pi-temp1
        #print([ph1,ph2,ph3,ph4])
        ph1 = atan2(sin(ph1),cos(ph1))
        ph2 = atan2(sin(ph2),cos(ph2))
        ph3 = atan2(sin(ph3),cos(ph3))
        ph4 = atan2(sin(ph4),cos(ph4))
        #print([ph1,ph2,ph3,ph4])

        x1 = x + cos(ph1)*fdiag
        y1 = y + sin(ph1)*fdiag
        x2 = x + cos(ph2)*rdiag
        y2 = y + sin(ph2)*rdiag
        x3 = x + cos(ph3)*rdiag
        y3 = y + sin(ph3)*rdiag
        x4 = x + cos(ph4)*fdiag
        y4 = y + sin(ph4)*fdiag
        # self.point_out_map = False
        for i in [x1,y1,x2,y2,x3,y3,x4,y4]:
            if i <= 0.0 or i>= 10.0:
                # print(" Point is out, in calc_rect_vertices check!")
                self.point_out_map = True
        r1 = round(y1/self.world.resolution)
        c1 = round(x1/self.world.resolution)
        r2 = round(y2/self.world.resolution)
        c2 = round(x2/self.world.resolution)
        r3 = round(y3/self.world.resolution)
        c3 = round(x3/self.world.resolution)
        r4 = round(y4/self.world.resolution)
        c4 = round(x4/self.world.resolution)
        # print("Got rectangular vertices: ") # indices / not real coords
        # print([(r1,c1), (r2,c2),(r3,c3),(r4,c4)])
        self.vertices = [(r1,c1), (r2,c2),(r3,c3),(r4,c4)]
        # return [(r1,c1), (r2,c2),(r3,c3),(r4,c4)]

    def fill_rect_body(self, pts, canvas):            
        """ fill robot body rectangular """
        rr = []
        cc = []
        # p (x, y)
        for p in pts:
            rr.append(p[0])
            cc.append(p[1])
        # rr = np.array(rr)
        # cc = np.array(cc)
        canvas[polygon(rr,cc)] = 1


if __name__ == "__main__":
    param = CarParam()
    world = World(size_x = 20, size_y = 20, resolution = 0.01)
    robot = CarRobot(world,np.array([5.,5.,0.,0.,0.]),param)
    robots = []
    for i in range(10):
        robot = CarRobot(world,np.array([5+0.8*i,5+0.8*i,0.,0.,0.]),param)
        robots.append(robot)

    for i in range(int(60/robot.dt)):

        map = world.map.copy()
        start = time.time()
        for robot in robots:
            start_time = time.time()
            robot.move_base([10,10])
            print(robot.control_signal)
            robot.state_update()
            # print(robot.state)
            print(robot.vertices)
            robot.fill_rect_body(robot.vertices,map)
            end_time = time.time()
            print("Time costing for one loop: ", end_time-start_time)
        end = time.time()
        print("total_time: ", end-start)

        plt.clf()
        plt.imshow(map, origin='lower', cmap='gray')
        plt.draw()
        plt.pause(0.01)

    print(len(robot.sequence))
