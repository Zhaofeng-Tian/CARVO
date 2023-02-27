import sys
sys.path.append('C:\\Users\\61602\\Desktop\\Coding\\python')
import numpy as np
class CarParam:
    def __init__(self):
        self.kinematic_constraints=np.array([[-1,-0.6],[1,0.6]])
        self.dynamic_constraints = np.array([[-0.8,-1],[0.8,1]])
        # [Lw,Lf,Lr,W] wheelbase,front suspension, rear suspension, width
        self.body_structure = np.array([0.53,0.25,0.25,0.68])
        self.dt = 0.2
    
    
