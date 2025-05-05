import numpy as np

class BaseFeature(object):
    next_id = 0
    
    R_cam0_cam1 = None
    t_cam0_cam1 = None

    def __init__(self, new_id=0, optimization_config=None):
        self.id = new_id
        self.observations = dict() 
        self.position = np.zeros(3)
        self.is_initialized = False
        self.optimization_config = optimization_config