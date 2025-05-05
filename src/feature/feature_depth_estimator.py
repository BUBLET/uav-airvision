import numpy as np

class FeatureDepthEstimator:
    def generate_initial_guess(self, T_c1_c2, z1, z2):
        """
        Compute the initial guess of the feature's 3d position using 
        only two views.
        """
        m = T_c1_c2.R @ np.array([*z1, 1.0])
        a = m[:2] - z2*m[2]
        b = z2*T_c1_c2.t[2] - T_c1_c2.t[:2]

        depth = a @ b / (a @ a)
        p = np.array([*z1, 1.0]) * depth
        return p