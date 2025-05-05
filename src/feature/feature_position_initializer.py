import numpy as np
from .base_feature import BaseFeature
from .utils import Isometry3d, to_rotation

class FeaturePositionInitializer:
    def initialize_position(self, cam_states):
        """
        Initialize the feature position from all observations.
        """
        cam_poses = []
        measurements = []
        T_cam1_cam0 = Isometry3d(
            BaseFeature.R_cam0_cam1,
            BaseFeature.t_cam0_cam1).inverse()

        for cam_id, m in self.observations.items():
            if cam_id not in cam_states:
                continue
            measurements.extend([m[:2], m[2:]])
            cam0 = Isometry3d(
                to_rotation(cam_states[cam_id].orientation).T,
                cam_states[cam_id].position)
            cam1 = cam0 * T_cam1_cam0
            cam_poses.extend([cam0, cam1])

        T_c0_w = cam_poses[0]
        cam_poses = [(pose.inverse() * T_c0_w) for pose in cam_poses]

        initial_position = self.generate_initial_guess(
            cam_poses[-2], measurements[0], measurements[-2])
        solution = np.array([*initial_position[:2], 1.0]) / initial_position[2]

        lambd = self.optimization_config.initial_damping
        outer_count = inner_count = 0
        delta_norm = float('inf')
        total_cost = sum(self.cost(pose, solution, meas)
                         for pose, meas in zip(cam_poses, measurements))

        while (outer_count < self.optimization_config.outer_loop_max_iteration
               and delta_norm > self.optimization_config.estimation_precision):
            A = np.zeros((3, 3))
            b = np.zeros(3)
            for pose, meas in zip(cam_poses, measurements):
                J, r, w = self.jacobian(pose, solution, meas)
                if w == 1.0:
                    A += J.T @ J
                    b += J.T @ r
                else:
                    A += w*w * J.T @ J
                    b += w*w * J.T @ r

            is_cost_reduced = False
            while (inner_count < self.optimization_config.inner_loop_max_iteration
                   and not is_cost_reduced):
                delta = np.linalg.solve(A + lambd*np.eye(3), b)
                new_solution = solution - delta
                delta_norm = np.linalg.norm(delta)

                new_cost = sum(self.cost(pose, new_solution, meas)
                               for pose, meas in zip(cam_poses, measurements))
                if new_cost < total_cost:
                    is_cost_reduced = True
                    solution = new_solution
                    total_cost = new_cost
                    lambd = max(lambd/10., 1e-10)
                else:
                    lambd = min(lambd*10., 1e12)
                inner_count += 1
            outer_count += 1

        final_position = np.array([*solution[:2], 1.0]) / solution[2]
        is_valid = all((pose.R @ final_position + pose.t)[2] > 0
                       for pose in cam_poses)
        self.position = T_c0_w.R @ final_position + T_c0_w.t
        self.is_initialized = is_valid
        return is_valid