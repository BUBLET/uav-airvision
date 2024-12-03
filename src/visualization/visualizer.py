import open3d as o3d
import numpy as np
import threading
import time

class Visualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Visual Odometry', width=800, height=600)
        self.is_initialized = False

        # Geometry elements
        self.trajectory_line_set = o3d.geometry.LineSet()
        self.point_cloud = o3d.geometry.PointCloud()
        self.camera_frame = o3d.geometry.TriangleMesh()
        self.lock = threading.Lock()

        # Data storage
        self.trajectory_points = []
        self.trajectory_lines = []
        self.map_points = []
        self.line_colors = []

        # Coordinate frames
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # Start visualization in a separate thread
        self.thread = threading.Thread(target=self.visualization_thread)
        self.thread.daemon = True
        self.thread.start()

    def visualization_thread(self):
        self.vis.add_geometry(self.coordinate_frame)
        self.vis.add_geometry(self.point_cloud)
        self.vis.add_geometry(self.trajectory_line_set)

        while True:
            with self.lock:
                self.vis.update_geometry(self.coordinate_frame)
                self.vis.update_geometry(self.point_cloud)
                self.vis.update_geometry(self.trajectory_line_set)
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.01)

    def add_camera_pose(self, pose, is_keyframe=False):
        """
        Add a camera pose to the trajectory.

        Parameters:
        - pose: 4x4 numpy array representing the camera pose.
        - is_keyframe: Boolean indicating if this pose is a keyframe.
        """
        with self.lock:
            position = pose[:3, 3]
            self.trajectory_points.append(position)

            idx = len(self.trajectory_points) - 1
            if idx > 0:
                self.trajectory_lines.append([idx - 1, idx])
                if is_keyframe:
                    color = [1.0, 0.0, 0.0]  # Red for keyframes
                else:
                    color = [1.0, 1.0, 1.0]  # White for normal frames
                self.line_colors.append(color)

            # Update the LineSet
            points = o3d.utility.Vector3dVector(np.array(self.trajectory_points))
            lines = o3d.utility.Vector2iVector(np.array(self.trajectory_lines))
            colors = o3d.utility.Vector3dVector(np.array(self.line_colors))

            self.trajectory_line_set.points = points
            self.trajectory_line_set.lines = lines
            self.trajectory_line_set.colors = colors

    def add_map_points(self, points, colors=None):
        """
        Add map points to the point cloud.

        Parameters:
        - points: Nx3 numpy array of 3D points.
        - colors: Optional Nx3 numpy array of RGB colors.
        """
        with self.lock:
            self.map_points.extend(points)
            self.point_cloud.points = o3d.utility.Vector3dVector(np.array(self.map_points))
            if colors is not None:
                self.point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
            else:
                # Default color
                self.point_cloud.paint_uniform_color([0.0, 1.0, 0.0])  # Green

    def reset(self):
        """
        Reset the visualizer data.
        """
        with self.lock:
            self.trajectory_points = []
            self.trajectory_lines = []
            self.map_points = []
            self.line_colors = []
            self.point_cloud.clear()
            self.trajectory_line_set.clear()
