import open3d as o3d
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Visualizer3D:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=800, height=600)  # Создаём окно

        self.trajectory_line = o3d.geometry.LineSet()
        self.trajectory_line.points = o3d.utility.Vector3dVector()
        self.trajectory_line.lines = o3d.utility.Vector2iVector()
        self.trajectory_line.paint_uniform_color([0, 0, 1])  # Синий
        self.vis.add_geometry(self.trajectory_line)

        self.point_cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.point_cloud)

        self.trajectory_points = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.trajectory_points)

    def update_trajectory(self, trajectory):
        if len(trajectory) == 0:
            logger.warning("[VIS] No trajectory")
            return

        points = np.asarray(trajectory)

        if len(points) >= 2:
            lines = [[i, i + 1] for i in range(len(points) - 1)]
            self.trajectory_line.points = o3d.utility.Vector3dVector(points)
            self.trajectory_line.lines = o3d.utility.Vector2iVector(lines)
            self.vis.update_geometry(self.trajectory_line)

        self.trajectory_points.points = o3d.utility.Vector3dVector(points)
        colors = np.tile(np.array([[0, 1, 0]]), (len(points), 1))
        self.trajectory_points.colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(self.trajectory_points)

    def update_map_points(self, map_points):
        if len(map_points) == 0:
            logger.warning("[VIS] Map points are empty.")
            return


        self.point_cloud.points = o3d.utility.Vector3dVector(map_points)
        self.point_cloud.paint_uniform_color([1, 0, 0])  
        self.vis.update_geometry(self.point_cloud)

    def render(self):
        self.vis.poll_events()
        self.vis.update_renderer()
        #self.vis.reset_view_point(True)  # Сбрасывает видовую точку

    def close(self):
        self.vis.destroy_window()
