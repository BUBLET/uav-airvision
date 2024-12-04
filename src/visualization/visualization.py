# src/visualization/visualization.py

import open3d as o3d
import numpy as np

class Visualizer3D:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=800, height=600)  # Установка размера окна
        
        # Геометрия для траектории
        self.trajectory_line = o3d.geometry.LineSet()
        self.trajectory_line.points = o3d.utility.Vector3dVector()
        self.trajectory_line.lines = o3d.utility.Vector2iVector()
        self.trajectory_line.paint_uniform_color([0, 0, 1])  # Синий цвет

        # Облако точек карты
        self.point_cloud = o3d.geometry.PointCloud()

        self.vis.add_geometry(self.trajectory_line)
        self.vis.add_geometry(self.point_cloud)

    def update_trajectory(self, trajectory):
        if len(trajectory) < 2:
            print("Trajectory is too short.")
            return

        points = np.array(trajectory)
        lines = [[i, i + 1] for i in range(len(points) - 1)]

        self.trajectory_line.points = o3d.utility.Vector3dVector(points)
        self.trajectory_line.lines = o3d.utility.Vector2iVector(lines)
        self.vis.update_geometry(self.trajectory_line)

    def update_map_points(self, map_points):
        if len(map_points) == 0:
            print("Map points are empty.")
            return

        self.point_cloud.points = o3d.utility.Vector3dVector(map_points)
        self.point_cloud.paint_uniform_color([1, 0, 0])  # Красный цвет
        self.vis.update_geometry(self.point_cloud)

    def render(self):
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.reset_view_point(True)  # Сбрасывает видовую точку
