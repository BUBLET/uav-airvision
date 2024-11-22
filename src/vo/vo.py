import open3d as o3d
import numpy as np

# Генерация тестовых данных
def generate_test_data():
    # Траектория из VO
    vo_trajectory = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 1, 0],
        [3, 1, 1],
        [4, 2, 1],
        [5, 2, 2]
    ])

    # Ground truth траектория
    ground_truth_trajectory = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 1, 0],
        [3, 1, 1],
        [4, 2, 1],
        [5, 2, 2]
    ])

    # Ключевые кадры (например, каждый второй кадр)
    keyframes = [0, 2, 4]

    # Картографические точки
    map_points = np.random.rand(100, 3) * 5  # 100 точек в пределах [0, 5]

    # Новые триангулированные точки
    new_map_points = np.random.rand(10, 3) * 5  # 10 новых точек в пределах [0, 5]

    return vo_trajectory, ground_truth_trajectory, keyframes, map_points, new_map_points

# Визуализация данных
def visualize_trajectory(vo_trajectory, ground_truth_trajectory, keyframes, map_points, new_map_points):
    # Создание линий для траекторий
    vo_line_set = o3d.geometry.LineSet()
    vo_line_set.points = o3d.utility.Vector3dVector(vo_trajectory)
    vo_line_set.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(vo_trajectory) - 1)])
    vo_line_set.colors = o3d.utility.Vector3dVector([[1, 1, 1] for _ in range(len(vo_trajectory) - 1)])

    ground_truth_line_set = o3d.geometry.LineSet()
    ground_truth_line_set.points = o3d.utility.Vector3dVector(ground_truth_trajectory)
    ground_truth_line_set.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(ground_truth_trajectory) - 1)])
    ground_truth_line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(ground_truth_trajectory) - 1)])

    # Создание точек для картографических точек
    map_point_cloud = o3d.geometry.PointCloud()
    map_point_cloud.points = o3d.utility.Vector3dVector(map_points)
    map_point_cloud.colors = o3d.utility.Vector3dVector([[0, 0, 1] for _ in range(len(map_points))])

    new_map_point_cloud = o3d.geometry.PointCloud()
    new_map_point_cloud.points = o3d.utility.Vector3dVector(new_map_points)
    new_map_point_cloud.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(new_map_points))])

    # Создание маркеров для ключевых кадров
    keyframe_spheres = []
    for idx in keyframes:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere.translate(vo_trajectory[idx])
        sphere.paint_uniform_color([1, 0, 0])
        keyframe_spheres.append(sphere)

    # Визуализация
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(vo_line_set)
    vis.add_geometry(ground_truth_line_set)
    vis.add_geometry(map_point_cloud)
    vis.add_geometry(new_map_point_cloud)
    for sphere in keyframe_spheres:
        vis.add_geometry(sphere)

    vis.run()
    vis.destroy_window()

# Генерация тестовых данных
vo_trajectory, ground_truth_trajectory, keyframes, map_points, new_map_points = generate_test_data()

# Визуализация данных
visualize_trajectory(vo_trajectory, ground_truth_trajectory, keyframes, map_points, new_map_points)
