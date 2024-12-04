import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import logging

logger = logging.getLogger(__name__)

class BundleAdjustment:
    def __init__(self, camera_matrix):
        self.camera_matrix = camera_matrix

    def rotate(self, points, rot_vecs):
        """Вращает точки с помощью векторов вращения (формула Родрига)."""
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cross = np.cross(v, points)

        return cos_theta * points + sin_theta * cross + dot * (1 - cos_theta) * v

    def project(self, points, camera_params):
        """Проецирует 3D точки на 2D плоскость изображения."""
        points_rotated = self.rotate(points, camera_params[:, :3])
        points_translated = points_rotated + camera_params[:, 3:6]

        points_proj = points_translated @ self.camera_matrix.T
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]

        return points_proj

    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        """Вычисляет вектор резидуалов."""
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))

        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])

        residuals = (points_proj - points_2d).ravel()
        return residuals

    def bundle_adjustment_sparsity(self, n_cameras, n_points, camera_indices, point_indices):
        """Создает разреженную матрицу Якоби для оптимизации."""
        m = camera_indices.size * 2
        n = n_cameras * 6 + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

        return A

    def run_bundle_adjustment(self, camera_params, points_3d, camera_indices, point_indices, points_2d):
        """Запускает процедуру оптимизации Bundle Adjustment."""
        n_cameras = camera_params.shape[0]
        n_points = points_3d.shape[0]

        x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
        f0 = self.fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
        logger.info(f"Начальная стоимость: {0.5 * np.sum(f0 ** 2)}")

        A = self.bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

        res = least_squares(
            self.fun,
            x0,
            jac_sparsity=A,
            verbose=2,
            x_scale='jac',
            ftol=1e-2,
            max_nfev=5,
            method='trf',
            args=(n_cameras, n_points, camera_indices, point_indices, points_2d)
        )

        logger.info(f"Конечная стоимость: {0.5 * np.sum(res.fun ** 2)}")
        logger.info(f"Оптимизация завершена: {res.success}, сообщение: {res.message}")

        # Обновляем параметры камер и 3D-точек
        optimized_camera_params = res.x[:n_cameras * 6].reshape((n_cameras, 6))
        optimized_points_3d = res.x[n_cameras * 6:].reshape((n_points, 3))

        return optimized_camera_params, optimized_points_3d
