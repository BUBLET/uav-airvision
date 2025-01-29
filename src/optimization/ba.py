import logging
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from typing import Tuple

class BundleAdjustment:

    def __init__(
        self,
        camera_matrix: np.ndarray,
        ftol: float = 1e-4,
        xtol: float = 1e-4,
        gtol: float = 1e-4,
        max_nfev: int = 50  
    ):
        """
        Инициализирует объект BundleAdjustment.

        """
        self.logger = logging.getLogger(__name__)
        self.metrics_logger = logging.getLogger("metrics_logger")

        self.camera_matrix = camera_matrix
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.max_nfev = max_nfev

    def _rotate(
            self, 
            points: np.ndarray, 
            rot_vecs: np.ndarray
    ) -> np.ndarray:
        """
        Вращает 3D-точки с помощью векторов вращения (формула Родрига).

        """

        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]

        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)  
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        dot = np.sum(points * v, axis=1)[:, np.newaxis]

        cross = np.cross(v, points)

        # Формула Родрига:
        # R(p) = p*cosθ + (v×p)*sinθ + v*(v·p)*(1 - cosθ)
        return cos_theta * points + sin_theta * cross + dot * (1 - cos_theta) * v

    def _project(
            self, 
            points: np.ndarray, 
            camera_params: np.ndarray
    ) -> np.ndarray:
        """
        Проецирует 3D-точки в 2D, используя параметры камеры (rvec, tvec).

        """

        points_rotated = self._rotate(points, camera_params[:, :3])

        points_translated = points_rotated + camera_params[:, 3:6]

        points_proj = points_translated @ self.camera_matrix.T
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]

        return points_proj

    def _fun(
        self,
        params: np.ndarray,
        n_cameras: int,
        n_points: int,
        camera_indices: np.ndarray,
        point_indices: np.ndarray,
        points_2d: np.ndarray
    ) -> np.ndarray:
        """
        Вычисляет вектор ошибок между проектируемыми точками и наблюдаемыми.

        """
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))

        points_proj = self._project(points_3d[point_indices], camera_params[camera_indices])

        residuals = (points_proj - points_2d).ravel()
        return residuals

    def _bundle_adjustment_sparsity(
        self,
        n_cameras: int,
        n_points: int,
        camera_indices: np.ndarray,
        point_indices: np.ndarray
    ) -> lil_matrix:
        """
        Создает разреженную матрицу Якоби для оптимизации (структуру ненулевых элементов).

        """
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

    def run_bundle_adjustment(
        self,
        camera_params: np.ndarray,
        points_3d: np.ndarray,
        camera_indices: np.ndarray,
        point_indices: np.ndarray,
        points_2d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Запускает процедуру оптимизации Bundle Adjustment.

        """
        n_cameras = camera_params.shape[0]
        n_points = points_3d.shape[0]

        x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

        f0 = self._fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
        initial_cost = 0.5 * np.sum(f0 ** 2)
        self.logger.info(f"Начальная стоимость: {initial_cost:.4f}")

        A = self._bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

        res = least_squares(
            self._fun,
            x0,
            jac_sparsity=A,
            verbose=0,
            x_scale='jac',
            ftol=self.ftol,
            xtol=self.xtol,
            gtol=self.gtol,
            max_nfev=self.max_nfev,
            method='trf',
            loss='huber',  
            args=(n_cameras, n_points, camera_indices, point_indices, points_2d)
        )

        final_cost = 0.5 * np.sum(res.fun ** 2)
        self.logger.info(f"Конечная стоимость: {final_cost:.4f}")
        self.logger.info(f"Оптимизация завершена: {res.success}, сообщение: {res.message}")

        self.metrics_logger = logging.getLogger("metrics_logger")
        self.metrics_logger.info(
            f"[BA] Initial cost={initial_cost:.2f}, Final cost={final_cost:.2f}, Iterations={res.nfev}"
        )

        optimized_camera_params = res.x[:n_cameras * 6].reshape((n_cameras, 6))
        optimized_points_3d = res.x[n_cameras * 6:].reshape((n_points, 3))

        return optimized_camera_params, optimized_points_3d
