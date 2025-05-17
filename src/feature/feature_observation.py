import numpy as np

class FeatureObservation:
    def cost(self, T_c0_ci, x, z):
        """
        Вычисляет стоимость (ошибку) наблюдений камеры
        """
        alpha, beta, rho = x
        h = T_c0_ci.R @ np.array([alpha, beta, 1.0]) + rho * T_c0_ci.t
        z_hat = h[:2] / h[2]
        e = ((z_hat - z)**2).sum()
        return e

    def jacobian(self, T_c0_ci, x, z):
        """
        Вычисляет якобиан наблюдения камеры
        """
        alpha, beta, rho = x
        h = T_c0_ci.R @ np.array([alpha, beta, 1.0]) + rho * T_c0_ci.t
        h1, h2, h3 = h

        W = np.zeros((3, 3))
        W[:, :2] = T_c0_ci.R[:, :2]
        W[:, 2] = T_c0_ci.t

        J = np.zeros((2, 3))
        J[0] = W[0]/h3 - W[2]*h1/(h3*h3)
        J[1] = W[1]/h3 - W[2]*h2/(h3*h3)

        z_hat = np.array([h1/h3, h2/h3])
        r = z_hat - z

        e = np.linalg.norm(r)
        if e <= self.optimization_config.huber_epsilon:
            w = 1.0
        else:
            w = self.optimization_config.huber_epsilon / (2*e)

        return J, r, w
