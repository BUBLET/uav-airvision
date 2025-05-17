import numpy as np

class Isometry3d(object):
    """
    Жёсткое преобразование в 3D.
    """
    def __init__(self, R, t):
        self.R = R
        self.t = t

    def matrix(self):
        m = np.identity(4)
        m[:3, :3] = self.R
        m[:3, 3] = self.t
        return m

    def inverse(self):
        return Isometry3d(self.R.T, -self.R.T @ self.t)

    def __mul__(self, T1):
        R = self.R @ T1.R
        t = self.R @ T1.t + self.t
        return Isometry3d(R, t)
    
def to_rotation(q):
    """
    Преобразует кватернион в соответствующую матрицу вращения.
    Здесь применяется формула из
    "Indirect Kalman Filter for 3D Attitude Estimation: A Tutorial for Quaternion Algebra", уравнение (78).
    На вход подаётся кватернион в виде [q1, q2, q3, q4(скалярная часть)].
    """

    q = q / np.linalg.norm(q)
    vec = q[:3]
    w = q[3]

    R = (2*w*w-1)*np.identity(3) - 2*w*skew(vec) + 2*vec[:, None]*vec
    return R

def skew(vec):
    """
    Формирует кососимметричную матрицу из 3-мерного вектора.
    """
    x, y, z = vec
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]])