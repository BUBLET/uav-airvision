import numpy as np

def skew(vec):
    x, y, z = vec
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]])

def select(data, selectors):
    return [d for d, s in zip(data, selectors) if s]