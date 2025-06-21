import pickle

import numpy as np


def init_point_cloud(x_min, y_min, z_min, n=10000, bleed=0.1):
    x_lim = x_min * (1 + bleed)
    x = np.random.uniform(x_lim,  -x_lim, (n, 1))
    y_lim = y_min * (1 + bleed)
    y = np.random.uniform(y_lim, -y_lim, (n, 1))
    z_lim = z_min * (1 + bleed)
    z = np.random.uniform(z_lim, -z_lim, (n, 1))

    points = np.concatenate([x, y, z], axis=1)
    return points


def load_point_cloud(file_path):
    with open(file_path, 'rb') as f:
        points = pickle.load(f)
    return points

