import numpy as np


def angle_to_x_deg(row):
    dx = row['X_left'] - row['X_right']
    dy = row['Y_left'] - row['Y_right']
    norm = np.sqrt(dx**2 + dy**2)
    ux = dx / norm
    uy = dy / norm
    rad = np.arccos(np.clip(np.dot((ux, uy), (1.0, 0.0)), -1.0, 1.0))
    return rad * 180.0 / np.pi
