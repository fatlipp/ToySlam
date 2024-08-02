import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu
from scipy.spatial.transform import Rotation

def mat_to_angle(mat):
    return Rotation.from_matrix(mat).as_euler("xyz")[2]

def mat_to_angle_2d(mat):
    m = np.identity(3)
    m[:2, :2] = mat
    return mat_to_angle(m)

def angle_to_mat_2d(angle):
    return Rotation.from_euler('z', angle).as_matrix()[:2,:2]

def transform_cloud(transform, cloud):
    cloud_h = np.vstack([cloud, np.ones((1, cloud.shape[1]))])
    r = transform @ cloud_h
    return r[:2,:]

def convert_2d_rays_to_cloud(landmarks):
    return np.array([[dist * np.cos(angle), dist * np.sin(angle)] for dist, angle in landmarks]).T

def eucledian_to_radial_2d(lm):
    ang = np.arctan2(lm[1], lm[0])
    return np.array([np.linalg.norm(lm[:2]), ang])

def convert_eucledian_to_radial_2d(lm, pos):
    diffPos = lm - pos[:2,2]
    ang = np.arctan2(diffPos[1], diffPos[0]) - mat_to_angle_2d(pos[:2,:2])
    return np.array([np.linalg.norm(diffPos), ang])

def radial_to_euclidean_2d(lm):
    x = lm[0] * np.cos(lm[1])
    y = lm[0] * np.sin(lm[1])
    return np.array([x, y])

def convert_radial_to_euclidean_2d(lm, pos):
    x = lm[0] * np.cos(lm[1])
    y = lm[0] * np.sin(lm[1])
    return np.dot(pos, np.array([x, y, 1]))[:2]