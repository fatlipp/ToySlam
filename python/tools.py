import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu
from scipy.spatial.transform import Rotation

def angle_to_mat(angle):
    return Rotation.from_euler('z', angle).as_matrix()

def mat_to_angle(mat):
    return Rotation.from_matrix(mat).as_euler("xyz")[2]

def mat_to_angle_2d(mat):
    m = np.identity(3)
    m[:2, :2] = mat
    return Rotation.from_matrix(m).as_euler("xyz")[2]

def angle_to_mat_2d(angle):
    return Rotation.from_euler('z', angle).as_matrix()[:2,:2]

def state_mat_to_angle(mat):
    return Rotation.from_matrix(mat[:3,:3]).as_euler("xyz")[2]

def transform_cloud(transform, cloud):
    cloud_h = np.vstack([cloud, np.ones((1, cloud.shape[1]))])
    r = transform @ cloud_h
    return r[:3,:]

def convert_2d_rays_to_cloud(landmarks):
    return np.array([[dist * np.cos(angle), dist * np.sin(angle), 0] for dist, angle in landmarks]).T

def normalize_rotation_matrix(R):
    rotation = Rotation.from_matrix(R)
    R_normalized = rotation.as_matrix()
    return R_normalized

def normalize_vec(vec):
    return np.array(vec / np.sqrt(np.sum(vec**2)))

def convert_eucledian_to_radial_2d(lm, pos):
    diffPos = lm[0:2] - pos[0:2]
    ang = np.arctan2(diffPos[1], diffPos[0]) - pos[2]
    return np.array([np.linalg.norm(diffPos), ang])

def convert_radial_to_euclidean_2d(lm, pos):
    x = lm[0] * np.cos(lm[1])
    y = lm[0] * np.sin(lm[1])
    return np.dot(pos, np.array([x, y, 1]))[:2]

def convert_eucledian_to_radial_3d(lm, state):
    diffPos = lm - state[0:3,3]
    ang = state_mat_to_angle(state)
    ang = np.arctan2(diffPos[1], diffPos[0]) - ang
    return np.array([np.linalg.norm(diffPos[:2]), ang])