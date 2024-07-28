import numpy as np
from tools import *

def solveQuadratic(a, b, c, x0, x1):
    discr = b * b - 4 * a * c
    if discr < 0:
        return False
    
    if discr == 0:
        pt = -0.5 * b / a
        return pt, pt
    
    q = -0.5 * (b + np.sqrt(discr)) if b > 0 else -0.5 * (b - np.sqrt(discr))
    x0 = q / a
    x1 = c / q

    if x0 > x1:
        return x1, x0
    
    return x0, x1

def hit_sphere(ray_orig, ray_dir, obj_center, radius):
    oc = obj_center - ray_orig
    a = np.dot(ray_dir, ray_dir)

    tca = np.dot(oc, ray_dir)
    d2 = np.dot(oc, oc) - tca * tca

    if d2 > radius * radius:
        return None

    thc = np.sqrt(radius * radius - d2)
    t0 = tca - thc
    t1 = tca + thc
    
    L = ray_orig - obj_center
    a = np.dot(ray_dir, ray_dir)
    b = 2 * np.dot(ray_dir, L)
    c = np.dot(L, L) - radius * radius

    hit_points = solveQuadratic(a, b, c, t0, t1)
    if hit_points == False:
        return None
    
    t0, t1 = hit_points
    if t0 < 0 and t1 < 0:
        return None
    
    return t1 if t0 < 0 else t0 

def calc_lidar_measurements(state_mat, environment, radius, lidar_fov, lidar_ray_step):
    """
    Inp:
    * state_mat: 4x4 matrix
    * environment: array of obstacles (circles)
    * radius - obstacle radius
    * lidar_fov - field of View, rad.
    * lidar_ray_step - angle between lidar beams, rad.

    Note: a current implementation returns an obstacle pos, instead of a hit point (pt)

    return: list of detected ranges [(range, angle), ...] + [obstacle_id, ...]
    """
    landmarks = []
    landmark_ids = []

    state_pos = state_mat[:3,3]
    state_angle = state_mat_to_angle(state_mat)

    # left bound
    max_angle = state_angle + lidar_fov * 0.5
    # right bound
    min_angle = state_angle - lidar_fov * 0.5

    ray_count = int(lidar_fov / lidar_ray_step)
    for angle in np.linspace(min_angle, max_angle, num = ray_count):
        dir = np.array([np.cos(angle), np.sin(angle), 0])

        closest_p = None
        closest_d = 999999
        closest_id = -1

        for id in range(len(environment)):
            point = environment[id]
            dd = hit_sphere(state_pos, dir, point, radius)

            if dd is not None:
                dir = np.array(point - state_pos)
                dir = dir / np.linalg.norm(dir)

                pt = state_pos + dir * dd

                if closest_p is None:
                    closest_p = point
                    closest_d = np.linalg.norm(state_pos - point)
                    closest_id = id
                else:
                    dist = np.linalg.norm(state_pos - point)
                    if dist < closest_d:
                        closest_p = point
                        closest_d = dist
                        closest_id = id

        if closest_p is not None:
            d, a = convert_eucledian_to_radial_3d(closest_p, state_mat)
            landmarks.append([d, a])
            landmark_ids.append(closest_id)
    
    landmarks = np.array(landmarks)
    landmark_ids = np.array(landmark_ids)

    if landmarks.shape[0] == 0:
        return None, None

    return landmarks, landmark_ids