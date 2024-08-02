import numpy as np
from tools import radial_to_euclidean_2d, eucledian_to_radial_2d

def add_to_graph(graph, state, landmarks, landmark_ids, noise, odom = None):
    noisy_lms = {}
    for i in range(len(landmarks)):
        lm_local = radial_to_euclidean_2d(landmarks[i])
        lm_local[0] += np.random.normal(0, noise[0, 0])
        lm_local[1] += np.random.normal(0, noise[1, 1])
        lm_glob = (state @ np.array([lm_local[0], lm_local[1], 1]))[:2]
        graph.add_landmark(landmark_ids[i], lm_glob)
        noisy_lms[landmark_ids[i]] = eucledian_to_radial_2d(lm_local)
    graph.add_pose(state, noisy_lms, odom)

    return graph.get_last_pos_id()

def motion_model(state, motion):
    x = state @ motion
    return x