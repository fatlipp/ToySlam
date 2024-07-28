import numpy as np
from tools import * 
import itertools
import time
from graph.graph2d import *
from graph.edges2d import EdgeLidarVirtual2d

STATE_SIZE = 3 # x, y, yaw

class GraphOptimizer:
    def __init__(self, graph) -> None:
        self.graph = graph
        self.lidar_edges = []
        self.on_step_cb = None

    def set_on_step_cb(self, cb):
        self.on_step_cb = cb

    def construct(self, graph):
        self.lidar_edges = []

        def get_pairs(obs, lm_id):
            pairs = np.array(list(itertools.combinations(range(len(obs)), 2)))
            res = set()
            for p in pairs:
                if abs(obs[p[0]] - obs[p[1]]) > 5:
                    continue
                res.add((obs[p[0]], obs[p[1]], lm_id))
            return res

        ss = set()

        for lm_id in graph.observers_lm:
            count = len(graph.observers_lm[lm_id])
            if count > 1:
                pairs = get_pairs(graph.observers_lm[lm_id], lm_id)
                for pair in pairs:
                    pos_id_1, pos_id_2, _ = pair
                    s = str(pos_id_1) + ":" + str(pos_id_2) + ":" + str(lm_id)
                    if s in ss:
                        print("Duplicate edge = ", s)
                        continue
                    ss.add(s)
                    e = EdgeLidarVirtual2d(pos_id_1, pos_id_2, lm_id, graph.landmark_edges[pos_id_1][lm_id].information)
                    self.lidar_edges.append(e)

        self.lidar_edges = np.array(self.lidar_edges)
        print("lidar_edges: ", self.lidar_edges.shape)

    def optimize(self, iterations, lr = 1.0):
        if len(self.lidar_edges) == 0:
            return
        
        prev_err = -1
        penalty = 0

        for iter in range(iterations):
            t_start = time.time()
            H, b, err = self.calculate_H_b()

            if prev_err == -1:
                prev_err = err
            else:
                if err > prev_err:
                    penalty += 1
                    if penalty > 5:
                        print("Opt is getting worse: ", prev_err, " -> ", err)
                        break
                else:
                    penalty = 0
                prev_err = err

            t_end = time.time()
            duration_H = t_end - t_start

            if H is None or b is None:
                print("Opt result is NONE...")
                break

            if np.linalg.det(H) == 0:
                print("det == 0!!! break..\n H:\n", H)
                break
            t_start = time.time()
            dx = np.linalg.solve(H, -b) * lr
            t_end = time.time()
            duration_solve = t_end - t_start

            def update_pos(pos, delta):
                theta = mat_to_angle_2d(pos[:2, :2]) + delta[2]
                c, s = np.cos(theta), np.sin(theta)
                pos[:2, :2] = np.array([[c, -s], [s, c]])
                pos[0, 2] += delta[0]
                pos[1, 2] += delta[1]
                return pos

            t_start = time.time()
            pos_count = len(self.graph.positions)
            for pos_id in range(pos_count):
                index = pos_id * STATE_SIZE
                dd = dx[index:index + STATE_SIZE]
                self.graph.positions[pos_id].measurement = \
                    update_pos(self.graph.positions[pos_id].measurement, dd)

            t_end = time.time()
            duration_upd = t_end - t_start

            print('Duration H: {} sec, Solve: {} sec, Upd: {} sec, Error: {}'\
                  .format(duration_H, duration_solve, duration_upd, err))
            
            if self.on_step_cb is not None:
                self.on_step_cb(iter)

            if np.linalg.norm(dx) < 0.001:
                print("CONVERGED")
                break

    def calculate_H_b(self):
        pos_count = len(self.graph.positions)
        SS = pos_count * STATE_SIZE
        H = np.zeros((SS, SS))
        b = np.zeros((SS))

        err = 0

        for edge in self.lidar_edges:
            pos_id_1 = edge.pos_id_1
            pos_id_2 = edge.pos_id_2

            e, A, B = edge.calc_error(self.graph)
            err += np.linalg.norm(e)

            index1 = pos_id_1 * STATE_SIZE
            index2 = pos_id_2 * STATE_SIZE

            INF = edge.information
            H[index1:index1 + STATE_SIZE, index1:index1 + STATE_SIZE] += (A.T @ INF @ A)
            H[index1:index1 + STATE_SIZE, index2:index2 + STATE_SIZE] += (A.T @ INF @ B)
            H[index2:index2 + STATE_SIZE, index1:index1 + STATE_SIZE] += (B.T @ INF @ A)
            H[index2:index2 + STATE_SIZE, index2:index2 + STATE_SIZE] += (B.T @ INF @ B)
            b[index1:index1 + STATE_SIZE] += A.T @ INF @ e
            b[index2:index2 + STATE_SIZE] += B.T @ INF @ e

        for edge in self.graph.odometry_edges:
            pos_id_1 = edge.pos_id_1
            pos_id_2 = edge.pos_id_2

            e, A, B = edge.calc_error(self.graph)
            err += np.linalg.norm(e)

            index1 = pos_id_1 * STATE_SIZE
            index2 = pos_id_2 * STATE_SIZE

            INF = edge.information
            H[index1:index1 + STATE_SIZE, index1:index1 + STATE_SIZE] += (A.T @ INF @ A)
            H[index1:index1 + STATE_SIZE, index2:index2 + STATE_SIZE] += (A.T @ INF @ B)
            H[index2:index2 + STATE_SIZE, index1:index1 + STATE_SIZE] += (B.T @ INF @ A)
            H[index2:index2 + STATE_SIZE, index2:index2 + STATE_SIZE] += (B.T @ INF @ B)
            b[index1:index1 + STATE_SIZE] += A.T @ INF @ e
            b[index2:index2 + STATE_SIZE] += B.T @ INF @ e

        for fixed_pos_id in self.graph.fixed_positions:
            index = fixed_pos_id * STATE_SIZE
            H[index:index + STATE_SIZE, index:index + STATE_SIZE] += np.eye(STATE_SIZE) * 1e10
            b[index:index + STATE_SIZE] = np.zeros(STATE_SIZE)

        return H, b, err