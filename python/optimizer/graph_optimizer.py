import numpy as np
from tools import * 
import time
from optimizer.opt_graph import *
from scipy.optimize import least_squares
from scipy.linalg import solve

STATE_SIZE = 3 # x, y, yaw
LM_SIZE = 2

class GraphOptimizer:
    def __init__(self, graph) -> None:
        self.graph = graph
        self.on_step_cb = None

    def set_on_step_cb(self, cb):
        self.on_step_cb = cb

    def optimize(self, iterations, lr = 1.0):
        prev_err = -1
        penalty = 0

        lambda_val = 1e-3
        lambda_max = 1e1  
        lambda_min = 1e-6  
        lambda_factor = 1.1

        def update_lambda(residuals_are_increasing, lambda_val, lambda_factor):
            if residuals_are_increasing:
                return min(lambda_val * lambda_factor, lambda_max)
            return max(lambda_val / lambda_factor, lambda_min)

        for iter in range(iterations):
            t_start = time.time()
            H, b, err = self.calculate_H_b()

            if H is None or b is None:
                print("Opt result is NONE...")
                break
            
            lambda_val = update_lambda(prev_err > -1 and err > prev_err, lambda_val, lambda_factor)
            H_regularized = H + lambda_val * np.eye(H.shape[0])

            prev_err = err
            
            if prev_err != -1:
                if err > prev_err:
                    penalty += 1
                    if penalty > 2:
                        print("Opt is getting worse: ", prev_err, " -> ", err)
                        break
                else:
                    penalty = 0

            t_end = time.time()
            duration_H = t_end - t_start

            if np.linalg.cond(H_regularized) > 1 / np.finfo(H_regularized.dtype).eps:
                print("det == 0!!! break..\n H:\n", H_regularized)
                break
            t_start = time.time()
            dx = np.linalg.lstsq(H_regularized, -b)[0]
            # dx = solve(H_regularized, -b)
            dx *= lr
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
                self.graph.positions[pos_id] = \
                    update_pos(self.graph.positions[pos_id], dd)

            for lm_id in self.graph.landmarks:
                index = pos_count * STATE_SIZE + lm_id * LM_SIZE
                dd = dx[index:index + LM_SIZE]
                self.graph.landmarks[lm_id] += dd

            t_end = time.time()
            duration_upd = t_end - t_start

            print('Duration H: {} sec, Solve: {} sec, Upd: {} sec, Error: {}'\
                  .format(duration_H, duration_solve, duration_upd, err))
            
            if self.on_step_cb is not None:
                if not self.on_step_cb(iter):
                    print("STOPPED")
                    break

            if np.linalg.norm(dx) < 0.001:
                print("CONVERGED")
                break
    
    def calculate_H_b(self):
        pos_count = len(self.graph.positions)
        lm_count = len(self.graph.landmarks)
        H_SIZE = pos_count * STATE_SIZE +  lm_count * LM_SIZE
        H = np.zeros((H_SIZE, H_SIZE))
        b = np.zeros((H_SIZE))

        delta = 1.5
        deltaSqr = delta**2
            
        def calc_error_huber(e, delta):
            if e <= deltaSqr:
                return e, 1
            sqrte = np.sqrt(e)
            return 2 * sqrte * delta - deltaSqr, delta / sqrte
        
        err = 0
        for edge in self.graph.landmark_edges:
            index1 = edge.pos_id * STATE_SIZE
            index2 = pos_count * STATE_SIZE + edge.lm_id * LM_SIZE
            e, A, B = edge.calc_error(self.graph)

            chi_2 = np.dot(e, edge.information @ e)
            er, err_J = calc_error_huber(chi_2, delta)
            INF = edge.information * err_J
            INF_W = INF @ e

            H[index1:index1 + STATE_SIZE, index1:index1 + STATE_SIZE] += A.T @ INF @ A
            H[index2:index2 + LM_SIZE, index2:index2 + LM_SIZE]       += B.T @ INF @ B
            H[index1:index1 + STATE_SIZE, index2:index2 + LM_SIZE]    += A.T @ INF @ B
            H[index2:index2 + LM_SIZE, index1:index1 + STATE_SIZE]    += B.T @ INF @ A
            b[index1:index1 + STATE_SIZE] += A.T @ INF_W
            b[index2:index2 + LM_SIZE]    += B.T @ INF_W
            err += er

        for edge in self.graph.se2_edges:
            index1 = edge.pos_id_1 * STATE_SIZE
            index2 = edge.pos_id_2 * STATE_SIZE
            e, A, B = edge.calc_error(self.graph)
            chi_2 = np.dot(e, edge.information @ e)
            er, err_J = calc_error_huber(chi_2, delta)
            INF = edge.information * err_J
            INF_W = INF @ e

            H[index1:index1 + STATE_SIZE, index1:index1 + STATE_SIZE] += (A.T @ INF @ A)
            H[index2:index2 + STATE_SIZE, index2:index2 + STATE_SIZE] += (B.T @ INF @ B)
            H[index1:index1 + STATE_SIZE, index2:index2 + STATE_SIZE] += (A.T @ INF @ B)
            H[index2:index2 + STATE_SIZE, index1:index1 + STATE_SIZE] += (B.T @ INF @ A)
            b[index1:index1 + STATE_SIZE] += A.T @ INF_W
            b[index2:index2 + STATE_SIZE] += B.T @ INF_W
            err += er

        for fixed_pos_id in self.graph.fixed_positions:
            index = fixed_pos_id * STATE_SIZE
            H[index:index + STATE_SIZE, index:index + STATE_SIZE] += np.eye(STATE_SIZE) * 1e10
            b[index:index + STATE_SIZE] = np.zeros(STATE_SIZE)

        # for fixed_lm_id in self.graph.fixed_lms:
        #     index = pos_count * STATE_SIZE + fixed_lm_id * LM_SIZE
        #     H[index:index + LM_SIZE, index:index + LM_SIZE] += np.eye(LM_SIZE) * 1e10
        #     b[index:index + LM_SIZE] = np.zeros(LM_SIZE)

        return H, b, err