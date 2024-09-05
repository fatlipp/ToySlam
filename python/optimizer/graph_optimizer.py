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
        self.vertex_ids_map = None

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
            err = self.calculate_H_b()

            if self.H is None or self.b is None:
                print("Opt result is NONE...")
                break
            
            lambda_val = update_lambda(prev_err > -1 and err > prev_err, lambda_val, lambda_factor)
            H_regularized = self.H + lambda_val * np.eye(self.H.shape[0])

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
            # dx = np.linalg.lstsq(H_regularized, -self.b)[0]
            dx = solve(H_regularized, -self.b)

            dx *= lr
            t_end = time.time()
            duration_solve = t_end - t_start


            t_start = time.time()
            
            for v_id in self.graph.vertices:
                v = self.graph.vertices[v_id]
                index = self.vertex_ids_map[v_id]
                dd = dx[index:index + v.get_dims()]
                self.graph.vertices[v_id].update(dd)

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
        if self.vertex_ids_map is None:
            vertex_ids_map = {}
            s_id = 0
            for v in self.graph.vertices:
                vertex_ids_map[v] = s_id
                s_id += self.graph.vertices[v].get_dims()
            self.vertex_ids_map = vertex_ids_map

            self.size = s_id

        self.H = np.zeros((self.size, self.size))
        self.b = np.zeros((self.size))

        delta = 1.5
        deltaSqr = delta**2
            
        def calc_error_huber(e, delta):
            if e <= deltaSqr:
                return e, 1
            sqrte = np.sqrt(e)
            return 2 * sqrte * delta - deltaSqr, delta / sqrte
        
        def get_state_size(id, graph):
            return graph.vertices[id].get_dims()
    
        err = 0

        for edge in self.graph.edges:
            e, A, B = edge.calc_error(self.graph)

            chi_2 = np.dot(e, edge.information @ e)
            er, err_J = calc_error_huber(chi_2, delta)
            INF = edge.information * err_J
            INF_W = INF @ e

            index1 = self.vertex_ids_map[edge.get_id(0)]
            index2 = self.vertex_ids_map[edge.get_id(1)]

            BLOCK_SIZE_1 = get_state_size(edge.get_id(0), self.graph)
            BLOCK_SIZE_2 = get_state_size(edge.get_id(1), self.graph)
            self.H[index1:index1 + BLOCK_SIZE_1, index1:index1 + BLOCK_SIZE_1] += A.T @ INF @ A
            self.H[index2:index2 + BLOCK_SIZE_2, index2:index2 + BLOCK_SIZE_2] += B.T @ INF @ B
            self.H[index1:index1 + BLOCK_SIZE_1, index2:index2 + BLOCK_SIZE_2] += A.T @ INF @ B
            self.H[index2:index2 + BLOCK_SIZE_2, index1:index1 + BLOCK_SIZE_1] += B.T @ INF @ A
            self.b[index1:index1 + BLOCK_SIZE_1] += A.T @ INF_W
            self.b[index2:index2 + BLOCK_SIZE_2] += B.T @ INF_W
            err += er

            # print("lm id: {}-{}, errRobust: {}".format(
            #     edge.id_1, edge.id_2, er))

        for v_id in self.graph.fixed_vertices:
            index = self.vertex_ids_map[v_id]
            STATE_SIZE = get_state_size(v_id, self.graph)
            self.H[index:index + STATE_SIZE, index:index + STATE_SIZE] += np.eye(STATE_SIZE) * 1e6
            self.b[index:index + STATE_SIZE] = np.zeros(STATE_SIZE)

        print("err: ", err)
        # print("H: ", np.sum(self.H))
        # print("b: ", np.sum(self.b))

        return err