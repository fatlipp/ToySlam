import numpy as np

from tools import mat_to_angle_2d
class BaseEdge2:
    def __init__(self, id_1, id_2, measurement, information) -> None:
        self.id_1 = id_1
        self.id_2 = id_2
        self.measurement = measurement
        self.information = information
    
    def get_type(self):
        raise NotImplementedError("'get_type()' is not implemented")

class EdgeLandmark2d(BaseEdge2):
    def __init__(self, id_1, id_2, measurement, information) -> None:
        super().__init__(id_1, id_2, measurement, information)

    def get_type(self):
        return 1
    
    def calc_error(self, graph):
        pos = graph.vertices[self.id_1].position
        lm = graph.vertices[self.id_2].position

        lm_local = [self.measurement[0] * np.cos(self.measurement[1]), 
                self.measurement[0] * np.sin(self.measurement[1])]

        pp = (np.linalg.inv(pos) @ np.array([lm[0], lm[1], 1]))[:2]
        err = pp - lm_local


        # print("lm id: {}-{}, pp: {}, lm_local: {}".format(
        #     self.id_1, self.id_2, pp, lm_local))

        x1, y1, th1 = pos[0, 2], pos[1, 2],  mat_to_angle_2d(pos[:2,:2])
        cosA = np.cos(th1)
        sinA = np.sin(th1)

        A = np.zeros((2, 3))
        A[0, 0] = -cosA
        A[0, 1] = -sinA
        A[0, 2] = cosA * lm[1] - sinA * lm[0] - cosA * y1 + sinA * x1
        A[1, 0] = sinA
        A[1, 1] = -cosA
        A[1, 2] = -sinA * lm[1] - cosA * lm[0] + sinA * y1 + cosA * x1 

        B = -np.eye(2, 2)
        B[0, 0] = cosA
        B[0, 1] = sinA
        B[1, 0] = -sinA
        B[1, 1] = cosA

        return err, A, B
    
    def get_id(self, index):
        return self.id_1 if index == 0 else self.id_2
    
class EdgeOdometry2d(BaseEdge2):
    def __init__(self, id_1, id_2, measurement, information) -> None:
        super().__init__(id_1, id_2, measurement, information)

    def get_type(self):
        return 0
    
    def calc_error(self, graph):
        pos_1 = graph.vertices[self.id_1].position
        pos_2 = graph.vertices[self.id_2].position
        odom = self.measurement

        pp = np.linalg.inv(pos_1) @ pos_2

        delta = np.linalg.inv(odom) @ pp
        err = np.array([delta[0,2], delta[1,2], mat_to_angle_2d(delta[:2,:2])])

        A = -np.eye(3)
        B = np.eye(3)        

        return err, A, B
    
    def get_id(self, index):
        return self.id_1 if index == 0 else self.id_2
    
# class EdgeVirtualLandmark2d:
#     def __init__(self, pos_id_1, pos_id_2, lm_meas_1, lm_meas_2, information) -> None:
#         self.pos_id_1 = pos_id_1
#         self.pos_id_2 = pos_id_2
#         self.lm_meas_1 = lm_meas_1
#         self.lm_meas_2 = lm_meas_2
#         self.information = information

#     def calc_error(self, graph):
#         pos_1 = graph.positions[self.pos_id_1].measurement
#         pos_2 = graph.positions[self.pos_id_2].measurement

#         lm1_from_1 = pos_2 @ np.array([self.lm_meas_1[0] * np.cos(self.lm_meas_1[1]), 
#                       self.lm_meas_1[0] * np.sin(self.lm_meas_1[1]),
#                       1])
#         lm2_from_2 = pos_1 @ np.array([self.lm_meas_2[0] * np.cos(self.lm_meas_2[1]), 
#                          self.lm_meas_2[0] * np.sin(self.lm_meas_2[1]),
#                          1])
#         err = (lm1_from_1 - lm2_from_2)[:2]

#         A = self.calc_J(pos_1, self.lm_meas_1)
#         B = self.calc_J(pos_2, self.lm_meas_2)
#         return err, A, B

#     def calc_J(self, pos, lm):
#         theta = mat_to_angle_2d(pos[:2, :2])
#         c = np.cos(theta)
#         s = np.sin(theta)
#         dist, angle = lm

#         J = np.zeros((2, 3))
#         J[0, 0] = 1  # d(err_x)/d(x)
#         J[0, 1] = 0  # d(err_x)/d(y)
#         J[0, 2] = -dist * s * np.cos(angle) - dist * c * np.sin(angle)  # d(err_x)/d(theta)
#         J[1, 0] = 0  # d(err_y)/d(x)
#         J[1, 1] = 1  # d(err_y)/d(y)
#         J[1, 2] = dist * c * np.cos(angle) - dist * s * np.sin(angle)  # d(err_y)/d(theta)

#         return J    