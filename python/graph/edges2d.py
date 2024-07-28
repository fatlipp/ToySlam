import numpy as np

from tools import mat_to_angle_2d, convert_radial_to_euclidean_2d

class EdgeLidarVirtual2d:
    def __init__(self, pos_id_1, pos_id_2, lm_id, information) -> None:
        self.pos_id_1 = pos_id_1
        self.pos_id_2 = pos_id_2
        self.lm_id = lm_id
        self.information = information
        self.cc = 0

    def calc_error(self, graph):
        pos_1 = graph.positions[self.pos_id_1].measurement
        pos_2 = graph.positions[self.pos_id_2].measurement
        meas_1 = graph.landmark_edges[self.pos_id_1][self.lm_id].measurement
        meas_2 = graph.landmark_edges[self.pos_id_2][self.lm_id].measurement

        m_1_p1 = convert_radial_to_euclidean_2d(meas_1, pos_1)
        m_2_p2 = convert_radial_to_euclidean_2d(meas_2, pos_2)
        err = m_1_p1 - m_2_p2

        A = self.calc_J(pos_1, meas_1)

        # Since the error is defined as the difference between the expected 
        # position of the landmark as seen from p1 and p2, de/dpos2 = negative value 
        B = -self.calc_J(pos_2, meas_2)

        return err, A, B

    def calc_J(self, pos, lm):
        """
        pos - 3x3 robot transform relative to some origin
        lm - (dist, angle) lidar measurement
        """
        theta = mat_to_angle_2d(pos[:2, :2])
        c = np.cos(theta)
        s = np.sin(theta)
        dist, angle = lm

        J = np.zeros((2, 3))
        J[0, 0] = 1  # d(err_x)/d(x)
        J[0, 1] = 0  # d(err_x)/d(y)
        J[0, 2] = -dist * s * np.cos(angle) - dist * c * np.sin(angle)  # d(err_x)/d(theta)
        J[1, 0] = 0  # d(err_y)/d(x)
        J[1, 1] = 1  # d(err_y)/d(y)
        J[1, 2] = dist * c * np.cos(angle) - dist * s * np.sin(angle)  # d(err_y)/d(theta)

        return J    
    
class EdgeOdometry:
    def __init__(self, pos_id_1, pos_id_2, measurement, information):
        self.pos_id_1 = pos_id_1
        self.pos_id_2 = pos_id_2
        self.measurement = measurement
        self.information = information

    def calc_error(self, graph):
        pos_1 = graph.positions[self.pos_id_1].measurement
        pos_2 = graph.positions[self.pos_id_2].measurement

        pp = np.linalg.inv(pos_1) @ pos_2
        odom = self.measurement

        err = np.linalg.inv(odom) @ pp
        err = np.array([err[0,2], err[1,2], mat_to_angle_2d(err[:2,:2])])

        A = -np.eye(3)
        B = np.eye(3)        

        return err, A, B