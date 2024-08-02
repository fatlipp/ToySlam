import numpy as np
from optimizer.edges2d import EdgeOdometry2d, EdgeLandmark2d#, EdgeVirtualLandmark2d

class OptGraph:
    def __init__(self):
        self.positions = []
        self.se2_edges = []
        self.landmarks = {}
        self.landmark_edges = []
        self.fixed_positions = set()
        self.fixed_lms = set()

    def fix_position(self, id):
        if id >= len(self.positions):
            raise RuntimeError("Fix Pos: {} is not found".format(id))
        self.fixed_positions.add(id)

    def fix_lm(self, id):
        if id not in self.landmarks:
            raise RuntimeError("Fix LM: {} is not found".format(id))
        self.fixed_lms.add(id)

    def add_pose(self, pos, fixed=False):
        self.positions.append(pos)
        if fixed:
            self.fix_position(len(self.positions) - 1)

    def add_landmark(self, id, measurement, fixed = False):
        if id not in self.landmarks:
            self.landmarks[id] = measurement
        if fixed:
            self.fix_lm(id)

    def add_odometry_edge(self, id_from, id_to, odom, information=np.identity(3)):
        if id_from >= len(self.positions):
            raise RuntimeError("{} is not found".format(id_from))
        self.se2_edges.append(EdgeOdometry2d(id_from, id_to, odom, information))

    def add_landmark_edge(self, lm_id, pos_id, measurement, information=np.identity(2)):
        if pos_id >= len(self.positions):
            raise RuntimeError("add_lidar_edge() {} is not found".format(pos_id))
        self.landmark_edges.append(EdgeLandmark2d(pos_id, lm_id, measurement, information))

    # def add_virtual_edge(self, id_from, id_to, meas_1, meas_2, information=np.identity(3)):
    #     if id_from >= len(self.positions):
    #         raise RuntimeError("Pos ({}) is not found".format(id_from))
    #     if id_to >= len(self.positions):
    #         raise RuntimeError("Pos ({}) is not found".format(id_to))
    #     self.se2_edges.append(EdgeVirtualLandmark2d(id_from, id_to, meas_1, meas_2, information))

    def get_position(self, id):
        if id >= len(self.positions):
            raise RuntimeError("get_position() {} is not found".format(id))
        return self.positions[id]

    def get_landmark(self, id):
        if id not in self.landmarks:
            raise RuntimeError("get_landmark() {} is not found".format(id))
        return self.landmarks[id]
