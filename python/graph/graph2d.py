import numpy as np
from collections import defaultdict
from tools import *
from graph.edges2d import EdgeOdometry

class Vertex2d:
    def __init__(self, measurement) -> None:
        if measurement.shape[0] == 4:
            self.measurement = np.identity(3)
            self.measurement[:2,:2] = measurement[:2, :2]
            self.measurement[:2,2] = measurement[:2, 3]
        else:
            self.measurement = measurement

class MeasurementData2d:
    def __init__(self, measurement, information) -> None:
        self.measurement = measurement
        self.information = information
    
class Graph2d:
    def __init__(self):
        self.vertex_id = 0
        self.positions_count = 0

        self.positions = []
        self.odometry_edges = []
        self.fixed_positions = set()

        self.observers_lm = defaultdict(list)
        self.landmark_edges = {}

    def fix_position(self, id):
        if id >= len(self.positions):
            raise RuntimeError("Fix: {} is not found".format(id))
        self.fixed_positions.add(id)

    def add_pose(self, pose, fixed=False):
        self.positions.append(Vertex2d(pose))
        id = len(self.positions) - 1

        if fixed:
            self.fix_position(id)

        self.landmark_edges[id] = {}
        self.positions_count += 1
        return id

    def add_odometry(self, id_from, odom, information=np.identity(3)):
        if id_from >= len(self.positions):
            raise RuntimeError("{} is not found".format(id_from))
        
        odom_2d = odom
        if odom_2d.shape[0] == 4:
            odom_2d = np.identity(3)
            odom_2d[:2,:2] = odom[:2, :2]
            odom_2d[:2,2] = odom[:2, 3]

        current_pos = self.positions[id_from].measurement @ odom_2d
        id_to = self.add_pose(current_pos)

        self.odometry_edges.append(EdgeOdometry(id_from, id_to, odom_2d, information))

        return id_to

    def add_lidar_edge(self, pos_id, measurement, lidar_id, information=np.identity(2)):
        """
        pos_id - robot position id
        measurement - LiDAR measurement (range, angle)
        lidar_id - lidar id
        """
        if pos_id >= len(self.positions):
            raise RuntimeError("add_lidar_edge() {} is not found".format(pos_id))
        
        self.observers_lm[lidar_id].append(pos_id)
        self.landmark_edges[pos_id][lidar_id] = MeasurementData2d(measurement, information)
    
    def get_size(self):
        return self.positions_count