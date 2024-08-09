import numpy as np
from collections import defaultdict
from tools import *

class PositionData:
    def __init__(self, id, position, landmark_measurements, is_fixed = False):
        self.id = id
        self.position = position
        self.landmark_measurements = landmark_measurements
        self.odometry = None
        self.is_fixed = is_fixed

    def set_odometry(self, odom):
        self.odometry = odom

class Graph2d:
    def __init__(self):
        self.positions = {}
        self.landmarks = {}
        self.pos_id = -1

    def add_pose(self, pos, landmark_measurements, is_fixed = False):
        self.pos_id += 1
        self.positions[self.pos_id] = PositionData(self.pos_id, pos, landmark_measurements, is_fixed)
        return self.pos_id

    def get_pose(self, id):
        return self.positions[id]
    
    def add_landmark(self, id, position):
        if id not in self.landmarks:
            self.landmarks[id] = position

    def get_size(self):
        return len(self.positions)
    
    def get_positions(self):
        return self.positions
    
    def get_landmarks(self):
        return self.landmarks
