import numpy as np
from collections import defaultdict
from tools import *

class PositionData:
    def __init__(self, position, landmark_measurements, odometry = None):
        self.position = position
        self.landmark_measurements = landmark_measurements
        self.odometry = odometry

class Graph2d:
    def __init__(self):
        self.positions = []
        self.landmarks = {}

    def add_pose(self, pos, landmark_measurements, odometry = None):
        self.positions.append(PositionData(pos, landmark_measurements, odometry))

    def add_landmark(self, id, position):
        if id not in self.landmarks:
            self.landmarks[id] = position

    def get_last_pos_id(self):
        return self.get_size() - 1

    def get_size(self):
        return len(self.positions)
    
    def get_positions(self):
        return self.positions
    
    def get_landmarks(self):
        return self.landmarks
