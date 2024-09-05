from enum import Enum
import numpy as np
from tools import *

class BaseVertex:
    def __init__(self, position) -> None:
        self.position = position

    def get_type(self):
        raise NotImplementedError("'get_type()' is not implemented")

    def get_dims(self):
        raise NotImplementedError("'get_dims()' is not implemented")

    def update(self, delta):
        raise NotImplementedError("'update()' is not implemented")

class VertexPose2d(BaseVertex):
    def __init__(self, position) -> None:
        super().__init__(position)

    def get_type(self):
        return 0

    def get_dims(self):
        return 3

    def update(self, delta):
        theta = mat_to_angle_2d(self.position[:2, :2]) + delta[2]
        c, s = np.cos(theta), np.sin(theta)
        self.position[:2, :2] = np.array([[c, -s], [s, c]])
        self.position[0, 2] += delta[0]
        self.position[1, 2] += delta[1]

class Vertex2d(BaseVertex):
    def __init__(self, position) -> None:
        super().__init__(position)

    def get_type(self):
        return 1

    def get_dims(self):
        return 2

    def update(self, delta):
        self.position += delta