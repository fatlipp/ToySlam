import numpy as np
from tools import mat_to_angle_2d

class GraphView2d:
    def __init__(self, ax, size, color_pos, color_map, alpha):
        self.ax = ax
        self.dir_len = 1
        self.dir_width = 1

        self.dir_views = []
        self.positions_view = ax.plot([], [], '.', 
                markersize=size, alpha=alpha, color=color_pos,
                label=f'Positions')[0]
        self.landmark_view = ax.plot([], [], '.', 
                markersize=size, alpha=alpha, color=color_map,
                label=f'Landmarks')[0]

    def update(self, graph):
        self.clear()
        
        positions = []

        for pos in graph.get_positions():
            trans = pos.position
            p = trans[:2,2]
            positions.append(p)

            angle = mat_to_angle_2d(trans[:2,:2])
            
            self.dir_views.append(\
                    self.ax.plot([p[0], p[0] + np.cos(angle) * self.dir_len],
                                 [p[1], p[1] + np.sin(angle) * self.dir_len], '-', 
                                 linewidth=self.dir_width, alpha=1.0, color='black')[0])
            
        positions = np.array(positions)
        self.positions_view.set_data(positions[:,0], positions[:,1])

        landmarks = []
        for lm_id in graph.get_landmarks():
            landmarks.append(graph.get_landmarks()[lm_id])
        landmarks = np.array(landmarks)
        self.landmark_view.set_data(landmarks[:,0], landmarks[:,1])

    def clear(self):
        for view in self.dir_views:
            view.remove()
        self.dir_views = []

        
    def add_pos(self, transform):
        pass

    def update_pos(self, id, transform):
        pass

    def add_map_point(self, id, pos):
        pass

    def update_map_point(self, id, pos):
        pass