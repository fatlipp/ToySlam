import matplotlib.pyplot as plt
import numpy as np
from tools import *

class View:
    def __init__(self, min = 0, max = 0, size = 1) -> None:
        self.min = min
        self.max = max
        self.size_data = size
        self.drawables = []

        self.fig, self.ax = plt.subplots(figsize=(18,18))
        self.ax.set_aspect(1)

        self.cid = self.ax.figure.canvas.mpl_connect('draw_event', self._resize)

    def render_grid(self):
        if self.max - self.min > 0:
            for v in np.arange(self.min, self.max):
                self.ax.axvline(v - 0.5)
                self.ax.axvline(v + 0.5)
                self.ax.axhline(v - 0.5)
                self.ax.axhline(v + 0.5)
            self.ax.set_xlim(self.min - 0.5, self.max + 0.5)
            self.ax.set_ylim(self.min - 0.5, self.max + 0.5)

    # def draw(self, x, y, size = 1, **kwargs):
    #     self.n = len(x)
    #     self.ax.figure.canvas.draw()
    #     self.size_data=size
    #     self.size = size
    #     self.sc = self.ax.scatter(x,y,s=self.size,**kwargs)
    #     self._resize()
    #     self.cid = self.ax.figure.canvas.mpl_connect('draw_event', self._resize)

    def add_drawable(self, drawable, count):
        self.drawables.append((drawable, count))
        self._resize()

    def _resize(self, event = None):
        ppd = 72. / self.ax.figure.dpi
        trans = self.ax.transData.transform

        s =  ((trans((1, self.size_data)) - trans((0,0))) * ppd)[1]

        if s != self.size_data:
            # for d,sizes in self.drawables:
            #     d.set_markersize(s**2)
            self.size_data = s
            self._redraw_later()
    
    def _redraw_later(self):
        self.timer = self.ax.figure.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.ax.figure.canvas.draw_idle())
        self.timer.start()

    def show(self, block):
        self.fig.tight_layout()
        plt.show(block=block)

class RobotStateView:
    def __init__(self, state, lidar_fov, ax, size, color):
        self.state = state
        self.lidar_fov = lidar_fov
        self.dir_len = 2
        self.dir_width = 2
        self.lidar_fov_len = 10

        p = np.copy(self.state)

        self.cloud_view = ax.plot(
                [], [], '.', 
                markersize=size, alpha=1.0, color=color,
                label=f'Cloud')[0]

        self.view = ax.plot([p[0,2]], [p[1,2]], '.', 
                    markersize=size, alpha=1.0, color=color,
                    label=f'Rob')[0]
        
        angle = mat_to_angle_2d(p[:2,:2])
        self.dir_view = ax.plot([p[0,2], p[0,2] + self.dir_len * np.cos(angle)], 
                                     [p[1,2], p[1,2] + self.dir_len * np.sin(angle)], '-', 
                    linewidth=self.dir_width, alpha=1.0, color='black')[0]

        fov_2 = lidar_fov * 0.5
        dir_min = np.array([np.cos(angle - fov_2), np.sin(angle - fov_2)])
        dir_max = np.array([np.cos(angle + fov_2), np.sin(angle + fov_2)])
        self.lidar_view = ax.plot(
            [p[0,2], p[0,2] + dir_min[0] * self.lidar_fov_len, p[0,2], p[0,2] + dir_max[0] * self.lidar_fov_len], 
            [p[1,2], p[1,2] + dir_min[1] * self.lidar_fov_len, p[0,2], p[1,2] + dir_max[1] * self.lidar_fov_len], 
            '--', 
            linewidth=2, alpha=0.9, color=color,
            label=f'rays')[0]
        
        self.cloud = None

    def update_state(self, state):
        self.state = np.copy(state)

        # pos
        self.view.set_xdata([state[0,2]])
        self.view.set_ydata([state[1,2]])
        angle = mat_to_angle_2d(state[:2,:2])

        # dir
        self.dir_view.set_xdata([self.state[0,2], self.state[0,2] + self.dir_len * np.cos(angle)])
        self.dir_view.set_ydata([self.state[1,2], self.state[1,2] + self.dir_len * np.sin(angle)])

        # lidar angle
        fov_2 = self.lidar_fov * 0.5
        dir_min = np.array([np.cos(angle + fov_2), np.sin(angle + fov_2)])
        dir_max = np.array([np.cos(angle - fov_2), np.sin(angle - fov_2)])
        self.lidar_view.set_xdata(
            [self.state[0,2], self.state[0,2] + dir_min[0] * self.lidar_fov_len, 
             self.state[0,2], self.state[0,2] + dir_max[0] * self.lidar_fov_len])
        self.lidar_view.set_ydata(
            [self.state[1,2], self.state[1,2] + dir_min[1] * self.lidar_fov_len, 
             self.state[1,2], self.state[1,2] + dir_max[1] * self.lidar_fov_len])

        # cloud
        self.update_landmarks_view()

    def set_landmarks(self, landmarks):
        if landmarks is None:
            return
        
        self.cloud = convert_2d_rays_to_cloud(landmarks)
        self.update_landmarks_view()
        
    def update_landmarks_view(self):
        if self.cloud is None:
            return
        
        cloud_trans = transform_cloud(self.state, self.cloud)
        self.cloud_view.set_xdata(cloud_trans[0,:])
        self.cloud_view.set_ydata(cloud_trans[1,:])

class FootprintView2d:
    def __init__(self, ax, size, color, alpha):
        self.x_data = []
        self.y_data = []
        self.x_dir_data = []
        self.y_dir_data = []
        self.dir_len = 1
        self.dir_width = 1

        self.pos_view = ax.plot([], [], '.', 
                    markersize=size, alpha=alpha, color=color,
                    label=f'pos')[0]
        self.dir_view = ax.plot([], [], '-', 
                    linewidth=self.dir_width, alpha=1.0, color='black')[0]
        
    def add_footprint(self, transform):
        self.x_data.append(transform[0,2])
        self.y_data.append(transform[1,2])

        angle = mat_to_angle_2d(transform[:2,:2])
        self.x_dir_data.append([transform[0,2], transform[0,2] + np.cos(angle) * self.dir_len])
        self.y_dir_data.append([transform[1,2], transform[1,2] + np.sin(angle) * self.dir_len])
        
        self.update()

    def update_footprint(self, id, transform):
        self.x_data[id] = transform[0,2]
        self.y_data[id] = transform[1,2]

        angle = mat_to_angle_2d(transform[:2,:2])
        self.x_dir_data[id] = [transform[0,2], transform[0,2] + np.cos(angle) * self.dir_len]
        self.y_dir_data[id] = [transform[1,2], transform[1,2] + np.sin(angle) * self.dir_len]
        
        self.update()

    def update(self):
        self.pos_view.set_xdata(self.x_data)
        self.pos_view.set_ydata(self.y_data)
        self.dir_view.set_xdata(self.x_dir_data)
        self.dir_view.set_ydata(self.y_dir_data)