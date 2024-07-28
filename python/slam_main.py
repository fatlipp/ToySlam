import numpy as np
import matplotlib.pyplot as plt
import asyncio

from lidar_sensor import calc_lidar_measurements
from environment import load_env
from vis import View, RobotStateView, FootprintView2d, FootprintViewWithCloud2d
from graph.graph2d import Graph2d
from graph.graph2d_optimizer import GraphOptimizer
from tools import *

def motion_model(state, motion):
    x = state @ motion
    return x

np.random.seed(0)


# vis
view = View(-5, 75)
view.render_grid()
view.fig.legend()

# environment
environment, radius = load_env()
print("environment: ", environment.shape)

STEPS = 90
OPTIMIZATION_STEPS = 50
LR = 0.6

lidar_fov = np.deg2rad(120)
lidar_ray_step = np.deg2rad(3)

# Optimization based on virtual measurements, algorithm rely on precise LiDAR data
# and noisy odometry:
# A current implementation doesn't optimize lidar measurements.

lidar_std_dev = 0.01
position_std_dev = 0.6
orientation_std_dev_deg = np.deg2rad(0.04)

LIDAR_NOISE = np.identity(2) * lidar_std_dev**2
ODOMETRY_NOISE = np.identity(3)
ODOMETRY_NOISE[:2, :2] *= position_std_dev**2
ODOMETRY_NOISE[2, 2] *= orientation_std_dev_deg**2

LIDAR_INF = np.linalg.inv(LIDAR_NOISE)
ODOM_INF = np.linalg.inv(ODOMETRY_NOISE)

SIZE = 40
env_view = view.ax.plot(environment[:,0], environment[:,1], '.', 
                         markersize=SIZE, alpha=0.15, color='black',
                         label=f'env')[0]

state_mat = np.eye(4, 4, dtype=float)
state_mat[0,3] = 5
state_mat[1,3] = 15
state_mat_gt = np.copy(state_mat)

state_transform_mat = np.eye(4, 4)
state_transform_mat[:3,:3] = angle_to_mat(np.deg2rad(2.))
state_transform_mat[:3, 3] = np.array([1.2, 0.0, 0.0])

robot_gt = RobotStateView(state_mat_gt, lidar_fov, view.ax, SIZE, 'green')
robot_est = RobotStateView(state_mat, lidar_fov, view.ax, SIZE, 'orange')
footprint_view_gt = FootprintView2d(view.ax, size=SIZE * 0.8, color='green', alpha=0.8)

graph = Graph2d()
prev_id = graph.add_pose(state_mat, True)
landmarks_curr, ids_curr = \
    calc_lidar_measurements(state_mat_gt, environment, radius, lidar_fov, lidar_ray_step)

if landmarks_curr is None or ids_curr is None:
    exit(-1)

for i in range(len(landmarks_curr)):
    graph.add_lidar_edge(prev_id, landmarks_curr[i], ids_curr[i] + 1000, LIDAR_INF)

robot_est.update_state(state_mat)
robot_est.set_landmarks(landmarks_curr)
footprint_view_gt.add_footprint(state_mat_gt)

robot_pos_history = {}
robot_pos_history[prev_id] = FootprintViewWithCloud2d(state_mat, landmarks_curr, view.ax, SIZE * 0.85, 'red', 0.7)

async def step():
    global state_mat, state_mat_gt, prev_id, ll_view, state_transform_mat

    # some motion behaviour
    if prev_id < 10:
        state_transform_mat = np.eye(4, 4)
        state_transform_mat[:3,:3] = angle_to_mat(np.deg2rad(3.))
        state_transform_mat[:3, 3] = np.array([1.0, 0.0, 0.0])
    elif prev_id < 20:
        state_transform_mat = np.eye(4, 4)
        state_transform_mat[:3,:3] = angle_to_mat(np.deg2rad(6.))
        state_transform_mat[:3, 3] = np.array([0.9, 0.0, 0.0])
    elif prev_id < 60:
        state_transform_mat = np.eye(4, 4)
        state_transform_mat[:3,:3] = angle_to_mat(np.deg2rad(-6.))
        state_transform_mat[:3, 3] = np.array([0.9, 0.0, 0.0])
    #########

    # 1. GT
    state_mat_gt = motion_model(state_mat_gt, state_transform_mat)

    # 1.1 Measurements: Real measurements
    landmarks_curr, ids_curr = \
        calc_lidar_measurements(state_mat_gt, environment, radius, lidar_fov, lidar_ray_step)
    
    if landmarks_curr is None or ids_curr is None:
        return

    # 2. Noisy data (i.e. odometry sensor is not precise)
    RT = np.copy(state_transform_mat)
    RT[:2, 3] = RT[:2, 3] + np.random.normal(0, ODOMETRY_NOISE[0,0])
    RT[:3,:3] = angle_to_mat(mat_to_angle(state_transform_mat[:3,:3])
                            + np.random.normal(0, ODOMETRY_NOISE[2,2]))
    RT[2, 2] = 1

    # Add to graph
    state_mat = motion_model(state_mat, RT)
    prev_id = graph.add_odometry(prev_id, RT, ODOM_INF)

    for i in range(len(landmarks_curr)):
        landmarks_curr_noise = np.copy(landmarks_curr[i])
        landmarks_curr_noise[0:2] += np.random.normal(0, LIDAR_NOISE[0,0])
        landmarks_curr[i] = landmarks_curr_noise
        graph.add_lidar_edge(prev_id, landmarks_curr_noise, ids_curr[i] + 1000, LIDAR_INF)

    # 3. Vis
    robot_gt.update_state(state_mat_gt)
    robot_gt.set_landmarks(landmarks_curr)

    robot_est.update_state(state_mat)
    robot_est.set_landmarks(landmarks_curr)
    footprint_view_gt.add_footprint(state_mat_gt)
    robot_pos_history[prev_id] = FootprintViewWithCloud2d(state_mat, landmarks_curr, view.ax, SIZE * 0.85, 'red', 0.36)

def on_step(iter):
    print("CB: ", iter)
    poses = graph.positions
    # view_poses = []
    for i in range(len(poses)):
        trans = poses[i].measurement
        
        p = np.identity(4)
        p[:2,:2] = trans[:2,:2]
        p[:2,3] = trans[:2,2]
        robot_pos_history[i].update_transform(p)

    p = np.identity(4)
    p[:2,:2] = poses[len(poses) - 1].measurement[:2,:2]
    p[:2,3] = poses[len(poses) - 1].measurement[:2,2]
    robot_est.update_state(p)

    plt.pause(0.05)

async def optimize(graph):
    optimizer = GraphOptimizer(graph)
    optimizer.construct(graph)
    optimizer.set_on_step_cb(on_step)
    optimizer.optimize(OPTIMIZATION_STEPS, LR)

async def start_client(host, port):
    plt.ion()
    plt.show()

    optimized = False

    while plt.fignum_exists(view.fig.number):
        if graph.get_size() < STEPS:
            await step()
        elif not optimized:
            optimized = True
            await optimize(graph)
        
        plt.pause(0.01)
        await asyncio.sleep(0.01)

    
# host, port = sys.argv[1], sys.argv[2]
host, port = '127.0.0.1', 8888

async def main():
    await start_client(host, port)

asyncio.run(main())