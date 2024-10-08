import numpy as np
import matplotlib.pyplot as plt
import asyncio

from lidar_sensor import calc_lidar_measurements
from environment import load_env
from view.robot_view_2d import View, RobotStateView, FootprintView2d
from view.graph_view_2d import GraphView2d
from optimizer.graph_optimizer import GraphOptimizer
from optimizer.opt_graph import OptGraph
from optimizer.vertices import *
from optimizer.edges2d import EdgeOdometry2d, EdgeLandmark2d
from slam.graph2d import Graph2d
from slam.slam_helper import add_to_graph, motion_model
from tools import *

from remote.graph_client import GraphClient

np.random.seed(0)

# RED Points - robots currect observation and position
# GREEN - Ground Truth
# Orange - history
# Blue - Global Map

# MAIN CONFIG IS HERE:
POINT_SIZE = 60

ROBOT_STEPS = 150
OPTIMIZATION_STEPS = 10
LR = .2

lidar_fov = np.deg2rad(120)
lidar_ray_step = np.deg2rad(6) # decreas

lidar_std_ved = .15
position_std_dev = 0.5
orientation_std_dev_deg = np.deg2rad(7.1)
# END MAIN CONFIG

# INF and NOISE matrices (dep on config)
LIDAR_NOISE = np.identity(2)
LIDAR_NOISE[0, 0] = lidar_std_ved**2
LIDAR_NOISE[1, 1] = lidar_std_ved**2

ODOMETRY_NOISE = np.identity(3)
ODOMETRY_NOISE[:2,:2] *= position_std_dev**2
ODOMETRY_NOISE[2, 2] *= orientation_std_dev_deg**2

LIDAR_INF = np.linalg.inv(LIDAR_NOISE) * 1
ODOM_INF = np.linalg.inv(ODOMETRY_NOISE) * 1

# environment
environment, radius = load_env()
print("environment: ", environment.shape)

# state
state_mat = np.eye(3, 3, dtype=float)
state_mat[0,2] = 5
state_mat[1,2] = 15
state_mat_gt = np.copy(state_mat)

state_transform_mat = np.eye(3, 3)
state_transform_mat[:2,:2] = angle_to_mat_2d(np.deg2rad(2.))
state_transform_mat[:2, 2] = np.array([1.2, 0.0])

## VIEW
# 0. Main
view = View(-5, 75)
view.render_grid()
view.fig.legend()
# 1. Env.
env_view = view.ax.plot(environment[:,0], environment[:,1], '.', 
                         markersize=POINT_SIZE, alpha=0.15, color='black',
                         label=f'env')[0]
# 2. current measurements
robot_gt = RobotStateView(state_mat_gt, lidar_fov, view.ax, POINT_SIZE * 0.55, 'green')
robot_est = RobotStateView(state_mat, lidar_fov, view.ax, POINT_SIZE * 0.6, 'red')
# 3. previous GT
footprint_view_gt = FootprintView2d(view.ax, size=POINT_SIZE * 0.4, color='green', alpha=0.6)
# 4. graph
graph_view = GraphView2d(view.ax, size=POINT_SIZE * 0.8, color_pos='orange',\
                         color_map='blue', alpha=0.6)
####

graph = Graph2d()
landmarks_curr, ids_curr = \
    calc_lidar_measurements(state_mat, environment, radius, lidar_fov, lidar_ray_step)

if landmarks_curr is None or ids_curr is None:
    exit(-1)
pos_id = add_to_graph(graph, state_mat, landmarks_curr, ids_curr, LIDAR_NOISE, True)

robot_est.update_state(state_mat)
robot_est.set_landmarks(landmarks_curr)
footprint_view_gt.add_footprint(state_mat_gt)
graph_view.update(graph)

async def step():
    global state_mat, state_mat_gt, state_transform_mat, pos_id
    
    # TODO:
    # some motion behaviour
    if pos_id < 10:
        state_transform_mat = np.eye(3, 3)
        state_transform_mat[:2,:2] = angle_to_mat_2d(np.deg2rad(3.))
        state_transform_mat[:2, 2] = np.array([2.0, 0.0])
    elif pos_id < 20:
        state_transform_mat = np.eye(3, 3)
        state_transform_mat[:2,:2] = angle_to_mat_2d(np.deg2rad(6.))
        state_transform_mat[:2, 2] = np.array([0.9, 0.0])
    elif pos_id < 40:
        state_transform_mat = np.eye(3, 3)
        state_transform_mat[:2,:2] = angle_to_mat_2d(np.deg2rad(-6.))
        state_transform_mat[:2, 2] = np.array([0.9, 0.0])
    elif pos_id < 60:
        state_transform_mat = np.eye(3, 3)
        state_transform_mat[:2,:2] = angle_to_mat_2d(np.deg2rad(5.))
        state_transform_mat[:2, 2] = np.array([0.8, 0.0])
    else:
        state_transform_mat = np.eye(3, 3)
        state_transform_mat[:2,:2] = angle_to_mat_2d(np.deg2rad(3.))
        state_transform_mat[:2, 2] = np.array([0.7, 0.0])
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
    RT[0, 2] = RT[0, 2] + np.random.normal(0, ODOMETRY_NOISE[0,0])
    RT[1, 2] = RT[1, 2] + np.random.normal(0, ODOMETRY_NOISE[1,1])
    RT[:2,:2] = angle_to_mat_2d(mat_to_angle_2d(state_transform_mat[:2,:2])
                            + np.random.normal(0, ODOMETRY_NOISE[2,2]))
    RT[2, 2] = 1

    graph.get_pose(pos_id).set_odometry(RT)

    state_mat = motion_model(state_mat, RT)
    pos_id = add_to_graph(graph, state_mat, landmarks_curr, ids_curr, LIDAR_NOISE, False)

    # 3. Vis
    robot_gt.update_state(state_mat_gt)
    robot_gt.set_landmarks(landmarks_curr)
    robot_est.update_state(state_mat)
    robot_est.set_landmarks(landmarks_curr)
    footprint_view_gt.add_footprint(state_mat_gt)
    graph_view.update(graph)

def construct_optimizer_graph(graph):
    graph_opt = OptGraph()

    positions = graph.get_positions()
    for i in range(len(positions)):
        pos = positions[i]
        graph_opt.add_vertex(pos.id, VertexPose2d(pos.position), pos.is_fixed)

    for i in range(len(positions)):
        pos = positions[i]
        if pos.odometry is not None:
            graph_opt.add_edge(EdgeOdometry2d(pos.id, pos.id + 1, pos.odometry, ODOM_INF))

    lm_id_glob = len(positions)
    landmarks_id_map = {}
    for i in range(len(positions)):
        pos = positions[i]
        for lm_id in pos.landmark_measurements:
            if lm_id not in landmarks_id_map:
                landmarks_id_map[lm_id] = lm_id_glob
                lm_id_glob += 1
            graph_opt.add_edge(EdgeLandmark2d(pos.id, landmarks_id_map[lm_id],\
                                        pos.landmark_measurements[lm_id], LIDAR_INF))

    landmarks = graph.get_landmarks()
    for lm_id in landmarks:
        if lm_id not in landmarks_id_map:
            continue
        graph_opt.add_vertex(landmarks_id_map[lm_id], Vertex2d(landmarks[lm_id]), False)

    return graph_opt, landmarks_id_map

def on_step(iter):
    global graph_opt, landmarks_id_map
    
    # update_graph(graph, graph_opt, landmarks_id_map)

    return plt.fignum_exists(view.fig.number)

def update_graph(graph, graph_opt, landmarks_id_map):

    positions = graph.get_positions()

    for i in range(len(positions)):
        pos = positions[i]
        positions[i].position = graph_opt.get_vertex(pos.id).position

    for lm_id in graph.get_landmarks():
        if lm_id not in landmarks_id_map:
            continue
        graph.get_landmarks()[lm_id] = graph_opt.get_vertex(landmarks_id_map[lm_id]).position

    robot_est.update_state(positions[len(positions) - 1].position)

    graph_view.update(graph)

    plt.pause(0.001)

async def optimize(client, graph):

    global graph_opt, landmarks_id_map
    graph_opt = OptGraph()
    graph_opt, landmarks_id_map = construct_optimizer_graph(graph)

    import time
    t_start = time.time()
    if client is None:
        optimizer = GraphOptimizer(graph_opt)
        optimizer.set_on_step_cb(on_step)
        optimizer.optimize(OPTIMIZATION_STEPS, LR)
    else:
        graph_opt = await client.optimize(graph_opt)
    t_end = time.time()
    duration = t_end - t_start

    print("OPT DURATION: ", duration)

    # optimizer = GraphOptimizer(graph_opt)
    # optimizer.set_on_step_cb(on_step)
    # optimizer.optimize(OPTIMIZATION_STEPS, LR)

    update_graph(graph, graph_opt, landmarks_id_map)

async def start_client(host, port):
    plt.ion()
    plt.show()

    client = GraphClient(host, port)

    try:
        await client.connect()
    except:
        print("Cannot connect to server: ", host, port)
        client = None

    optimized = False

    while plt.fignum_exists(view.fig.number):
        if graph.get_size() < ROBOT_STEPS:
            await step()
        elif not optimized:
            optimized = True
            await optimize(client, graph)
        
        plt.pause(0.001)
        await asyncio.sleep(0.001)

    if client is not None:
        await client.close()

    
# host, port = sys.argv[1], sys.argv[2]
host, port = '127.0.0.1', 8888

async def main():
    await start_client(host, port)

asyncio.run(main())