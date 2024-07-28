import numpy as np
import matplotlib.pyplot as plt

from tools import mat_to_angle_2d, convert_radial_to_euclidean_2d

class Pose:
    def __init__(self, measurement):
        self.measurement = measurement

class LandmarkEdge:
    def __init__(self, measurement):
        self.measurement = measurement

class Graph:
    def __init__(self):
        self.positions = {}
        self.landmark_edges = {}

class SLAM:
    def __init__(self, pos_id_1, pos_id_2):
        self.pos_id_1 = pos_id_1
        self.pos_id_2 = pos_id_2

    def calc_error(self, graph, lm_id):
        pos_1 = graph.positions[self.pos_id_1].measurement
        pos_2 = graph.positions[self.pos_id_2].measurement
        meas_1 = graph.landmark_edges[self.pos_id_1][lm_id].measurement
        meas_2 = graph.landmark_edges[self.pos_id_2][lm_id].measurement

        m_1_p1 = convert_radial_to_euclidean_2d(meas_1, pos_1)
        m_2_p2 = convert_radial_to_euclidean_2d(meas_2, pos_2)
        err = m_1_p1 - m_2_p2

        A = self.calc_J(pos_1, meas_1)
        B = self.calc_J(pos_2, meas_2)

        return err, A, B

    def calc_J(self, pos, lm):
        J = np.zeros((2, 3))

        theta = mat_to_angle_2d(pos[:2, :2])
        c = np.cos(theta)
        s = np.sin(theta)
        r, phi = lm

        J[0, 0] = 1  # d(err_x)/d(x)
        J[0, 1] = 0  # d(err_x)/d(y)
        J[0, 2] = -r * s * np.cos(phi) - r * c * np.sin(phi)  # d(err_x)/d(theta)
        J[1, 0] = 0  # d(err_y)/d(x)
        J[1, 1] = 1  # d(err_y)/d(y)
        J[1, 2] = r * c * np.cos(phi) - r * s * np.sin(phi)  # d(err_y)/d(theta)

        return J

    
    def update_state(self, graph):
        H = np.zeros((6, 6))  # Assuming 2 poses, each with 3 DoF (x, y, theta)
        b = np.zeros(6)

        for i in range(1, 4):
            error, A, B = self.calc_error(graph, i)
            # Update H and b
            noise_odometry = 0.01
            SIG = np.diag([noise_odometry**2, noise_odometry**2])

            H[:3, :3] += A.T @ SIG @ A
            H[:3, 3:] += A.T @ SIG @ B
            H[3:, :3] += B.T @ SIG @ A
            H[3:, 3:] += B.T @ SIG @ B

            b[:3] += A.T @ SIG @ error
            b[3:] += B.T @ SIG @ error

        H[:3, :3] += np.eye(3, 3) * 1000

        if np.linalg.det(H) == 0:
            print("BAD DET")
            return

        # Solve for delta state
        delta_state = -np.linalg.solve(H, -b)

        # Update poses
        delta_pos_1 = delta_state[:3] * 0.1
        delta_pos_2 = delta_state[3:] * 0.1

        # Apply the delta to the poses
        pos_1 = graph.positions[self.pos_id_1].measurement
        pos_2 = graph.positions[self.pos_id_2].measurement

        graph.positions[self.pos_id_1].measurement = self.apply_delta(pos_1, delta_pos_1)
        graph.positions[self.pos_id_2].measurement = self.apply_delta(pos_2, delta_pos_2)

    def apply_delta(self, pos, delta):
        theta = mat_to_angle_2d(pos[:2, :2]) + delta[2]
        c, s = np.cos(theta), np.sin(theta)

        pos[:2, :2] = np.array([[c, -s], [s, c]])
        pos[0, 2] += delta[0]
        pos[1, 2] += delta[1]

        return pos

# Initialize graph
graph = Graph()

# Define initial poses (3x3 transformation matrices)
pos_1 = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
pos_2 = np.array([[0,  1, 0.1],
                  [1,  0, 1.1],
                  [0,  0, 1]])
pos_2_gt = np.array([[1,  0, 0],
                     [0,  1, 3],
                     [0,  0, 1]])
lm_1 = np.array([3, 1])
lm_2 = np.array([3.1, 2])
lm_3 = np.array([3.6, 1.2])

# Define initial measurements (range, bearing)
ang_1 = [pos_1[0, 2], pos_1[1, 2], mat_to_angle_2d(pos_1[:2, :2])]
ang_2 = [pos_2_gt[0, 2], pos_2_gt[1, 2], mat_to_angle_2d(pos_2_gt[:2, :2])]
meas_1 = convert_radial_to_euclidean_2d(lm_1, ang_1)
meas_12 = convert_radial_to_euclidean_2d(lm_2, ang_1)
meas_13 = convert_radial_to_euclidean_2d(lm_3, ang_1)
meas_2 = convert_radial_to_euclidean_2d(lm_1, ang_2)
meas_22 = convert_radial_to_euclidean_2d(lm_2, ang_2)
meas_23 = convert_radial_to_euclidean_2d(lm_3, ang_2)

graph.positions[1] = Pose(pos_1)
graph.positions[2] = Pose(pos_2)
graph.landmark_edges[1] = {1: LandmarkEdge(meas_1), 2: LandmarkEdge(meas_12), 3: LandmarkEdge(meas_13)}
graph.landmark_edges[2] = {1: LandmarkEdge(meas_2), 2: LandmarkEdge(meas_22), 3: LandmarkEdge(meas_23)}

slam = SLAM(1, 2)

def visualize_graph(graph, landmarks):
    fig, ax = plt.subplots()

    for pos_id, pose in graph.positions.items():
        x, y = pose.measurement[0, 2], pose.measurement[1, 2]
        theta = mat_to_angle_2d(pose.measurement[:2, :2])
        dx = np.cos(theta) * 5
        dy = np.sin(theta) * 5
        
        ax.plot(x, y, 'bo')  # Plot position
        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.2, fc='blue', ec='blue')  # Plot orientation

    for lm_id, lm in landmarks.items():
        x, y = lm
        ax.plot(x, y, 'rx')  # Plot landmark

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Graph Visualization')
    ax.axis('equal')
    plt.grid()
    plt.show()

landmarks = {
    1: lm_1,
    2: lm_2,
    3: lm_3
}

# Visualize initial graph
visualize_graph(graph, landmarks)

# Perform state update
for i in range(100):
    slam.update_state(graph)
visualize_graph(graph, landmarks)
