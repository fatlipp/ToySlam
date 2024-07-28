import numpy as np

def load_env():
    """
    return obstacles pos and radius of each point
    *****
    *   *
    *****
    """
    # wall1
    SCALE = 1
    SIZE = 30 * SCALE
    WALL = 4 * SCALE
    SS = np.array([SIZE, SIZE, 0])
    w1 = SS + np.array([(x, SIZE, 0) for x in range(-SIZE * 2, SIZE* 2)])
    w2 = SS + np.array([(x, -SIZE, 0) for x in range(-SIZE * 2, SIZE * 2)])
    w3 = SS + np.array([(-SIZE, y, 0) for y in range(-SIZE, SIZE)])
    w4 = SS + np.array([(SIZE, y, 0) for y in range(-SIZE, SIZE)])

    b1_1 = SS + np.array([(x, SIZE - WALL, 0) for x in range(0, SIZE - WALL)])
    b1_2 = SS + np.array([(0, y, 0) for y in range(SIZE - (WALL - 1), SIZE)])
    b1_3 = SS + np.array([(SIZE - WALL, y, 0) for y in range(0, SIZE - (WALL - 1))])
    b1_4 = SS + np.array([(x, 0, 0) for x in range(SIZE - (WALL - 1), SIZE)])

    return np.concatenate([ w1, w2, w3, w4, b1_1, b1_2, b1_3, b1_4]).reshape(-1, 3) / SCALE, 0.25 / SCALE
    # return np.array([[15, 25, 0], [35, 10, 0], [35, 15, 0], [25, 5, 0], 
    #                  [25, 35, 0], [25, 5, 0], [30, 45, 0], [15, 0, 0],
    #                  [5, 35, 0], [15, 25, 0], [35, 35, 0], [15, 5, 0]]), 0.5 / SCALE

def load_env_grid():
    # width x height
    shape = (21, 21, 21)
    grid = np.zeros(shape)
    for y in range(0, shape[0]):
        grid[y, 0] = 1
        grid[y, shape[1] - 1] = 1
    for x in range(0, shape[1]):
        grid[0, x] = 1
        grid[shape[0] - 1, x] = 1
    return grid, shape