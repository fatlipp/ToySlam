import numpy as np
from optimizer.edges2d import *
from optimizer.vertices import *
from optimizer.opt_graph import OptGraph

def bytes_to_uint(b, offset=0):
    return int(np.frombuffer(b[offset:offset+4], dtype=np.uint32)[0])

def bytes_to_float(b, offset=0):
    return float(np.frombuffer(b[offset:offset+4], dtype=np.float32)[0])

def bytes_to_string(b, offset=0):
    length = bytes_to_uint(b, offset)
    return ''.join(map(chr, b[offset+4:offset+4+length]))

def bytearray_to_matrix(b, offset, is_diag=False):
    rows = bytes_to_uint(b, offset)
    cols = bytes_to_uint(b, offset + 4)
    offset += 8
    
    if is_diag:
        matrix = np.zeros((cols, cols), dtype=np.float32)
        for i in range(cols):
            matrix[i, i] = bytes_to_float(b, offset)
            offset += 4
    else:
        matrix = np.frombuffer(b[offset:offset+4*rows*cols], dtype=np.float32)
        matrix = matrix.reshape((rows, cols))
        offset += 4 * rows * cols
    
    return matrix, offset

def bytearray_to_matrix_pos(type, b, offset):
    if type == 'pose2d':
        x = bytes_to_float(b, offset)
        y = bytes_to_float(b, offset + 4)
        theta = bytes_to_float(b, offset + 8)
        matrix = np.array([[np.cos(theta), -np.sin(theta), x],
                           [np.sin(theta), np.cos(theta), y],
                           [0, 0, 1]], dtype=np.float32)
        offset += 12
    elif type == 'lm2d':
        x = bytes_to_float(b, offset)
        y = bytes_to_float(b, offset + 4)
        matrix = np.array([x, y], dtype=np.float32)
        offset += 8
    return matrix, offset

def bytes_to_graph(b):
    offset = 0
    
    num_vertices = bytes_to_uint(b, offset)
    offset += 4
    
    vertices = {}
    for _ in range(num_vertices):
        v_id = bytes_to_uint(b, offset)
        offset += 4
        v_type = bytes_to_string(b, offset)
        offset += 4 + len(v_type)
        pos, offset = bytearray_to_matrix_pos(v_type, b, offset)

        if v_type == 'pose2d':
            vertex = VertexPose2d(pos)
        if v_type == 'lm2d':
            vertex = Vertex2d(pos)
        vertices[v_id] = vertex
    
    num_edges = bytes_to_uint(b, offset)
    offset += 4
    
    edges = []
    for _ in range(num_edges):
        e_type = bytes_to_string(b, offset)
        offset += 4 + len(e_type)
        id_1 = bytes_to_uint(b, offset)
        offset += 4
        id_2 = bytes_to_uint(b, offset)
        offset += 4
        measurement, offset = bytearray_to_matrix(b, offset, False)
        information, offset = bytearray_to_matrix(b, offset, True)
        if e_type == 'se2':
            edge = EdgeOdometry2d(id_1, id_2, measurement, information)
        if e_type == 'se2point2':
            edge = EdgeLandmark2d(id_1, id_2, measurement, information)
        edges.append(edge)
        # print("EDGE: ", id_1, id_2, e_type, "meas: ", edge.measurement, "\n INF:", edge.information)
    
    num_fixed_vertices = bytes_to_uint(b, offset)
    offset += 4
    
    fixed_vertices = []
    for _ in range(num_fixed_vertices):
        v_id = bytes_to_uint(b, offset)
        offset += 4
        fixed_vertices.append(v_id)
    
    graph = OptGraph()
    for v_id, vertex in vertices.items():
        # print("VERTEX: ", v_id, vertex.position)
        graph.add_vertex(v_id, vertex)

    
    for edge in edges:
        graph.add_edge(edge)
    
    for v_id in fixed_vertices:
        graph.fix_vertex(v_id)
    
    return graph