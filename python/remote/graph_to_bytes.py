import numpy as np
from tools import mat_to_angle_2d

def uint_to_bytes(value):
    return np.uint32(value).tobytes(order='C')
def float_to_bytes(value):
    return np.float32(value).tobytes(order='C')
def string_to_bytes(value):
    b = bytearray()
    b += uint_to_bytes(len(value))
    b.extend(map(ord, value))
    return b
def matrix_to_bytearray(matrix, is_diag = False):
    if len(matrix.shape) == 0:
        return None
    
    rows = b""
    cols = b""
    payload = b""

    if is_diag:
        rows = np.uint32(0).tobytes(order='C')
        cols = np.uint32(matrix.shape[0]).tobytes(order='C')
        for i in range(matrix.shape[0]):
            payload += float_to_bytes(matrix[i,i])
    else:    
        if len(matrix.shape) == 1:
            rows = np.uint32(0).tobytes(order='C')
            cols = np.uint32(matrix.shape[0]).tobytes(order='C')
        else:
            rows = np.uint32(matrix.shape[0]).tobytes(order='C')
            cols = np.uint32(matrix.shape[1]).tobytes(order='C')
        payload = matrix.astype(np.float32).tobytes(order='C')

    return rows + cols + payload

def graph_to_bytes(graph):
    def matrix_to_bytearray_pos(type, matrix):
        if type == 'pose2d':
            return  float_to_bytes(matrix[0, 2]) +\
                    float_to_bytes(matrix[1, 2]) +\
                    float_to_bytes(mat_to_angle_2d(matrix[:2,:2]))
        elif type == 'lm2d':
            return  float_to_bytes(matrix[0]) + float_to_bytes(matrix[1])
    
    total = b""
    
    vertices = graph.get_vertices()
    total += uint_to_bytes(len(vertices))
    for v_id in vertices:
        v = vertices[v_id]
        a = uint_to_bytes(v_id)
        b = string_to_bytes(v.get_type())
        c = matrix_to_bytearray_pos(v.get_type(), v.position)
        total += a + b + c

    edges = graph.get_edges()
    total += uint_to_bytes(len(edges))
    for e in edges:
        total += string_to_bytes(e.get_type())
        total += uint_to_bytes(e.id_1)
        total += uint_to_bytes(e.id_2)
        total += matrix_to_bytearray(e.measurement, False)
        total += matrix_to_bytearray(e.information, True)

    fixed_vertices = graph.get_fixed_vertices()
    total += uint_to_bytes(len(fixed_vertices))
    for v_id in fixed_vertices:
        total += uint_to_bytes(v_id)

    print("Total bytes: {}".format(len(total) + 4))
    return uint_to_bytes(len(total)) + total