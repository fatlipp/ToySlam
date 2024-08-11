class OptGraph:
    def __init__(self):
        self.vertices = {}
        self.edges = []
        self.fixed_vertices = set()

    def add_vertex(self, id, vertex, fixed=False):
        self.vertices[id] = vertex
        if fixed:
            self.fix_vertex(id)

    def fix_vertex(self, id):
        if id not in self.vertices:
            raise RuntimeError("Fix LM: {} is not found".format(id))
        self.fixed_vertices.add(id)

    def add_edge(self, edge):
        self.edges.append(edge)

    def get_vertex(self, id):
        if id not in self.vertices:
            raise RuntimeError("get_position() {} is not found".format(id))
        return self.vertices[id]
    
    def get_vertices(self):
        return self.vertices
    
    def get_edges(self):
        return self.edges
    
    def get_fixed_vertices(self):
        return self.fixed_vertices