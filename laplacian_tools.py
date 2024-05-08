import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from graph_tools import Graph

class Laplacian_Handler():
    def __init__(self, graph) -> None:
        self.graph = graph
        
        self.lap = self.graph.reduced_laplacian()
        self.lap_det = la.det(self.lap)
        self.lap_inv = la.inv(self.lap)

    def add_edge(self, node1: int, node2: int) -> float:
        self.graph.add_edge(node1, node2)
        self.lap_det = self.simulate_add_edge(node1, node2)
        self.lap = self.graph.reduced_laplacian()
        self.lap_inv = la.inv(self.lap) # TODO: use rank one update instead of inverting

        return self.lap_det
    
    def simulate_add_edge(self, node1: int, node2: int) -> float:
        # column of reduced incidence matrix
        a_uv = np.zeros(len(self.graph.graph_keys) - 1) 
        # if either of the nodes are 0, a_uv contains only one 1
        if node1 == 0:
            a_uv[node2 - 1] = 1
        elif node2 == 0:
            a_uv[node1 - 1] = 1
        else:
            a_uv[node1 - 1] = -1
            a_uv[node2 - 1] = 1

        det_prior = self.lap_det
        # Linv_a = la.solve(self.lap, a_uv)
        # lap_det = det_prior*(1 + np.inner(a_uv, Linv_a))
        lap_det = det_prior*(1 + np.inner(a_uv, np.dot(self.lap_inv, a_uv)))
        return lap_det
        


if __name__ == '__main__':
    g = Graph()
    l = Laplacian_Handler(g)
    print(l.lap_det)

    print(l.add_edge(73, 20))

