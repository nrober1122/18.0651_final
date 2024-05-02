import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from graph_tools import Graph

class Laplacian_Handler():
    def __init__(self, waypoints = None, odom_dist = 1, max_dist = 0.25) -> None:
        self.graph = Graph(waypoints, odom_dist, max_dist)
        
        self.lap = self.graph.reduced_laplacian()
        self.lap_det = la.det(self.lap)

    def add_edge(self, node1: int, node2: int) -> float:
        self.graph.add_edge(node1, node2)

        # column of reduced incidence matrix
        a_uv = np.zeros(len(self.graph.graph_dict) - 1) 
        a_uv[[node1 - 1, node2 - 1]] = 1 # TODO Same deal, not sure what the indices should be after reducing

        det_prior = self.lap_det
        Linv_a = la.solve(self.lap, a_uv)
        self.lap_det = det_prior*(1 + np.inner(a_uv, Linv_a))

        return self.lap_det

if __name__ == '__main__':
    l = Laplacian_Handler()
    print(l.lap_det)

    print(l.add_edge(73, 20))

