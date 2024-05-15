import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from graph_tools import Graph, linear_interpolation

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
    
    def simulate_traveling_to_waypoint(self, waypoint: np.ndarray) -> float:
        lc_node = None
        # find the index of the waypoint in the graph_keys
        for idx, pt in self.graph.graph_keys.items():
            if la.norm(pt - waypoint) < self.graph.max_dist:
                lc_node = idx
                break
            
        assert lc_node is not None, "Waypoint not found in graph"
        
        # compute number of odometry points that need to be added
        n = len(self.graph.graph_keys)
        num_odom_to_add = len(linear_interpolation(self.graph.graph_keys[n-1], self.graph.graph_keys[lc_node], self.graph.odom_dist, True)) - 1
        
        # Only up to 3 elements of L_inv are needed to compute the determinant
        Linv_mult = {} # maps idx --> a multiplier
        
        Linv_mult[tuple([n+num_odom_to_add-2, n+num_odom_to_add-2])] = 1
        if lc_node != 0:
            Linv_mult[tuple([n+num_odom_to_add-2, lc_node - 1])] = -2
            Linv_mult[tuple([lc_node - 1, lc_node - 1])] = 1
            
        aTLia = 0.0 # a.T @ L_inv @ a
        for idx, weight in Linv_mult.items():
            if idx[0] < n-1 and idx[1] < n-1:
                aTLia += weight * self.lap_inv[idx[0], idx[1]]
            elif idx[0] < n-1 or idx[1] < n-1:
                if idx[0] >= n-1:
                    aTLia += weight * self.lap_inv[n-2, idx[1]]
                elif idx[1] >= n-1:
                    aTLia += weight * self.lap_inv[idx[0], n-2]
            else:
                assert idx[0] == n+num_odom_to_add-2 and idx[1] == n+num_odom_to_add-2
                aTLia += weight * (num_odom_to_add + self.lap_inv[-1,-1])
            
        return self.lap_det*(1 + aTLia)
    
    def num_odom_to_waypoint(self, waypoint: np.ndarray) -> int:
        lc_node = None
        # find the index of the waypoint in the graph_keys
        for idx, pt in self.graph.graph_keys.items():
            if la.norm(pt - waypoint) < self.graph.max_dist:
                lc_node = idx
                break
            
        assert lc_node is not None, "Waypoint not found in graph"
        
        # compute number of odometry points that need to be added
        n = len(self.graph.graph_keys)
        num_odom_to_add = len(linear_interpolation(self.graph.graph_keys[n-1], self.graph.graph_keys[lc_node], self.graph.odom_dist, True)) - 1
        return num_odom_to_add
            
        
        
        # # column of reduced incidence matrix
        # a_uv = np.zeros(len(self.graph.graph_keys) - 1) 
        # a_uv[waypoint - 1] = 1

        # det_prior = self.lap_det
        # # Linv_a = la.solve(self.lap, a_uv)
        # # lap_det = det_prior*(1 + np.inner(a_uv, Linv_a))
        # lap_det = det_prior*(1 + np.inner(a_uv, np.dot(self.lap_inv, a_uv)))
        # return lap_det
        


if __name__ == '__main__':
    g = Graph()
    l = Laplacian_Handler(g)
    print(l.lap_det)

    print(l.add_edge(73, 20))

