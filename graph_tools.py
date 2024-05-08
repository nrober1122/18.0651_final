import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from colour import Color

def linear_interpolation_unit_spacing(wpt1: np.array, wpt2: np.array, odom_dist: float = 1.0) -> np.ndarray:
    direction = (wpt2 - wpt1)/la.norm(wpt2 - wpt1)
    pts = None
    
    for i, _ in enumerate(wpt1):
        if pts is None:
            pts = np.arange(wpt1[i], wpt2[i], odom_dist*direction[i])
        else:
            pts = np.vstack((pts, np.arange(wpt1[i], wpt2[i], odom_dist*direction[i])))

    return pts.T

def linear_interpolation(wpt1: np.array, wpt2: np.array, odom_dist: float = 1.0, final_wpt = False) -> np.ndarray:
    dist = la.norm(wpt2-wpt1)
    num_pts = int(np.ceil(dist/odom_dist)) # ensure odom measurements are taken at least every odom_dist units
    if final_wpt:
        num_pts += 1
    
    pts = np.linspace(wpt1, wpt2, num_pts, endpoint=final_wpt)

    return pts

# class used to do various functions relating to graph/laplacian generation
class Graph():
    def __init__(self, waypoints = None, odom_dist = 1, max_dist = 0.25) -> None:
        if waypoints is None:
            waypoints = np.array(
                [
                    [0, 0],
                    [10, 0],
                    [10, 10],
                    [0, 0],
                    [0, -10],
                    [10, -10],
                    [10, 0],
                    [0, 0],
                    [0, 10]
                ]
            )
        self.waypoints = waypoints
        self.max_dist = max_dist
        self.odom_dist = odom_dist
        self._generate_graph()

    # Generate a dict representation of a graph representation of the connected keypoints between waypoints
    def _generate_graph(self) -> np.ndarray:
        self.graph_keys = {}  # map index (for use in laplacian) -> coordinate
        self.graph_edges = {} # map index -> neighboring indices
        num_waypoints = len(self.waypoints)

        kpts = None
        final_waypoint = False

        # collect all keypoints together
        for i in range(num_waypoints-1):
            if i == num_waypoints - 2:
                final_waypoint = True

            if kpts is None:
                kpts = linear_interpolation(self.waypoints[i], self.waypoints[i+1], self.odom_dist)
            else:
                kpts = np.vstack((kpts, linear_interpolation(self.waypoints[i], self.waypoints[i+1], self.odom_dist, final_wpt=final_waypoint)))

        for i, kpt in enumerate(kpts):
            neighbors = [] 
            if i > 0:
                neighbors.append(i-1)
            if i < len(kpts) - 1:
                neighbors.append(i+1)

            self.add_node(i, kpt, neighbors)
    
    # Add a keypoint to the graph during initial generation
    def add_node(self, node_idx: int, node_pt: np.array, neighbors: list) -> None:
        existing_idxs = []

        for idx, existing_pt_tup in self.graph_keys.items():
            existing_pt = np.array(existing_pt_tup)
            dist = la.norm(node_pt - existing_pt)
            if dist < self.max_dist:
                node_pt = existing_pt
                existing_idxs.append(idx)
        
        self.graph_keys[node_idx] = node_pt
        self.graph_edges[node_idx] = []
        neighbors += existing_idxs
                
        for neighbor in neighbors:
            if not self._neighbor_in_list(node_idx, neighbor):
                self.graph_edges[node_idx].append(neighbor)
            if neighbor in self.graph_edges and not self._neighbor_in_list(neighbor, node_idx):
                self.graph_edges[neighbor].append(node_idx)

    # returns true if the candidate neighbor is already in the neighbor list
    def _neighbor_in_list(self, idx: int, idx_to_add: int) -> bool:
        return idx_to_add in self.graph_edges[idx]

    def degree_matrix(self) -> np.ndarray:
        num_pts = len(self.graph_edges)
        deg = np.zeros((num_pts, num_pts))

        indices = list(self.graph_keys.keys())

        for i in range(num_pts):
            idx = indices.index(i)
            deg[i, i] = len(self.graph_edges[idx])
        
        return deg
    
    def adjacency_matrix(self) -> np.ndarray:
        num_pts = len(self.graph_edges)
        adjacent = np.zeros((num_pts, num_pts))

        indices = list(self.graph_keys.keys())

        for i in range(num_pts):
            idx = indices.index(i)
            graph_key = idx

            neighbors = self.graph_edges[graph_key]
            for neighbor in neighbors:
                adjacent[idx, neighbor] = 1

        # import pdb; pdb.set_trace()
        return adjacent

    def laplacian(self) -> np.ndarray:
        return self.degree_matrix() - self.adjacency_matrix()
    
    def reduced_laplacian(self, idx: int = 0) -> np.ndarray:
        lap_mat = self.laplacian()
        return lap_mat[1:, 1:]
    
    # add edge to graph and calculate new determinant
    def add_edge(self, node1: int, node2: int) -> None:
        if not self._neighbor_in_list(node1, node2):
            self.graph_edges[node1].append(node2)
        if not self._neighbor_in_list(node2, node1):
            self.graph_edges[node2].append(node1)

    def plot(self):
        orange = Color("orange")
        num_segments = np.trace(self.degree_matrix())
        colors = list(orange.range_to(Color("purple"), int(num_segments)))
        seg_counter = 0
        delta = 0.1

        for idx in self.graph_keys:
            pt = np.array(self.graph_keys[idx])
            plt.scatter(pt[0], pt[1], c='k')
            plt.text(pt[0] + delta, pt[1] + delta, idx, size='small')
            
            for next_idx in self.graph_edges[idx]: # plot edges from pt
                next_pt = self.graph_keys[next_idx]
                pts = np.vstack((pt, next_pt)).T
                plt.plot(pts[0], pts[1], c=colors[seg_counter].hex)
                seg_counter += 1
        
        plt.gca().set_aspect('equal')
        plt.show()
        

if __name__ == "__main__":
    g = Graph(max_dist=0.5)
    lap = g.reduced_laplacian()
    print(la.det(lap))
    g.plot()

    g.add_edge(73, 20)
    lap = g.reduced_laplacian()
    print(la.det(lap))

    # import pdb; pdb.set_trace()
    g.plot()
