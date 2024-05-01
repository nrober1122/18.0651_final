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
        self.generate_graph()
    
    def calculate_adjacency(self) -> np.ndarray:
        print('Not done yet')

    # Generate a dict representation of a graph representation of the connected keypoints between waypoints
    def generate_graph(self) -> np.ndarray:
        self.graph = {} # map coordinate -> neighboring coordinate
        self.graph_keys = {} # map coordinate -> index (for use in laplacian)
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
            self.add_node(i, kpt, kpts)
        
    
    # Add a keypoint to the graph during initial generation
    def add_node(self, index: int, pt: np.array, kpts: np.ndarray) -> None:
        num_kpts = len(kpts)
        added = False

        for existing_pt_tup in self.graph:
            existing_pt = np.array(existing_pt_tup)
            dist = la.norm(pt - existing_pt)
            if dist < self.max_dist:
                pt = existing_pt
                added = True
        
        if not added:
            self.graph_keys[tuple(pt)] = len(self.graph)
            self.graph[tuple(pt)] = []
                
        
        if index > 0:
            if not tuple(kpts[index-1]) in [tuple(arr) for arr in self.graph[tuple(pt)]]:
                self.graph[tuple(pt)].append(kpts[index-1])
        if index < num_kpts - 1:
            if not tuple(kpts[index+1]) in [tuple(arr) for arr in self.graph[tuple(pt)]]:
                self.graph[tuple(pt)].append(kpts[index+1])

    def degree_matrix(self) -> np.ndarray:
        num_pts = len(self.graph)
        deg = np.zeros((num_pts, num_pts))

        indices = list(self.graph_keys.values())
        coords = list(self.graph_keys.keys())

        for i in range(num_pts):
            idx = indices.index(i)
            deg[i, i] = len(self.graph[coords[idx]])
        
        return deg
    
    def adjacency_matrix(self) -> np.ndarray:
        num_pts = len(self.graph)
        adjacent = np.zeros((num_pts, num_pts))

        indices = list(self.graph_keys.values())
        coords = list(self.graph_keys.keys())

        for i in range(num_pts):
            idx = indices.index(i)
            graph_key = coords[idx]

            neighbors = [tuple(pt) for pt in self.graph[graph_key]]
            for neighbor in neighbors:
                idx_n = coords.index(neighbor)
                adjacent[idx, idx_n] = 1

        # import pdb; pdb.set_trace()
        return adjacent

    def laplacian(self) -> np.ndarray:
        return self.degree_matrix() - self.adjacency_matrix()

    def plot(self):
        orange = Color("orange")
        num_segments = np.trace(self.degree_matrix())
        colors = list(orange.range_to(Color("purple"), int(num_segments)))
        seg_counter = 0

        for key in self.graph:
            pt = np.array(key)
            plt.scatter(pt[0], pt[1], c='k')
            for next_pt in self.graph[key]:
                pts = np.vstack((pt, next_pt)).T
                plt.plot(pts[0], pts[1], c=colors[seg_counter].hex)
                seg_counter += 1
        
        plt.gca().set_aspect('equal')
        plt.show()

if __name__ == "__main__":
    g = Graph(max_dist=0.5)
    lap = g.laplacian()
    g.plot()
