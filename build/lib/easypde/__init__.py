import numpy as np

__version__ = "1.0"

def calculate_weights(A, b, sigma):
    A /= sigma
    solution = np.linalg.lstsq(A, b, rcond=None)[0]
    solution /= sigma
    return solution

def find_closest_points(points, point, n):
    return np.lexsort([np.sum(np.square(points-point), axis=1)])[: n]

def generate_A(points, order=2, space_type="2d"):
    if space_type=="2d":
        x = points[:, 0]
        y = points[:, 1]
        if order==1:
            return np.array([
                np.ones_like(x),
                x, y
            ])
        elif order==2:
            return np.array([
                np.ones_like(x),
                x, y,
                x**2*0.5,
                x*y,
                y**2*0.5
            ])
        elif order==3:
            return np.array([
                np.ones_like(x),
                x, y,
                x**2*0.5,
                x*y,
                y**2*0.5,
                x*x*x/6,
                x*x*y*0.5,
                x*y*y*0.5,
                y*y*y/6
            ])
    elif space_type=="3d":
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        if order==2:
            return np.array([
                [1.]*len(points),
                 x, y, z,
                 x**2*0.5,
                 y**2*0.5,
                 z**2*0.5,
                 x*y,
                 y*z,
                 z*x
            ])
    
    return None

def edit_A(row: int, A, points, point, neighbor_n: int, operator, weight_decay_power: float=3, weight_distribution_radius: float=None, order=2, space_type="2d"):
    if weight_distribution_radius is None:
        weight_distribution_radius = pointcloud.get_typical_distance(points, samples=10)*0.1
    neighbors = pointcloud.find_closest_points(points, point, neighbor_n)
    neighbors_relative_location = points[neighbors]-point
    A_neighbors = generate_A(neighbors_relative_location, order=order, space_type=space_type)
    b_neighbors = np.array(operator, dtype=np.float32)
    sigma_neighbors = np.sum(np.square(neighbors_relative_location), axis=1)**(weight_decay_power*0.5)+weight_distribution_radius**weight_decay_power
    A[row, neighbors] = calculate_weights(A_neighbors, b_neighbors, sigma_neighbors)

def edit_A_and_b(row: int, A, b, points, point, neighbor_n: int, operator, value: float=0, weight_decay_power: float=3, weight_distribution_radius: float=None, order=2, space_type="2d"):
    if weight_distribution_radius is None:
        weight_distribution_radius = pointcloud.get_typical_distance(points, samples=10)*0.1
    neighbors = pointcloud.find_closest_points(points, point, neighbor_n)
    neighbors_relative_location = points[neighbors]-point
    A_neighbors = generate_A(neighbors_relative_location, order=order, space_type=space_type)
    b_neighbors = np.array(operator, dtype=np.float32)
    sigma_neighbors = np.sum(np.square(neighbors_relative_location), axis=1)**(weight_decay_power*0.5)+weight_distribution_radius**weight_decay_power
    A[row, neighbors] = calculate_weights(A_neighbors, b_neighbors, sigma_neighbors)
    b[row] = value

class pointcloud:
    @classmethod
    def sigmoid(cls, a):
        return 1./(1.+np.exp(-a))

    @classmethod
    def find_closest_points(cls, points, point, n):
        return np.lexsort([np.sum(np.square(points-point), axis=1)])[: n]
    
    @classmethod
    def get_typical_distance(cls, points, neighbors=3, samples=None, random_sample=False):
        if samples is None or samples>=len(points):
            point_samples = points
        else:
            if random_sample:
                point_samples = points[np.random.choice(len(points), samples, replace=False)]
            else:
                point_samples = points[-samples:]
        a = 0
        for point in point_samples:
            closest_points = points[cls.find_closest_points(points, point, neighbors+1)[1:]]
            a += np.mean(np.sqrt(np.sum(np.square(closest_points-point), axis=-1)))
        return a/len(point_samples)

    @classmethod
    def repulsive_force(cls, points1, points2, prefered_distance, sharpness):  # 计算第一组点受到的第二组点的斥力，返回向量模长在0和1之间
        v = np.reshape(points1, (-1, 1, points1.shape[-1]))-points2
        distances = np.sqrt(np.sum(np.square(v), axis=-1))
        factor = np.divide(cls.sigmoid(sharpness/prefered_distance*(prefered_distance-distances)), distances, where=np.abs(distances)>prefered_distance*0.0001)
        forces = np.sum(np.transpose(v, axes=(2, 0, 1))*factor, axis=2).T
        return forces
    
    @classmethod
    def relax_points(cls, fixed_points, movable_points, prefered_distance: float, iterations: int=30, repulse_strength: float=None):
        if repulse_strength is None:
            repulse_strength = prefered_distance*0.1
        points = np.concatenate([fixed_points, movable_points])
        for i in range(iterations):
            points[len(fixed_points):] += cls.repulsive_force(points[len(fixed_points):], points, prefered_distance, 10.)*repulse_strength
        return points
    
    @classmethod
    def center_of_convex_polygon(cls, vertices):
        area_total = 0
        a = vertices[0]
        v = np.zeros_like(vertices[0])
        for b, c in zip(vertices[1:-1], vertices[2:]):
            area = np.abs(np.cross(b-a, c-a))*0.5
            area_total += area
            v += area/3*(a+b+c)
        return v/area_total
    
    @classmethod
    def relax_points_voronoi(cls, boundary_points, internal_points, in_domain_function, iterations: int=30):
        from scipy.spatial import Voronoi

        points = np.concatenate([boundary_points, internal_points])
        for step in range(iterations):
            vor = Voronoi(points)
            for i, j in enumerate(vor.point_region):
                if i<len(boundary_points):
                    continue
                points[i] = cls.center_of_convex_polygon(vor.vertices[vor.regions[j]])
            internal_points = points[len(boundary_points):]
            points = np.concatenate([boundary_points, internal_points[in_domain_function(internal_points)]])
        return points
    
    @classmethod
    def scatter_points_on_disk(cls, n: int, r: float=1, iterations=30):
        a = np.linspace(0, 2*np.pi, int(np.sqrt(n)*3.), endpoint=False)
        boundary_points = np.array([np.cos(a), np.sin(a)]).T
        internal_points = np.random.rand(int((n-len(boundary_points))*4/np.pi), 2)*2-1
        internal_points = internal_points[np.sum(np.square(internal_points), axis=-1)<1]
        def in_domain(points):
            return np.sum(np.square(points), axis=-1)<1
        return cls.relax_points_voronoi(boundary_points, internal_points, in_domain, iterations=iterations)*r
    
    @classmethod
    def scatter_points_on_square(cls, n: int, size: float=1, iterations=30):
        boundary_points = []
        for t in np.linspace(0, 1, int(np.sqrt(n)), endpoint=False):
            boundary_points.append([t, 0])
            boundary_points.append([1, t])
            boundary_points.append([1-t, 1])
            boundary_points.append([0, 1-t])
        internal_points = np.random.rand(n-len(boundary_points), 2)
        def in_domain(points):
            return np.logical_and(np.logical_and(points[:, 0]>=0, points[:, 0]<=1), np.logical_and(points[:, 1]>=0, points[:, 1]<=1))
        return cls.relax_points_voronoi(boundary_points, internal_points, in_domain, iterations=iterations)*size

def plot_points(points, field=None, point_size=None):
    if points.shape[-1]==2:
        import matplotlib.pyplot as plt

        plt.scatter(points[:, 0], points[:, 1], s=point_size, c=field)
        plt.axis('equal')
        if not field is None:
            plt.colorbar()
        plt.show()
    elif points.shape[-1]==3:
        import pyvista

        particles = pyvista.PolyData(points)
        if not field is None:
            particles.point_data["color"] = field
        particles.plot(notebook=False, point_size=point_size, render_points_as_spheres=False)
