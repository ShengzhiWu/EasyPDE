import numpy as np

__version__ = "1.0"

class math:
    @classmethod
    def partitions(cls, total, number_of_parts):
        if number_of_parts==1:
            return [[total]]
        result = []
        for i in range(total, -1, -1):
            result += [[i]+e for e in cls.partitions(total-i, number_of_parts-1)]
        return result

    @classmethod
    def factorial(cls, n):
        result = 1
        for i in range(2, n+1):
            result *= i
        return result

    @classmethod
    def sigmoid(cls, a):
        return 1./(1.+np.exp(-a))
    
    @classmethod
    def orthogonize(cls, vectors):
        vectors = np.array(vectors)
        for i in range(len(vectors)):
            vectors[i] /= np.sqrt(np.sum(np.square(np.abs(vectors[i]))))
            for j in range(i+1, len(vectors)):
                vectors[j] -= vectors[i]*np.dot(vectors[j], np.conjugate(vectors[i]))
        return vectors
    
    @classmethod
    def make_orthogonal_basis(cls, vectors, guide_directions=None):
        vectors = [e for e in vectors]
        if not guide_directions is None:
            vectors += [e for e in guide_directions]
        while len(vectors)<len(vectors[0]):
            vectors.append(np.random.rand(len(vectors[0]))*2-1)
        return cls.orthogonize(vectors)
    
    @classmethod
    def get_perpendicular_subspace(cls, subspace, guide_directions=None):
        space = cls.make_orthogonal_basis(subspace, guide_directions=guide_directions)
        return space[len(subspace):]

def calculate_weights(A, b, sigma):
    A /= sigma
    solution = np.linalg.lstsq(A, b, rcond=None)[0]
    solution /= sigma
    return solution

def find_closest_points(points, point, n):
    return np.lexsort([np.sum(np.square(points-point), axis=1)])[: n]

def get_operator_order(dimension, order, axis_names=None):
    result = []
    for i in range(order+1):
        result += math.partitions(i, dimension)
        
    if not axis_names is None:
        for i in range(len(result)):
            s = ''
            for j, e in enumerate(result[i]):
                s += axis_names[j]*e
            result[i] = s
        
    return result

def generate_A(points, order=2, space_type='full', basis_of_subspace=None):
    if space_type=='full':
        result = []
        for i in range(order+1):
            for partition in math.partitions(i, points.shape[-1]):
                factor = 1
                result.append(np.ones(len(points)))
                for j, p in enumerate(partition):
                    result[-1] *= points[:, j]**p
                    factor /= math.factorial(p)
                result[-1] *= factor
        return np.array(result)
    
    if space_type=='subspace':
        return generate_A(points@basis_of_subspace.T, order=order)
    
    raise ValueError("Unknown space type: '"+space_type+"'.")

def edit_A(row: int, A, points, point, neighbor_n: int, operator, weight_decay_power: float=3, weight_distribution_radius: float=None, order=2, space_type="full", basis_of_subspace=None, dtype=np.float32):
    if weight_distribution_radius is None:
        weight_distribution_radius = pointcloud.get_typical_distance(points, samples=10)*0.1
    neighbors = pointcloud.find_closest_points(points, point, neighbor_n)
    neighbors_relative_location = points[neighbors]-point
    A_neighbors = generate_A(neighbors_relative_location, order=order, space_type=space_type, basis_of_subspace=basis_of_subspace)
    b_neighbors = np.array(operator, dtype=dtype)
    sigma_neighbors = np.sum(np.square(neighbors_relative_location), axis=1)**(weight_decay_power*0.5)+weight_distribution_radius**weight_decay_power
    A[row, neighbors] = calculate_weights(A_neighbors, b_neighbors, sigma_neighbors)

def edit_A_and_b(row: int, A, b, points, point, neighbor_n: int, operator, value: float=0, weight_decay_power: float=3, weight_distribution_radius: float=None, order=2, space_type="full", basis_of_subspace=None, dtype=np.float32):
    if weight_distribution_radius is None:
        weight_distribution_radius = pointcloud.get_typical_distance(points, samples=10)*0.1
    neighbors = pointcloud.find_closest_points(points, point, neighbor_n)
    neighbors_relative_location = points[neighbors]-point
    A_neighbors = generate_A(neighbors_relative_location, order=order, space_type=space_type, basis_of_subspace=basis_of_subspace)
    b_neighbors = np.array(operator, dtype=dtype)
    sigma_neighbors = np.sum(np.square(neighbors_relative_location), axis=1)**(weight_decay_power*0.5)+weight_distribution_radius**weight_decay_power
    A[row, neighbors] = calculate_weights(A_neighbors, b_neighbors, sigma_neighbors)
    b[row] = value

class pointcloud:

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
        factor = np.divide(math.sigmoid(sharpness/prefered_distance*(prefered_distance-distances)), distances, where=np.abs(distances)>prefered_distance*0.0001)
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
    
    @classmethod
    def scatter_points_on_rectangle(cls, n: int, width: float=1, height: float=1, iterations=30):
        boundary_points = []
        for t in np.linspace(0, width, int(np.sqrt(n*width/height)), endpoint=False):
            boundary_points.append([t, 0])
            boundary_points.append([width-t, height])
        for t in np.linspace(0, height, int(np.sqrt(n*height/width)), endpoint=False):
            boundary_points.append([width, t])
            boundary_points.append([0, height-t])
        internal_points = np.random.rand(n-len(boundary_points), 2)*np.array([width, height])
        def in_domain(points):
            return np.logical_and(np.logical_and(points[:, 0]>=0, points[:, 0]<=width), np.logical_and(points[:, 1]>=0, points[:, 1]<=height))
        return cls.relax_points_voronoi(boundary_points, internal_points, in_domain, iterations=iterations)

def hsv_to_rgb(hsv):
    rgb = np.zeros_like(hsv)
    h = hsv[:, 0]%1
    c = hsv[:, 1]*hsv[:, 2]
    x = c*(1-np.abs((h*6)%2-1))
    m = hsv[:, 2]-c
    condition = np.logical_and(h>=0, h<1/6)
    rgb[condition, 0] = c[condition]
    rgb[condition, 1] = x[condition]
    condition = np.logical_and(h>=1/6, h<2/6)
    rgb[condition, 1] = c[condition]
    rgb[condition, 0] = x[condition]
    condition = np.logical_and(h>=2/6, h<3/6)
    rgb[condition, 1] = c[condition]
    rgb[condition, 2] = x[condition]
    condition = np.logical_and(h>=3/6, h<4/6)
    rgb[condition, 2] = c[condition]
    rgb[condition, 1] = x[condition]
    condition = np.logical_and(h>=4/6, h<5/6)
    rgb[condition, 2] = c[condition]
    rgb[condition, 0] = x[condition]
    condition = np.logical_and(h>=5/6, h<=1)
    rgb[condition, 0] = c[condition]
    rgb[condition, 2] = x[condition]
    rgb += m.reshape([-1, 1])
    return rgb

def plot_points(points, field=None, point_size=None, adaptive_point_size=False, color_map=None):
    if points.shape[-1]==2:
        import matplotlib.pyplot as plt

        if adaptive_point_size:
            distance_to_neighbor = []
            for point in points:
                closest_points = points[pointcloud.find_closest_points(points, point, 2)[1]]
                distance_to_neighbor.append(np.sqrt(np.sum(np.square(closest_points-point))))
            distance_to_neighbor = np.array(distance_to_neighbor)
            point_size_factor = distance_to_neighbor**2
            point_size_factor /= np.mean(point_size_factor)
        
        if point_size is None:
            point_size = 6

        if color_map == "complex_hsv":
            plt.scatter(points[:, 0], points[:, 1], s=point_size**2 if not adaptive_point_size else point_size**2*point_size_factor, c=hsv_to_rgb(np.array([np.arctan2(np.imag(field), np.real(field))/(2*np.pi),
                                                                                         np.ones(len(field)),
                                                                                         np.abs(field)/np.max(np.abs(field))]).T))
        else:
            plt.scatter(points[:, 0], points[:, 1], s=point_size**2 if not adaptive_point_size else point_size**2*point_size_factor, c=field, cmap=color_map)
        plt.axis('equal')
        if not (field is None or color_map == "complex_hsv" or len(field.shape)!=1):
            plt.colorbar()
        plt.show()
    elif points.shape[-1]==3:
        import pyvista

        if point_size is None:
            point_size = 5

        if adaptive_point_size:
            print("Warning by EasyPDE: Option 'adaptive_point_size=True' is not applicable for 3D visualization.")

        particles = pyvista.PolyData(points)
        if not field is None:
            particles.point_data["color"] = field
        if (not field is None) and len(field.shape)==2 and field.shape[1]==3:
            particles.plot(notebook=False, point_size=point_size, rgb="color", render_points_as_spheres=False)
        else:
            particles.plot(notebook=False, point_size=point_size, cmap=color_map, render_points_as_spheres=False)
    else:
        raise ValueError("Only 2D or 3D point clouds are supported.")
