import numpy as np
import easypde

points = np.mgrid[0:1:10j, 0:1:10j, 0:1:10j].T.reshape([-1, 3])

easypde.get_operators(3, 2, axis_names=['x', 'y', 'z'])
# Result: ['', 'x', 'y', 'z', 'xx', 'xy', 'xz', 'yy', 'yz', 'zz']

A = np.zeros((len(points), len(points)))
b = np.zeros(len(points))
weight_distribution_radius = easypde.pointcloud.get_typical_distance(points)*0.1
for i, point in enumerate(points):
    x = point[0]
    y = point[1]
    z = point[2]
    if x==0 or x==1 or y==0 or y==1 or z==0 or z==1:  # On boundary
        a = np.arctan2(x, y)
        easypde.edit_A_and_b(i, A, b, points, point, 7, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             weight_distribution_radius=weight_distribution_radius)
    else:  # Internal
        easypde.edit_A_and_b(i, A, b, points, point, 27, [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
                             value=1,
                             weight_distribution_radius=weight_distribution_radius)

solution = np.linalg.solve(A, b)

# Visualize solution on half of the box
easypde.plot_points(points[:500], field=solution[:500], point_size=17)
