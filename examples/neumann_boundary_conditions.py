import numpy as np
import easypde

points = easypde.pointcloud.scatter_points_on_disk(1000)

A = np.zeros((len(points), len(points)))
b = np.zeros(len(points))
weight_distribution_radius = easypde.pointcloud.get_typical_distance(points) * 0.1
for i, point in enumerate(points[:-1]):  # The last point skiped.
    x = point[0]
    y = point[1]
    if x ** 2 + y ** 2 > 0.999:  # On boundary
        a = np.arctan2(x, y)
        easypde.edit_A_and_b(i, A, b, points, point, 5, [0, x, y, 0, 0, 0],
                             weight_distribution_radius=weight_distribution_radius)
    else:  # Internal
        easypde.edit_A_and_b(i, A, b, points, point, 16, [0, 0, 0, 1, 0, 1],
                             value=np.sin(15 * x),
                             weight_distribution_radius=weight_distribution_radius)
A[-1] = np.ones_like(A[-1])

solution = np.linalg.solve(A, b)

easypde.plot_points(points, field=solution)
