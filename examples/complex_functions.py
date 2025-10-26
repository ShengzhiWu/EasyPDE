import numpy as np
import easypde

points = easypde.pointcloud.scatter_points_on_disk(400)

A = np.zeros((len(points), len(points)), dtype=np.complex64)
b = np.zeros(len(points), dtype=np.complex64)
weight_distribution_radius = easypde.pointcloud.get_typical_distance(points)*0.1
for i, point in enumerate(points):
    x = point[0]
    y = point[1]
    if x**2+y**2>0.999:  # On boundary
        easypde.edit_A_and_b(i, A, b, points, point, 5, [1, 0, 0, 0, 0, 0],
                             value=(x+y*1j)**2,
                             weight_distribution_radius=weight_distribution_radius,
                             dtype=np.complex64)
    else:  # Internal
        easypde.edit_A_and_b(i, A, b, points, point, 16, [0, 1, 1j, 0, 0, 0],
                             weight_distribution_radius=weight_distribution_radius,
                             dtype=np.complex64)

solution = np.linalg.solve(A, b)

print(f'mse = {np.sqrt(np.mean(np.square(np.abs(solution - (points[:, 0] + points[:, 1] * 1j) ** 2))))}')

easypde.plot_points(points, field=solution, color_map='complex_hsv')
