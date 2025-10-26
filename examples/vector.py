import numpy as np
import easypde

np.random.seed(0)
points = easypde.pointcloud.scatter_points_on_disk(1000)

source_r = 0.25
source_blur = 0.02

# Prepare matrix
A = np.zeros((len(points)*2, len(points)*2))  # The vector field has 2 components, so sizes are multiplied by 2.
b = np.zeros(len(points)*2)
weight_distribution_radius = easypde.pointcloud.get_typical_distance(points)*0.1
for i, point in enumerate(points):
    r = np.sqrt(point[0]**2+point[1]**2)
    div = 1-max(0, min(1, (r-(source_r-source_blur))/(source_blur*2)))
    
    neighbors = easypde.pointcloud.find_closest_points(points, point, 16)
    # Here we precalculate the neighbors because this data will be used several times. This can save some time.
    
    easypde.edit_A_and_b(i, A, b, points, point, 16, [0, 1, 0, 0, 0, 0], value=div, neighbors=neighbors, row_channel=0, column_channel=0,
                         weight_distribution_radius=weight_distribution_radius)
    easypde.edit_A_and_b(i, A, b, points, point, 16, [0, 0, 1, 0, 0, 0], neighbors=neighbors, row_channel=0, column_channel=1,
                         weight_distribution_radius=weight_distribution_radius)
    easypde.edit_A_and_b(i, A, b, points, point, 16, [0, 0, 1, 0, 0, 0], neighbors=neighbors, row_channel=1, column_channel=0,
                         weight_distribution_radius=weight_distribution_radius)
    easypde.edit_A_and_b(i, A, b, points, point, 16, [0, -1, 0, 0, 0, 0], neighbors=neighbors, row_channel=1, column_channel=1,
                         weight_distribution_radius=weight_distribution_radius)

# Solve
solution = np.linalg.lstsq(A, b, rcond=1/40)[0]  # Calculate the least-squares solution
# Here we use np.linalg.lstsq instead of np.linalg.solve because the matrix is singular.
# The rcond parameter is critical. See NumPy's document about np.linalg.lstsq for more details.

#  Error analysis
r = np.sqrt(np.sum(np.square(points), axis=-1))
ground_truth =  np.array(points)
ground_truth = ground_truth.T
ground_truth[:, r >= source_r] *= r[r >= source_r] ** -2 * (np.pi * source_r ** 2 / 2 / np.pi)
ground_truth[:, r < source_r] *= (np.pi * source_r ** 2 / 2 / np.pi) / source_r / source_r
ground_truth = ground_truth.T
ground_truth = ground_truth.T.flatten()
print(f'mse = {np.sqrt(np.mean(np.square(solution - ground_truth)))}')
# Result on my machine: error = 0.006351124477499586.

# Visualize
easypde.plot_points(points, field=solution[:len(points)])  # Visualize the first component, namely E_x.
