import functools

import numpy as np
import trimesh


def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def scale_to_unit_cube(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / np.max(mesh.bounding_box.extents)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def scale_point_cloud_to_unit_sphere(point_cloud):
    # Note:  Don't center the point cloud on the centroid of all its points since we know the center is already set up
    # from the mocap markers, and the observed points might be biased to uncenter it.
    # # Center the point cloud on the centroid of the points on the point cloud.
    # point_cloud = point_cloud - np.average(point_cloud, axis=0)

    # Find the largest radius of the points on the object and normalize by it.
    distances = np.linalg.norm(point_cloud, axis=1)
    point_cloud /= np.max(distances)
    
    return point_cloud

def scale_point_cloud_to_unit_cube(point_cloud):
    # Note:  Don't center the point cloud on the centroid of all its points since we know the center is already set up
    # from the mocap markers, and the observed points might be biased to uncenter it.
    # # Center the point cloud on the centroid of the points on the point cloud.
    # point_cloud = point_cloud - np.average(point_cloud, axis=0)

    # Divide all point locations by the maximum x-, y-, or z-length of the cloud.
    max_length = np.max(np.max(point_cloud, axis=0) - np.min(point_cloud, axis=0))
    point_cloud *= 2 / max_length
    
    return point_cloud

def get_into_n_by_3_shape(array):
    assert 3 in array.shape, print(f'Cannot put array of shape {array.shape} into (n, 3) shape.')
    assert array.ndim == 2, print(f'Only set up to handle 2-dimensional arrays, given {array.shape} instead.')

    if array.shape[0] == 3:
        array = np.transpose(array, axes=(1, 0))

    return array

# Use get_raster_points.cache_clear() to clear the cache
@functools.lru_cache(maxsize=4)
def get_raster_points(voxel_resolution):
    points = np.meshgrid(
        np.linspace(-1, 1, voxel_resolution),
        np.linspace(-1, 1, voxel_resolution),
        np.linspace(-1, 1, voxel_resolution)
    )
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose().astype(np.float32)
    return points

def check_voxels(voxels):
    block = voxels[:-1, :-1, :-1]
    d1 = (block - voxels[1:, :-1, :-1]).reshape(-1)
    d2 = (block - voxels[:-1, 1:, :-1]).reshape(-1)
    d3 = (block - voxels[:-1, :-1, 1:]).reshape(-1)

    max_distance = max(np.max(d1), np.max(d2), np.max(d3))
    return max_distance < 2.0 / voxels.shape[0] * 3**0.5 * 1.1

def sample_uniform_points_in_unit_sphere(amount):
    unit_sphere_points = np.random.uniform(-1, 1, size=(amount * 2 + 20, 3))
    unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]

    points_available = unit_sphere_points.shape[0]
    if points_available < amount:
        # This is a fallback for the rare case that too few points are inside the unit sphere
        result = np.zeros((amount, 3))
        result[:points_available, :] = unit_sphere_points
        result[points_available:, :] = sample_uniform_points_in_unit_sphere(amount - points_available)
        return result
    else:
        return unit_sphere_points[:amount, :]
