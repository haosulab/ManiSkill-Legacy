import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------- #
# Convert in opne3d
# ---------------------------------------------------------------------------- #


def merge_mesh(meshes):
    # Merge without color and normal
    vertices = np.zeros((0, 3))
    triangles = np.zeros((0, 3))

    for mesh in meshes:
        vertices_i = np.asarray(mesh.vertices)
        triangles_i = np.asarray(mesh.triangles)
        triangles_i += vertices.shape[0]
        vertices = np.append(vertices, vertices_i, axis=0)
        triangles = np.append(triangles, triangles_i, axis=0)

    vertices = o3d.utility.Vector3dVector(vertices)
    triangles = o3d.utility.Vector3iVector(triangles)
    # print(vertices, triangles)
    # exit(0)
    mesh = o3d.geometry.TriangleMesh(vertices, triangles)
    mesh.compute_vertex_normals(normalized=True)
    mesh.compute_triangle_normals(normalized=True)
    return mesh



def mesh2pcd(mesh, sample_density, num_points=None):
    pcd_tmp = mesh.sample_points_uniformly(number_of_points=sample_density)
    points = np.asarray(pcd_tmp.points)
    normals = np.asarray(pcd_tmp.normals)

    pcd = o3d.geometry.PointCloud()
    if num_points:
        idx = np.arange(points.shape[0])
        np.random.shuffle(idx)
        idx = idx[:num_points]
        points = points[idx]
        normals = normals[idx]
    #print(vertices.shape, normals.shape)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


# ---------------------------------------------------------------------------- #
# Build from numpy
# ---------------------------------------------------------------------------- #

def np2mesh(vertices, triangles, colors=None, vertex_normals=None, triangle_normals=None):
    """Convert numpy array to open3d PointCloud."""
    # print(vertices, triangles)(
    # print(vertices.dtype, vertices.shape)
    vertices = o3d.utility.Vector3dVector(vertices)
    triangles = o3d.utility.Vector3iVector(triangles)
    # print(vertices, triangles)
    # exit(0)
    mesh = o3d.geometry.TriangleMesh(vertices, triangles)
    if colors is not None:
        colors = np.array(colors)
        if colors.ndim == 2:
            assert len(colors) == len(vertices)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(vertices), 1))
        else:
            raise RuntimeError(colors.shape)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    if vertex_normals is not None:
        assert len(triangles) == len(vertex_normals)
        mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
    else:
        mesh.compute_vertex_normals(normalized=True)

    if triangle_normals is not None:
        assert len(triangles) == len(triangle_normals)
        mesh.triangle_normals = o3d.utility.Vector3dVector(triangle_normals)
    else:
        mesh.compute_triangle_normals(normalized=True)
    return mesh


def np2pcd(points, colors=None, normals=None):
    """Convert numpy array to open3d PointCloud."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.array(colors)
        if colors.ndim == 2:
            assert len(colors) == len(points)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(points), 1))
        else:
            raise RuntimeError(colors.shape)
        pc.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        assert len(points) == len(normals)
        pc.normals = o3d.utility.Vector3dVector(normals)
    return pc


def create_aabb(bbox, color=(0, 1, 0)):
    """Draw an axis-aligned bounding box."""
    assert len(bbox) == 6, f'The format of bbox should be xyzwlh, but received {len(bbox)}.'
    bbox = np.asarray(bbox)
    abb = o3d.geometry.AxisAlignedBoundingBox(bbox[0:3] - bbox[3:6] * 0.5, bbox[0:3] + bbox[3:6] * 0.5)
    abb.color = color
    return abb


def create_aabb_from_pcd(pcd, color=(0, 1, 0)):
    """Draw an axis-aligned bounding box."""
    assert pcd.shape[-1] == 3, f'The format of bbox should be xyzwlh, but received {pcd.shape}.'
    abb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pcd))
    abb.color = color
    return abb


def create_obb(bbox, R, color=(0, 1, 0)):
    """Draw an oriented bounding box."""
    assert len(bbox) == 6, f'The format of bbox should be xyzwlh, but received {len(bbox)}.'
    obb = o3d.geometry.OrientedBoundingBox(bbox[0:3], R, bbox[3:6])
    obb.color = color
    return obb


def create_obb_from_pcd(pcd, color=(0, 1, 0)):
    """Draw an axis-aligned bounding box."""
    assert pcd.shape[-1] == 3, f'The format of bbox should be xyzwlh, but received {pcd.shape}.'
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pcd))
    obb.color = color
    return obb


# ---------------------------------------------------------------------------- #
# Computation
# ---------------------------------------------------------------------------- #
def compute_pcd_normals(points, search_param=None, camera_location=(0.0, 0.0, 0.0)):
    """Compute normals."""
    pcd = np2pcd(points)
    if search_param is None:
        pcd.estimate_normals()
    else:
        pcd.estimate_normals(search_param=search_param)
    pcd.orient_normals_towards_camera_location(camera_location)
    normals = np.array(pcd.normals)
    return normals


def pcd_voxel_down_sample(points, voxel_size, min_bound=(-5.0, -5.0, -5.0), max_bound=(5.0, 5.0, 5.0)):
    """Downsample the point cloud and return sample indices."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsample_pcd, mapping, index_buckets = pcd.voxel_down_sample_and_trace(
        voxel_size, np.array(min_bound)[:, None], np.array(max_bound)[:, None])
    sample_indices = [int(x[0]) for x in index_buckets]
    return sample_indices
