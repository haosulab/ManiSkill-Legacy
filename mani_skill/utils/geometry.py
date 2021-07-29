import numpy as np
from scipy.spatial.transform import Rotation
from sapien.core import Pose, Articulation

def norm_3d(a):
    return np.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])

def sample_on_unit_sphere(rng):
    '''
    Algo from http://corysimon.github.io/articles/uniformdistn-on-sphere/
    '''
    v = np.zeros(3)
    while norm_3d(v) < 1e-4:
        v[0] = rng.normal()  # random standard normal
        v[1] = rng.normal()
        v[2] = rng.normal()
    
    v = v / norm_3d(v)
    return v

def norm_2d(a):
    return np.sqrt(a[0]*a[0]+a[1]*a[1])

def sample_on_unit_circle(rng):
    v = np.zeros(2)
    while norm_2d(v) < 1e-4:
        v[0] = rng.normal()  # random standard normal
        v[1] = rng.normal()
    
    v = v / norm_2d(v)
    return v

def rotation_between_vec(a, b): # from a to b
    a = a / norm_3d(a)
    b = b / norm_3d(b)
    axis = np.cross(a, b)
    axis = axis / norm_3d(axis) # norm might be 0
    angle = np.arccos(a @ b)
    R = Rotation.from_rotvec( axis * angle )
    return R

def angle_between_vec(a, b): # from a to b
    a = a / norm_3d(a)
    b = b / norm_3d(b)
    angle = np.arccos(a @ b)
    return angle

def wxyz_to_xyzw(q):
    return np.concatenate([ q[1:4], q[0:1] ])

def xyzw_to_wxyz(q):
    return np.concatenate([ q[3:4], q[0:3] ])

def rotate_2d_vec_by_angle(vec, theta):
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return rot_mat @ vec

def angle_distance(q0, q1): # q0, q1 are sapien pose
    qd = (q0.inv() * q1).q
    return 2 * np.arctan2(norm_3d(qd[1:]), qd[0]) / np.pi


def get_axis_aligned_bbox_for_articulation(art: Articulation):
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = -mins
    for link in art.get_links():
        lp = link.pose
        for s in link.get_collision_shapes():
            p = lp * s.get_local_pose()
            T = p.to_transformation_matrix()
            vertices = s.geometry.vertices * s.geometry.scale
            vertices = vertices @ T[:3, :3].T + T[:3, 3]
            mins = np.minimum(mins, vertices.min(0))
            maxs = np.maximum(maxs, vertices.max(0))
    return mins, maxs

def get_axis_aligned_bbox_for_actor(actor):
    mins = np.ones(3) * np.inf
    maxs = -mins

    for shape in actor.get_collision_shapes(): # this is CollisionShape
        scaled_vertices = shape.geometry.vertices * shape.geometry.scale
        local_pose = shape.get_local_pose()
        mat = (actor.get_pose() * local_pose).to_transformation_matrix()
        world_vertices = scaled_vertices @ (mat[:3, :3].T) + mat[:3, 3]
        mins = np.minimum(mins, world_vertices.min(0))
        maxs = np.maximum(maxs, world_vertices.max(0))

    return mins, maxs

def get_local_axis_aligned_bbox_for_link(link):
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = -mins
    for s in link.get_collision_shapes():
        p = s.get_local_pose()
        T = p.to_transformation_matrix()
        vertices = s.geometry.vertices * s.geometry.scale
        vertices = vertices @ T[:3, :3].T + T[:3, 3]
        mins = np.minimum(mins, vertices.min(0))
        maxs = np.maximum(maxs, vertices.max(0))
    return mins, maxs