import numpy as np
import open3d as o3d
import trimesh
from math import pi, atan2
from sapien.core import Pose
from shapely.geometry import Polygon
from transforms3d.quaternions import quat2axangle


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def compute_relative_vel(frame_pose, frame_vel, frame_ang_vel, p_world, p_world_vel):
    p_frame = frame_pose.inv().transform(Pose(p_world)).p
    H = frame_pose.to_transformation_matrix()
    R = H[:3, :3]
    o = H[:3, 3]
    S = skew(frame_ang_vel)
    return S @ (R @ p_frame) + frame_vel - p_world_vel


def get_unit_box_corners():
    corners = np.array([[0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [0.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0],
                        ])
    return corners - [0.5, 0.5, 0.5]


def to_generalized(x):
    if x.shape[-1] == 4:
        return x
    assert x.shape[-1] == 3
    output_shape = list(x.shape)
    output_shape[-1] = 4
    ret = np.ones(output_shape)
    ret[..., :3] = x
    return ret


def to_normal(x):
    if x.shape[-1] == 3:
        return x
    assert x.shape[-1] == 4
    return x[..., :3] / x[..., 3:]


def o3d_to_trimesh(x):
    vertices = np.asarray(x.vertices)
    faces = np.asarray(x.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def trimesh_to_o3d(x):
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(x.vertices), o3d.utility.Vector3iVector(x.faces))


def apply_pose_to_points(x, pose):
    return to_normal(to_generalized(x) @ pose.to_transformation_matrix().T)


def apply_pose(pose, x):
    return apply_pose_to_points(x, pose)


def apply_pose_to_point(x, pose, with_translation=True):
    mat = pose.to_transformation_matrix()
    if with_translation:
        return mat[:3, :3] @ x + mat[:3, 3]
    else:
        return mat[:3, :3] @ x


def get_articulations_obs(articulations, with_root_info=True):
    if not isinstance(articulations, list):
        articulations = [articulations]
    ret = []
    for articulation in articulations:
        ret = [articulation.get_qpos(), articulation.get_qvel()]
        if with_root_info:
            root_pose = articulation.get_root_pose()
            ret += [root_pose.p, root_pose.q]
    return np.concatenate(ret)


def get_actors_obs(actors, base_pose=Pose()):
    if not isinstance(actors, list):
        actors = [actors]
    ret = []
    inv_base = base_pose.inv()
    inv_rot = inv_base.to_transformation_matrix()[:3, :3]
    for actor in actors:
        pose_in_base = actor.get_pose() * inv_base
        vel_in_base = inv_rot @ actor.get_velocity()
        ang_vel_in_base = inv_rot @ actor.get_angular_velocity()
        ret += [pose_in_base.p, pose_in_base.q, vel_in_base, ang_vel_in_base]
    return np.concatenate(ret)


def normalize_vec(x):
    return x / np.clip(norm(x, True), a_min=1E-6, a_max=1E6)


def angle_between_vec(a, b):
    return np.arccos(np.dot(a, b)) / pi


def rew_close_to_target(pos, vel, target_pos, max_dist=1, max_vel=1, dist_coeff=1, vel_coeff=2, angle_coeff=1):
    ret = 0
    info = {}
    target_vel = normalize_vec(target_pos - pos)
    info['target_vel'] = target_vel
    if dist_coeff > 0:
        dist = norm(pos - target_pos)
        info['rew_dist'] = (1 - normalize_and_clip_in_interval(dist, 0, max_dist)) * dist_coeff
        ret += info['rew_dist'] # - normalize_and_clip_in_interval(dist, 0, max_dist) * dist_coeff + dist_coeff
    if vel_coeff > 0:
        vel_norm = np.dot(vel, target_vel)
        info['rew_vel'] = normalize_and_clip_in_interval(vel_norm, max_vel) * vel_coeff
        ret += info['rew_vel']
    if angle_coeff > 0:
        vel_angle = angle_between_vec(normalize_vec(vel), target_vel)
        info['rew_vel_angle'] = (1 - vel_angle) * angle_coeff
        ret += info['rew_vel_angle']
    return ret, info


def normalize_and_clip_in_interval(x, min_x, max_x=None):
    if max_x is None:
        min_x = -abs(min_x)
        max_x = abs(min_x)
    len_x = max_x - min_x
    return (min(max(x, min_x), max_x) - min_x) / len_x


def clip(x, min_x, max_x):
    return min(max(min_x, x), max_x)


def normalize_reward(x, norm_x):
    return x / norm_x


def norm(x, keepdims=False):
    return np.sqrt((x ** 2).sum(axis=-1, keepdims=keepdims))


def angle_distance(q0, q1):
    qd = (q0.inv() * q1).q
    return 2 * atan2(norm(qd[1:]), qd[0]) / pi


def quaternion_distance(q1, q2):
    q1, q2 = Pose(q=q1), Pose(q=q2)
    q = q1.inv() * q2
    # print(q.q)
    return 1 - np.abs(q.q[0])


def build_pose(forward, flat):
    extra = np.cross(forward, flat)
    ans = np.eye(4)
    ans[:3, :3] = np.array([forward, flat, extra])
    return Pose.from_transformation_matrix(ans)


def pose_vec_distance(pose1, pose2):
    dist_p = np.linalg.norm(pose1.p - pose2.p)
    dist_q = quaternion_distance(pose1.q, pose2.q)
    return dist_p + 0.01 * dist_q

def pose_corner_distance(pose1, pose2):
    unit_box = get_unit_box_corners()
    t1 = pose1.to_transformation_matrix()
    t2 = pose2.to_transformation_matrix()

    corner1 = to_generalized(unit_box) @ t1.T
    corner2 = to_generalized(unit_box) @ t2.T
    # print(corner1.shape, corner2.shape)
    return np.mean(np.linalg.norm(corner1 - corner2, axis=-1))


def generate_ducttape_mesh(
        inner_radius_range, width_range, height_range, n_polygons, num
):
    # Range: [low, high], low>=0, high>=low
    duct_tapes = []
    for _ in range(num):
        for i in range(n_polygons):
            r1 = np.random.uniform(inner_radius_range[0], inner_radius_range[0])
            r2 = r1 + np.random.uniform(width_range[0], width_range[1])
            height = np.random.uniform(height_range[0], height_range[1])
            scene = trimesh.Scene()
            theta1 = 2 * np.pi * i / n_polygons
            theta2 = 2 * np.pi * (i + 1) / n_polygons
            coord1 = np.array([np.cos(theta1), np.sin(theta1)])
            coord2 = np.array([np.cos(theta2), np.sin(theta2)])
            p = [coord1 * r2, coord1 * r1, coord2 * r1, coord2 * r2]
            g = trimesh.creation.extrude_polygon(Polygon(p), height)
            scene.add_geometry(g)
            duct_tapes.append(scene)
    return duct_tapes


def compute_dist2pcd(triangle_vertices, point):
    return np.min(np.linalg.norm(triangle_vertices - point, axis=-1))


def compute_dist2surface(triangle_vertices, triangle_indices, point):
    triangles = triangle_vertices[triangle_indices.reshape(-1, 3)]
    p = trimesh.triangles.closest_point(
        triangles, np.tile(point, (triangles.shape[0], 1))
    )
    return np.min(np.linalg.norm(p - point, axis=1))


def compute_dist2object(obj, point):
    point = obj.get_pose().inv().transform(Pose(point)).p
    ds = [
        compute_dist2surface(
            g.geometry.vertices, g.geometry.indices, point
        )
        for g in obj.get_collision_shapes()
    ]
    return np.min(ds)
