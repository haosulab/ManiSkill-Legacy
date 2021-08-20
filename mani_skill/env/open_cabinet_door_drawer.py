import numpy as np

from sapien.core import Pose, Articulation

from mani_skill.env.base_env import BaseEnv
from mani_skill.utils.contrib import apply_pose_to_points, o3d_to_trimesh, trimesh_to_o3d, apply_pose_to_points, normalize_and_clip_in_interval, norm, angle_distance
from mani_skill.utils.misc import sample_from_tuple_or_scalar
from mani_skill.utils.geometry import get_axis_aligned_bbox_for_articulation
from mani_skill.utils.o3d_utils import np2mesh, merge_mesh
import trimesh

import pathlib
_this_file = pathlib.Path(__file__).resolve()


class OpenCabinetEnvBase(BaseEnv):
    def __init__(self, yaml_file_path, fixed_target_link_id=None, joint_friction=(0.05, 0.15), joint_damping=(5, 20),
                 joint_stiffness=None, *args, **kwargs):

        self.joint_friction = joint_friction
        self.joint_damping = joint_damping
        self.joint_stiffness = joint_stiffness
        self.once_init = False
        self.last_id = None

        self.fixed_target_link_id = fixed_target_link_id
        super().__init__(
            _this_file.parent.joinpath(
                yaml_file_path
            ),
            *args,
            **kwargs,
        )

    def configure_env(self):
        self.cabinet_max_dof = 8 # actually, it is 6 for our data

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.cabinet: Articulation = self.articulations['cabinet']['articulation']

        self._place_cabinet()
        self._close_all_parts()
        self._find_handles_from_articulation()
        self._place_robot()
        self._choose_target_link()
        self._ignore_collision()
        self._set_joint_physical_parameters()
        self._prepare_for_obs()
        
        [[lmin, lmax]] = self.target_joint.get_limits()

        self.target_qpos = lmin + (lmax - lmin) * self.custom['open_extent']
        self.pose_at_joint_zero = self.target_link.get_pose()

        return self.get_obs()

    def _place_cabinet(self):
        mins, maxs = get_axis_aligned_bbox_for_articulation(self.cabinet)
        self.cabinet.set_pose(Pose([0,0, -mins[2]], [1,0,0,0]) )

    def _find_handles_from_articulation(self):
        handles_info = {}
        handles_visual_body_ids = {}
        o3d_info = {}
        grasp_pose = {}

        for link in self.cabinet.get_links():
            link_name = link.get_name()
            assert link_name not in handles_info
            handles_info[link_name] = []
            handles_visual_body_ids[link_name] = []

            o3d_info[link_name] = []
            for visual_body in link.get_visual_bodies():
                if 'handle' not in visual_body.get_name():
                    continue
                handles_visual_body_ids[link_name].append(visual_body.get_visual_id())
                for i in visual_body.get_render_shapes():
                    vertices = apply_pose_to_points(i.mesh.vertices * i.scale, i.pose)
                    mesh = np2mesh(vertices, i.mesh.indices.reshape(-1, 3))
                    o3d_info[link_name].append(mesh)
                    handles_info[link_name].append((i.mesh.vertices * i.scale, i.mesh.indices, i.pose))
            if len(handles_info[link_name]) == 0:
                handles_info.pop(link_name)
                handles_visual_body_ids.pop(link_name)
                o3d_info.pop(link_name)

        for link in self.cabinet.get_links():
            link_name = link.get_name()
            if link_name not in o3d_info:
                continue
            mesh = merge_mesh(o3d_info[link_name])

            mesh = trimesh.convex.convex_hull(o3d_to_trimesh(mesh))
            pcd = mesh.sample(500)
            pcd_world = apply_pose_to_points(pcd, link.get_pose())
            lens = (pcd_world.max(0) - pcd_world.min(0)) / 2
            center = (pcd_world.max(0) + pcd_world.min(0)) / 2
            box_size = lens


            if lens[1] > lens[2]:
                flat = np.array([0, 0, 1])
            else:
                flat = np.array([0, 1, 0])

            def crop_by_box():
                sign = np.logical_and(pcd_world >= center - box_size, pcd_world <= center + box_size)
                sign = np.all(sign, axis=-1)
                return pcd_world[sign]
            pcd_world = crop_by_box()
            if pcd_world.shape[0] > 100:
                pcd_world = pcd_world[:100]
            pcd = apply_pose_to_points(pcd_world, link.get_pose().inv())

            def build_pose(forward, flat):
                extra = np.cross(flat, forward)
                ans = np.eye(4)
                ans[:3, :3] = np.array([extra, flat, forward]).T
                return Pose.from_transformation_matrix(ans)

            grasp_pose[link_name] = (link.get_pose().inv() * build_pose([1, 0, 0], flat),
                                    link.get_pose().inv() * build_pose([1, 0, 0], -flat))
            o3d_info[link_name] = (link, trimesh_to_o3d(mesh), pcd)

        self.handles_info = handles_info
        self.handles_visual_body_ids = handles_visual_body_ids
        self.o3d_info = o3d_info
        self.grasp_pose = grasp_pose
        assert len(self.handles_info.keys()) > 0
    
    def _close_all_parts(self):
        qpos = []
        for joint in self.cabinet.get_active_joints():
            [[lmin, lmax]] = joint.get_limits()
            if lmin == -np.inf or lmax == np.inf:
                raise Exception('This object has an inf limit joint.')
            qpos.append(lmin)
        self.cabinet.set_qpos(np.array(qpos))

    def _choose_target_link(self, joint_type):
        links, joints = [], []
        for link, joint in zip(self.cabinet.get_links(), self.cabinet.get_joints()):
            if joint.type == joint_type and link.get_name() in self.handles_info:
                links.append(link)
                joints.append(joint)

        if self.fixed_target_link_id is not None:
            self.target_index = self.fixed_target_link_id % len(joints)
        else:
            self.target_index = self._level_rng.choice(len(joints)) # only sample revolute/prismatic joints
            # the above line will change leve_rng's internal state multiple times
        self.target_link = links[self.target_index]
        self.target_link_name = self.target_link.get_name()
        self.target_joint = joints[self.target_index]
        self.target_index_in_active_joints = self.cabinet.get_active_joints().index(self.target_joint)
        self.target_indicator = np.zeros(self.cabinet_max_dof)
        self.target_indicator[self.target_index_in_active_joints] = 1

    def _place_robot(self):
        # negative x is door/drawer

        # base pos
        center = np.array([0, 0.8]) # to make the gripper closer to the cabinet
        dist = self._level_rng.uniform(low=1.6, high=1.8)
        theta = self._level_rng.uniform(low=0.9*np.pi, high=1.1*np.pi)
        delta = np.array([np.cos(theta), np.sin(theta)]) * dist
        base_pos = center + delta

        # base orientation
        perturb_orientation = self._level_rng.uniform(low=-0.05*np.pi, high=0.05*np.pi)
        base_theta = -np.pi + theta + perturb_orientation

        self.agent.set_state({
            'base_pos': base_pos,
            'base_orientation': base_theta,
        }, by_dict=True)


    def _ignore_collision(self):
        '''ignore collision among all movable links'''
        cabinet = self.cabinet
        for joint, link in zip(cabinet.get_joints(), cabinet.get_links()):
            if joint.type in ['revolute', 'prismatic']:
                shapes = link.get_collision_shapes()
                for s in shapes:
                    g0,g1,g2,g3 = s.get_collision_groups()
                    s.set_collision_groups(g0,g1,g2|1<<31,g3)

    def _prepare_for_obs(self):
        self.handle_visual_ids = self.handles_visual_body_ids[self.target_link.get_name()]
        self.target_link_ids = [self.target_link.get_id()]

    def get_additional_task_info(self, obs_mode):
        if obs_mode == 'state':
            return self.target_indicator
        else:
            return np.array([])

    def get_all_objects_in_state(self):
        return [], [ (self.cabinet, self.cabinet_max_dof) ]

    def _set_joint_physical_parameters(self):
        for joint in self.cabinet.get_joints():
            if self.joint_friction is not None:
                joint.set_friction(sample_from_tuple_or_scalar(self._level_rng, self.joint_friction))
            if self.joint_damping is not None:
                joint.set_drive_property(stiffness=0, 
                                        damping=sample_from_tuple_or_scalar(self._level_rng, self.joint_damping),
                                        force_limit=3.4028234663852886e+38)

    def compute_other_flag_dict(self):
        ee_cords = self.agent.get_ee_coords_sample()  # [2, 10, 3]
        current_handle = apply_pose_to_points(self.o3d_info[self.target_link_name][-1],
                                              self.target_link.get_pose())  # [200, 3]
        ee_to_handle = ee_cords[..., None, :] - current_handle
        dist_ee_to_handle = np.linalg.norm(ee_to_handle, axis=-1).min(-1).min(-1) # [2]

        handle_mesh = trimesh.Trimesh(vertices=apply_pose_to_points(np.asarray(
            self.o3d_info[self.target_link_name][-2].vertices), self.target_link.get_pose()),
            faces=np.asarray(np.asarray(self.o3d_info[self.target_link_name][-2].triangles)))

        dist_ee_mid_to_handle = trimesh.proximity.ProximityQuery(handle_mesh).signed_distance(ee_cords.mean(0)).max()

        ee_close_to_handle = dist_ee_to_handle.max() <= 0.01 and dist_ee_mid_to_handle > 0
        other_info = {
            'dist_ee_to_handle': dist_ee_to_handle,
            'dist_ee_mid_to_handle': dist_ee_mid_to_handle,
            'ee_close_to_handle': ee_close_to_handle,
        }
        return other_info

    def compute_eval_flag_dict(self):
        flag_dict = {
            'cabinet_static': self.check_actor_static(self.target_link, max_v=0.1, max_ang_v=1),
            'open_enough': self.cabinet.get_qpos()[self.target_index_in_active_joints] >= self.target_qpos
        }
        flag_dict['success'] = all(flag_dict.values())
        return flag_dict

    def compute_dense_reward(self, action, state=None):
        if state is not None:
            self.set_state(state)

        actor = self.target_link

        flag_dict = self.compute_eval_flag_dict()
        other_info = self.compute_other_flag_dict()
        dist_ee_to_handle = other_info['dist_ee_to_handle']
        dist_ee_mid_to_handle = other_info['dist_ee_mid_to_handle']

        agent_pose = self.agent.hand.get_pose()
        target_pose = self.target_link.get_pose() * self.grasp_pose[self.target_link_name][0]
        target_pose_2 = self.target_link.get_pose() * self.grasp_pose[self.target_link_name][1]

        angle1 = angle_distance(agent_pose, target_pose)
        angle2 = angle_distance(agent_pose, target_pose_2)
        gripper_angle_err = min(angle1, angle2) / np.pi

        cabinet_vel = self.cabinet.get_qvel()[self.target_index_in_active_joints]

        gripper_vel_norm = norm(actor.get_velocity())
        gripper_ang_vel_norm = norm(actor.get_angular_velocity())
        gripper_vel_rew = -(gripper_vel_norm + gripper_ang_vel_norm * 0.5)

        scale = 1
        reward = 0
        stage_reward = 0

        vel_coefficient = 1.5
        dist_coefficient = 0.5

        gripper_angle_rew = -gripper_angle_err * 3 

        rew_ee_handle = -dist_ee_to_handle.mean() * 2
        rew_ee_mid_handle = normalize_and_clip_in_interval(dist_ee_mid_to_handle, -0.01, 4E-3) - 1

        reward = gripper_angle_rew + rew_ee_handle + rew_ee_mid_handle - (dist_coefficient + vel_coefficient)
        stage_reward = -(5 + vel_coefficient + dist_coefficient)

        vel_reward = 0
        dist_reward = 0

        if other_info['ee_close_to_handle']:
            stage_reward += 0.5
            vel_reward = normalize_and_clip_in_interval(cabinet_vel, -0.1, 0.5) * vel_coefficient  # Push vel to positive
            dist_reward = normalize_and_clip_in_interval(self.cabinet.get_qpos()[self.target_index_in_active_joints],
                                                         0, self.target_qpos) * dist_coefficient
            reward += dist_reward + vel_reward
            if flag_dict['open_enough']:
                stage_reward += (vel_coefficient + 2)
                reward = reward - vel_reward + gripper_vel_rew
                if flag_dict['cabinet_static']:
                    stage_reward += 1
        info_dict = {
            'dist_ee_to_handle': dist_ee_to_handle,
            'angle1': angle1,
            'angle2': angle2,
            'dist_ee_mid_to_handle': dist_ee_mid_to_handle,
            'rew_ee_handle': rew_ee_handle,
            'rew_ee_mid_handle': rew_ee_mid_handle,

            'qpos_rew': dist_reward,
            'qvel_rew': vel_reward,

            'gripper_angle_err': gripper_angle_err * 180,
            'gripper_angle_rew': gripper_angle_rew,
            'gripper_vel_norm': gripper_vel_norm,
            'gripper_ang_vel_norm': gripper_ang_vel_norm,

            'qpos': self.cabinet.get_qpos()[self.target_index_in_active_joints],
            'qvel': cabinet_vel,
            'target_qpos': self.target_qpos,
            'reward_raw': reward,
            'stage_reward': stage_reward,
        }
        reward = (reward + stage_reward) * scale
        return reward, info_dict

    def _post_process_view(self, view_dict):
        visual_id_seg = view_dict['seg'][..., 0] # (n, m)
        actor_id_seg = view_dict['seg'][..., 1] # (n, m)

        masks = [ np.zeros(visual_id_seg.shape, dtype=np.bool) for _ in range(3)]

        for visual_id in self.handle_visual_ids:
            masks[0] = masks[0] | ( visual_id_seg == visual_id )
        for actor_id in self.target_link_ids:
            masks[1] = masks[1] | ( actor_id_seg == actor_id )
        for actor_id in self.robot_link_ids:
            masks[2] = masks[2] | ( actor_id_seg == actor_id )

        view_dict['seg'] = np.stack(masks, axis=-1)

    def get_obs(self):
        return super().get_obs(seg='both')

    def num_target_links(self, joint_type):
        links, joints = [], []
        for link, joint in zip(self.cabinet.get_links(), self.cabinet.get_joints()):
            if joint.type == joint_type and link.get_name() in self.handles_info:
                links.append(link)
                joints.append(joint)
        return len(links)

class OpenCabinetDoorEnv(OpenCabinetEnvBase):
    def __init__(self, *args, split='train', **kwargs):
        super().__init__(
            f'../assets/config_files/open_cabinet_door.yml',
            *args, **kwargs
        )
    
    def _choose_target_link(self):
        super()._choose_target_link('revolute')

    @property
    def num_target_links(self):
        return super().num_target_links('revolute')


class OpenCabinetDrawerEnv(OpenCabinetEnvBase):
    def __init__(self, yaml_file_path=f'../assets/config_files/open_cabinet_drawer.yml', *args, **kwargs):
        super().__init__(
            yaml_file_path=yaml_file_path, *args, **kwargs
        )
    
    def _choose_target_link(self):
        super()._choose_target_link('prismatic')

    @property
    def num_target_links(self):
        return super().num_target_links('prismatic')
