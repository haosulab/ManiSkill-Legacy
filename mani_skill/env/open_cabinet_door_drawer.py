import pdb
import numpy as np

from sapien.core import Pose, Articulation

from mani_skill.env.base_env import BaseEnv
from mani_skill.utils.contrib import apply_pose_to_points, o3d_to_trimesh, trimesh_to_o3d, apply_pose_to_points, normalize_and_clip_in_interval, norm, angle_distance
from mani_skill.utils.misc import sample_from_tuple_or_scalar
from mani_skill.utils.geometry import get_axis_aligned_bbox_for_articulation, get_axis_aligned_bbox_for_actor,get_all_aabb_for_actor
from mani_skill.utils.o3d_utils import np2mesh, merge_mesh
import trimesh
from collections import OrderedDict

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
        self._choose_target_link()
        # self._find_handles_from_articulation()
        self._place_robot()
        # self._choose_target_link()
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
        base_theta = 0 #-np.pi + theta + perturb_orientation

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

    def get_obs(self, **kwargs):
        # warning, overwrite original get_obs
        s = 13 + 13 + 7 + 6  # qpos [13] qvel [13] hand(xyz,q) [7] bbox [6]
        dense_obs = np.zeros(s)
        qpos = self.agent.robot.get_qpos()
        qvel = self.agent.robot.get_qvel()
        hand = self.agent.hand
        hand_p, hand_q = hand.pose.p, hand.pose.q
        mins, maxs = self.get_aabb_for_min_x(self.target_link)
        dense_obs[:13] = qpos
        dense_obs[13:26] = qvel
        dense_obs[26:29] = hand_p
        dense_obs[29:33] = hand_q
        dense_obs[33:36] = mins
        dense_obs[36:39] = maxs
        return dense_obs

    def get_aabb_for_min_x(self, link): 
        all_mins, all_maxs = self.get_all_minmax(link)
        sorted_index = sorted(range(len(all_maxs)),key=lambda i: all_maxs[i][0])
        max_x_index = sorted_index[0]
        mins = all_mins[max_x_index]
        maxs = all_maxs[max_x_index]

        return mins, maxs
        
    def get_all_minmax(self,link):
        all_mins, all_maxs = get_all_aabb_for_actor(link)
        return all_mins, all_maxs

import sapien.core as sapien
from copy import deepcopy
from mani_skill.utils.config_parser import (
    process_variables,
    process_variants,
)

class OpenCabinetDrawerEnv_CabinetSelection(OpenCabinetDrawerEnv):
    def reset(self, level=None, cabinet_id=None, target_link_id=None, *args, **kwargs):
        if level is None:
            level = self._main_rng.randint(2 ** 32)
        self.level = level
        self._level_rng = np.random.RandomState(seed=self.level)

        # recreate scene
        scene_config = sapien.SceneConfig()
        for p, v in self.yaml_config['physics'].items():
            if p != 'simulation_frequency':
                setattr(scene_config, p, v)
        self._scene = self._engine.create_scene(scene_config)
        self._scene.set_timestep(self.timestep)

        config = deepcopy(self.yaml_config)
        config = process_variables(config, self._level_rng)
        self.all_model_ids = list(config['layout']['articulations'][0]['_variants']['options'].keys())
        self.id_to_parameters = deepcopy(config['layout']['articulations'][0]['_variants']['options'])
        self.level_config, self.level_variant_config = process_variants(
            config, self._level_rng, self.variant_config
        )
        if cabinet_id is not None:
            for k,v in self.id_to_parameters[str(cabinet_id)].items():
                self.level_config['layout']['articulations'][0][k] = v

        # load everything
        self._setup_renderer()
        self._setup_physical_materials()
        self._setup_render_materials()
        self._load_actors()
        self._load_articulations()
        self._setup_objects()
        self._load_agent()
        self._load_custom()
        self._setup_cameras()
        if self._viewer is not None:
            self._setup_viewer()

        self._init_eval_record()
        self.step_in_ep = 0



        self.cabinet: Articulation = self.articulations['cabinet']['articulation']

        self._place_cabinet()
        self._close_all_parts()
        self._find_handles_from_articulation()

        links, joints = [], []
        for link, joint in zip(self.cabinet.get_links(), self.cabinet.get_joints()):
            if joint.type == 'prismatic' and link.get_name() in self.handles_info:
                links.append(link)
                joints.append(joint)

        if self.fixed_target_link_id is not None:
            self.target_index = self.fixed_target_link_id % len(joints)
        else:
            self.target_index = self._level_rng.choice(len(joints)) # only sample revolute/prismatic joints
            # the above line will change leve_rng's internal state multiple times
        if target_link_id is not None:
            self.target_index = target_link_id % len(joints)
        self.target_link = links[self.target_index]
        self.target_link_name = self.target_link.get_name()
        self.target_joint = joints[self.target_index]
        self.target_index_in_active_joints = self.cabinet.get_active_joints().index(self.target_joint)
        self.target_indicator = np.zeros(self.cabinet_max_dof)
        self.target_indicator[self.target_index_in_active_joints] = 1

        # self._find_handles_from_articulation()
        self._place_robot()
        # self._choose_target_link()
        self._ignore_collision()
        self._set_joint_physical_parameters()
        self._prepare_for_obs()
        
        [[lmin, lmax]] = self.target_joint.get_limits()

        self.target_qpos = lmin + (lmax - lmin) * self.custom['open_extent']
        self.pose_at_joint_zero = self.target_link.get_pose()

        return self.get_obs()


class OpenCabinetDrawerMagicEnv(OpenCabinetEnvBase):
    def __init__(self, yaml_file_path=f'../assets/config_files/open_cabinet_door_magic.yml', *args, **kwargs):
        super().__init__(
            yaml_file_path=yaml_file_path, *args, **kwargs
        )
        self.magic_drive = None
        self.connected=False
    
    # def reset(self, *args, **kwargs):
    #     self.magic_drive = None
    #     self.connected = False
    #     return super().reset(*args, **kwargs)
    # def get_obs(self, **kwargs):
    #     dense_obs = np.zeros(5+5+1) # robot qpos [5] robot qvel [5] target_link (qpos) [1]
    #     robot = self.agent.robot
    #     qpos = robot.get_qpos()
    #     qvel = robot.get_qvel()
    #     dense_obs[:5] = qpos
    #     dense_obs[5:10] = qvel
    #     dense_obs[10:11] = self.cabinet.get_qpos()[self.target_index_in_active_joints]
    #     return dense_obs

    # def get_custom_observation(self):
    def get_obs(self, **kwargs):
        dense_obs = np.zeros(5+5+6) #qpos[5] qvel[5] bbox[6]
        robot = self.agent.robot
        qpos = robot.get_qpos()
        qvel = robot.get_qvel()
        mins, maxs = self.get_aabb_for_min_x(self.target_link)
        dense_obs[:5] = qpos
        dense_obs[5:10] = qvel
        dense_obs[10:13] = mins
        dense_obs[13:16] = maxs
        return dense_obs

        # agent_state = self.agent.get_state()
        # mins, maxs = self.get_aabb_for_min_x()
        # drawer_info = OrderedDict(mins=mins, max=maxs)
        # obs = OrderedDict(
        #     agent=agent_state,
        #     drawer=drawer_info,
        # )
        # return obs



    def get_handle_coord(self):
        handles_info = self.handles_info
        #pdb.set_trace()
        handle_pose = handles_info[self.target_link_name][-1][-1]
        assert type(handle_pose) == Pose
        coords = handle_pose.p
        coords2 = apply_pose_to_points(coords, self.cabinet.get_root_pose())
        return coords2

    def _place_robot(self):
        #print("placing robot")
        # self.agent.robot.set_qpos([-0.5,0,0.5,0.04,0.04]) # tmu: confirm this is 0.4 or 0.04, you write 0.4
        ## dummy call using level_rng? NO
        pass

    def _choose_target_link(self):
        super()._choose_target_link('prismatic')

    @property
    def num_target_links(self):
        return super().num_target_links('prismatic')

    @property
    def viewer(self):
        return self._viewer
        
    def magic_grasp(self):
        assert self.magic_drive is None
        assert self.connected is False

        actor1 = self.agent.grasp_site
        pose1 = Pose(p=[0,0,0], q=[1,0,0,0])
        
        actor2 = self.target_link
        # maybe first get handle's world pose and convert it to actor2's frame?
        #T_w_handle = self.handles_info[self.target_link_name][-1][-1] #world frame
        # actor 2 is the target link, get target link pose
        #pose2_mat = T_w_handle.inv().to_transformation_matrix() @ actor_2_frame.to_transformation_matrix()
        #pose2 = Pose.from_transformation_matrix(pose2_mat)
        #pose2 = apply_pose_to_points(T_w_handle.p, actor2.get_pose())
        #pose2 = Pose(p=[0,0,0],q=[1,0,0,0])

        pose2=actor1.pose.inv().to_transformation_matrix() @ actor2.pose.to_transformation_matrix()
        pose2=Pose.from_transformation_matrix(pose2) 





        magic_drive=self._scene.create_drive(actor1, pose1, actor2, pose2)
        magic_drive.set_x_properties(stiffness=5e4, damping=3e3)
        magic_drive.set_y_properties(stiffness=5e4, damping=3e3)
        magic_drive.set_z_properties(stiffness=5e4, damping=3e3)
        magic_drive.set_slerp_properties(stiffness=5e4, damping=3e3)

        self.connected = True
        self.magic_drive = magic_drive

    def magic_release(self):
        if self.connected is True:
            self.connected = False
           # self._scene.create_drive()
            self._scene.remove_drive(self.magic_drive)
            self.magic_drive = None

    def get_target_link_bbox(self):
        mins, maxs = get_axis_aligned_bbox_for_actor(self.target_link)
        return mins, maxs
    

    def draw_bboxes(self, mins, maxs, name='bbox'):
        center = (mins + maxs) / 2 
        half_size = (maxs - mins) / 2

        builder=self._scene.create_actor_builder()
        builder.add_box_visual(half_size=half_size, color=[1, 0, 0])
        bbox = builder.build_static(name)
        bbox.set_pose(Pose(p=center,q=[1,0,0,0]))

    def remove_drawer(self):
        self._scene.remove_articulation(self.cabinet)
        self.cabinet = None

    def get_all_minmax(self,link):
        all_mins, all_maxs = get_all_aabb_for_actor(link)
        return all_mins, all_maxs

    def draw_all_aabb(self, all_mins, all_maxs):
        assert len(all_mins) == len(all_maxs)
        for i in range(len(all_mins)):
            mins = all_mins[i]
            maxs = all_maxs[i]
            self.draw_bboxes(mins, maxs, name='bbox'+str(i))

    def get_aabb_for_min_x(self, link): 
        all_mins, all_maxs = self.get_all_minmax(link)
        sorted_index = sorted(range(len(all_maxs)),key=lambda i: all_maxs[i][0])
        max_x_index = sorted_index[0]
        mins = all_mins[max_x_index]
        maxs = all_maxs[max_x_index]

        return mins, maxs