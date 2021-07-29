from mani_skill.env.base_env import BaseEnv
from sapien.core import Pose, Articulation
import sapien.core as sp
import numpy as np
from mani_skill.utils.contrib import apply_pose_to_points
from scipy.spatial import distance
from mani_skill.utils.geometry import angle_between_vec
from transforms3d.quaternions import quat2mat
from transforms3d.euler import euler2quat, quat2euler

import pathlib

_this_file = pathlib.Path(__file__).resolve()


class PushChairEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            _this_file.parent.joinpath(
                f"../assets/config_files/push_chair.yml"
            ),
            *args,
            **kwargs,
        )

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.chair: Articulation = self.articulations["chair"]["articulation"]
        self.root_link = self.chair.get_links()[0]

        self._ignore_collision()
        self._set_target()
        self._place_chair()
        self._place_robot()
        self._load_chair_pcds()
        self._set_physical_parameters()

        return self.get_obs()

    def _is_wheel(self, link):
        for visual_body in link.get_visual_bodies():
            if 'wheel' in visual_body.get_name():
                return True
        return False

    def _set_physical_parameters(self):
        self.center_joint_idx = None
        # It is possible no helper links
        for joint in self.chair.get_active_joints():
            link = joint.get_parent_link()
            if link is not None and 'helper' in link.get_name(): # revolute joint between seat and support
                # assert joint.type == 'revolute'
                self.center_joint_idx = self.chair.get_active_joints().index(joint)
                joint.set_friction(self._level_rng.uniform(0.05, 0.15))
                joint.set_drive_property(stiffness=0, damping=self._level_rng.uniform(5, 15))
            else:
                joint.set_friction(self._level_rng.uniform(0.0, 0.1))
                joint.set_drive_property(stiffness=0, damping=self._level_rng.uniform(0, 0.5))

        # Need to update the following things everytime relaoding the chair
        self.wheel_links = []
        for link in self.chair.get_links():
            if self._is_wheel(link):
                self.wheel_links.append(link)

        self.wheel_material = self._scene.create_physical_material(
            static_friction=1, dynamic_friction=1, restitution=0)

        for link in self.wheel_links:
            shapes = link.get_collision_shapes()
            for shape in shapes:
                shape.set_physical_material(self.wheel_material)

    def _set_target(self):
        self.target_xy = np.zeros(2)
        # draw a target indicator on ground
        self.target_p = np.zeros(3)
        self.target_p[:2] = self.target_xy
        builder: sp.ActorBuilder = self._scene.create_actor_builder()
        builder.add_sphere_visual(
            pose=Pose(p=self.target_p, q=np.array([1,0,0,0])), 
            radius=self.custom['target_radius'], 
            color=(1,0,0)
        )
        target_indicator: sp.Actor = builder.build_static(name='target_indicator')

    def _place_chair(self):
        pose = self.chair.get_pose()
        p, q = pose.p, pose.q

        # find a pos
        center = self.target_xy
        dist = self._level_rng.uniform(low=0.8, high=1.2)
        theta = self._level_rng.uniform(low=-np.pi, high=np.pi)
        delta = np.array([np.cos(theta), np.sin(theta)]) * dist
        p[:2] = center + delta
        
        # chair (the back link) face to target, then add a perturbation
        _, _, az_seat = quat2euler(self.seat_link.get_pose().q, 'sxyz')
        ax, ay, az = quat2euler(q, 'sxyz')
        az = az - az_seat + np.pi*1.5 + theta # face to target
        perturb_theta = self._level_rng.uniform(low=-0.4*np.pi, high=0.4 * np.pi) # add a perturbation
        az += perturb_theta
        self.init_chair_orientation = theta + perturb_theta
        q = euler2quat(ax, ay, az, 'sxyz')

        self.chair.set_pose(Pose(p=p,q=q))

        self.chair.set_qvel(np.zeros_like(self.chair.get_qvel()))
        self.chair.set_root_velocity(np.zeros(3))
        self.chair.set_root_angular_velocity(np.zeros(3))

    def _place_robot(self):
        ############## base

        # base pos
        center = self.chair.get_pose().p
        dist = self._level_rng.uniform(low=0.8, high=1.2)
        theta = self._level_rng.uniform(low=-0.2*np.pi, high=0.2*np.pi)
        theta += self.init_chair_orientation
        delta = np.array([np.cos(theta), np.sin(theta)]) * dist
        base_pos = center[:2] + delta

        # base orientation
        perturb_orientation = self._level_rng.uniform(low=-0.05*np.pi, high=0.05*np.pi)
        base_theta = -np.pi + theta + perturb_orientation

        ############## set state
        self.agent.set_state({
            'base_pos': base_pos,
            'base_orientation': base_theta,
        }, by_dict=True)

    def _load_chair_pcds(self):
        o3d_info = {}
        from mani_skill.utils.o3d_utils import np2mesh, mesh2pcd, merge_mesh

        for link in self.chair.get_links():
            link_name = link.get_name()
            o3d_info[link_name] = []
            for visual_body in link.get_visual_bodies():
                for i in visual_body.get_render_shapes():
                    vertices = apply_pose_to_points(i.mesh.vertices * i.scale, i.pose)
                    mesh = np2mesh(vertices, i.mesh.indices.reshape(-1, 3))
                    o3d_info[link_name].append(mesh)
                    
            if len(o3d_info[link_name]) == 0:
                o3d_info.pop(link_name)
            else:
                mesh = merge_mesh(o3d_info[link_name])
                pcd = mesh2pcd(mesh, 512)
                o3d_info[link_name] = (link, mesh, pcd)

        self.o3d_info = o3d_info

    def _is_seat(self, link):
        for visual_body in link.get_visual_bodies():
            if 'seat' in visual_body.get_name():
                return True
        return False

    def _is_support(self, link):
        for visual_body in link.get_visual_bodies():
            name = visual_body.get_name()
            if 'leg' in name or 'foot' in name:
                return True
        return False

    def _ignore_collision(self):
        '''ignore collision between seat frame and the support'''
        chair = self.chair

        seat_link, support_link = None, None
        for link in chair.get_links():
            if self._is_seat(link):
                seat_link = link
            if self._is_support(link):
                support_link = link
            if seat_link is not None and support_link is not None:
                break
        self.seat_link = seat_link
        
        if seat_link == support_link: # sometimes they are the same one
            return

        for link in [seat_link, support_link]:
            shapes = link.get_collision_shapes()
            for s in shapes:
                g0,g1,g2,g3 = s.get_collision_groups()
                s.set_collision_groups(g0,g1,g2|1<<31,g3)
                
    def get_all_objects_in_state(self):
        return [], [ (self.chair, 25) ] # chair max dof is 20 in our data
            
    def compute_dense_reward(self, action):
        actor = self.root_link

        ee_coords = np.array(self.agent.get_ee_coords())
        
        target_points = []
        for link in self.chair.get_links():
            link_name = link.get_name()
            if link_name not in self.o3d_info:
                continue
            target_pcd = self.o3d_info[link_name][-1]
            target_points.append(apply_pose_to_points(np.asarray(target_pcd.points), link.get_pose()))
        target_points = np.concatenate(target_points, 0)

        dist_ee_actors = np.sqrt(((ee_coords[:, None] - target_points[None]) ** 2).sum(-1)).min(-1)
        dist_ee_actor = dist_ee_actors.mean()
        dist_robotroot_actor = np.linalg.norm(self.agent.get_base_link().get_pose().p[:2] - self.root_link.get_pose().p[:2])

        # EE Part [ Approximate EE origin -> handle]
        log_dist_ee_actor = np.log(dist_ee_actor + 1e-5)

        # For reward
        dist_pos = self.root_link.get_pose().p[:2] - self.target_xy
        dist_pos_norm = np.linalg.norm(dist_pos)
        z_axis_world = np.array([0,0,1])
        z_axis_bucket = quat2mat(self.root_link.get_pose().q) @ z_axis_world
        dist_ori = abs(angle_between_vec(z_axis_world, z_axis_bucket))

        ## Actor Part
        actor_vel = actor.get_velocity()
        actor_vel_norm = np.linalg.norm(actor_vel)
        actor_vel_dir = distance.cosine(actor_vel[:2], dist_pos)
        actor_ang_vel_norm = np.linalg.norm(actor.get_angular_velocity())
        action_norm = np.linalg.norm(action)

        info_dict = {
            'dist_ee_actor': dist_ee_actor,
            'dist_robotroot_actor': dist_robotroot_actor,
            'dist_pos_norm': dist_pos_norm,
            'dist_ori': dist_ori,
            'actor_vel_norm': actor_vel_norm,
            'actor_vel_dir': actor_vel_dir,
            'action_norm': action_norm,
        }

        stage_reward = -10
        reward_scale = 2.0

        reward = - dist_ee_actor * 1 - np.clip(log_dist_ee_actor, -10, 0) * 1 - dist_ori * 0.2 - action_norm * 1E-6# + dist_robotroot_actor * 1

        if dist_ori < 0.2 * np.pi:
            if dist_ee_actor < 0.1:
                stage_reward += 2

                if dist_pos_norm <= 0.15:
                    stage_reward += 2
                    reward += np.exp(-actor_vel_norm * 10) * 2
                    if (actor_vel_norm <= 0.1 and actor_ang_vel_norm <= 0.2):
                        stage_reward += 2
                else:
                    reward_vel = (actor_vel_dir - 1) * actor_vel_norm
                    reward += np.clip(1 - np.exp(-reward_vel), -1, np.inf) * 2 - dist_pos_norm * 2
        else:
            stage_reward -= 5

        reward += stage_reward
        info_dict['stage_reward'] = stage_reward * reward_scale
        reward *= reward_scale

        return reward, info_dict


    def compute_eval_flag_dict(self):
        z_axis_world = np.array([0,0,1])
        z_axis_chair = quat2mat(self.root_link.get_pose().q) @ z_axis_world

        flag_dict = {
            "chair_close_to_target": np.linalg.norm(self.root_link.get_pose().p[:2] - self.target_xy) < self.custom['target_radius'],
            "chair_standing": abs(angle_between_vec(z_axis_world, z_axis_chair)) < 0.05*np.pi,
            "chair_static": self.check_actor_static(self.root_link, max_v=0.1, max_ang_v=0.2),
        }
        flag_dict["success"] = all(flag_dict.values())
        return flag_dict

    @property
    def chair_mass(self):
        mass = 0
        for link in self.chair.get_links():
            mass += link.get_mass()
        return mass
