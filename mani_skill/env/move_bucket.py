import numpy as np
from sapien.core import Pose
import sapien.core as sp
from mani_skill.env.base_env import BaseEnv
from mani_skill.utils.contrib import compute_relative_vel, apply_pose_to_points
from mani_skill.utils.geometry import angle_between_vec, get_axis_aligned_bbox_for_articulation, get_local_axis_aligned_bbox_for_link
from scipy.spatial import distance
from transforms3d.quaternions import quat2mat
from transforms3d.euler import euler2quat, quat2euler
import pathlib

_this_file = pathlib.Path(__file__).resolve()


class MoveBucketEnv(BaseEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            _this_file.parent.joinpath(
                f'../assets/config_files/move_bucket.yml'
            ),
            *args,
            **kwargs,
        )

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.bucket = self.articulations['bucket']['articulation']
        self.root_link = self.bucket.get_links()[0]
        
        self._set_target()
        self._place_bucket()
        self._place_robot()
        self._place_balls()
        self._load_bucket_pcds()


        for i in range(25):
            self._scene.step()

        return self.get_obs()

    def _set_target(self, target_xy=None):
        self.target_xy = np.zeros(2)
        target_orientation = 0
        target_q = euler2quat(target_orientation, 0, 0, 'szyx')
        
        # place a target platform on ground
        self.target_p = np.zeros(3)
        self.target_p[:2] = self.target_xy
        builder: sp.ActorBuilder = self._scene.create_actor_builder()
        t = self.custom['target_radius']
        builder.add_box_visual(pose=Pose(p=self.target_p, q=target_q), half_size=[t, t, 0.1], material=self.render_materials['white_diffuse'])
        builder.add_box_collision(pose=Pose(p=self.target_p, q=target_q), half_size=[t, t, 0.1], material=self.physical_materials['object_material'], density=1000)
        target_platform: sp.Actor = builder.build_static(name='target_platform')

        self.target_info = np.zeros(3)
        self.target_info[:2] = self.target_xy
        self.target_info[2] = target_orientation
    
    def _place_bucket(self):
        pose = self.bucket.pose  # bucket2world
        bb = np.array(get_axis_aligned_bbox_for_articulation(self.bucket))  # bb in world

        # find a pos
        center = self.target_xy
        dist = self._level_rng.uniform(low=0.8, high=1.2)
        theta = self._level_rng.uniform(low=-np.pi, high=np.pi)
        self.init_bucket_to_target_theta = theta
        delta = np.array([np.cos(theta), np.sin(theta)]) * dist
        pos_xy = center + delta

        bb = np.array(get_axis_aligned_bbox_for_articulation(self.bucket))  # bb in world
        self.bucket_center_offset = (bb[1, 2] - bb[0, 2]) / 5
        self.bucket_body_link = self.bucket.get_active_joints()[0].get_parent_link()
        self.bb_local = np.array(get_local_axis_aligned_bbox_for_link(self.bucket_body_link))
        self.center_local = (self.bb_local[0] + self.bb_local[1]) / 2

        pose.set_p([pos_xy[0], pos_xy[1], pose.p[2] - bb[0, 2]])

        # find a orientation
        ax, ay, az = quat2euler(pose.q, 'sxyz')
        az = self._level_rng.uniform(low=-np.pi, high=np.pi)
        q = euler2quat(ax, ay, az, 'sxyz')
        pose.set_q(q)

        self.bucket.set_pose(pose)

        # ------
        lim = self.bucket.get_active_joints()[0].get_limits()
        v = (lim[0, 1] - lim[0, 0]) * 0.1
        lim[0, 0] += v
        lim[0, 1] -= v
        self.bucket.get_active_joints()[0].set_limits(lim)

        self.bucket.set_qpos(self.bucket.get_qlimits()[:, 0])
        self.bucket.set_qvel(np.zeros(self.bucket.dof))
        self.init_bucket_height = self.bucket_body_link.get_pose().transform(self.bucket_body_link.get_cmass_local_pose()).p[2]
        self.bucket_root = self.bucket.get_links()[0]
        
    def _place_robot(self):
        ############## base

        # base pos
        center = self.bucket.get_pose().p
        dist = self._level_rng.uniform(low=0.6, high=0.8)
        theta = self._level_rng.uniform(low=-0.4*np.pi, high=0.4*np.pi)
        theta += self.init_bucket_to_target_theta
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
    
    def _place_balls(self):
        R = 0.05
        bb = np.array(get_axis_aligned_bbox_for_articulation(self.bucket))
        self.balls_radius = R
        builder = self._scene.create_actor_builder()
        builder.add_sphere_collision(radius=R, density=1000)
        builder.add_sphere_visual(radius=R, color=[0, 1, 1])
        self.balls = []
        GX = GY = 1
        GZ = 1

        ball_id = 0
        for i in range(GX):
            for j in range(GY):
                for k in range(GZ):
                    dx = -GX * R * 2 / 2 + R + 2 * R * i
                    dy = -GY * R * 2 / 2 + R + 2 * R * j
                    dz = R + R * 2 * k
                    pose = self.bucket.pose
                    pose = Pose(
                        [pose.p[0] + dx, pose.p[1] + dy, bb[1, 2] - bb[0, 2] + dz]
                    )
                    ball_id += 1
                    actor = builder.build(name='ball_{:d}'.format(ball_id))
                    actor.set_pose(pose)
                    self.balls.append(actor)
        
    def _load_bucket_pcds(self):
        o3d_info = {}
        from mani_skill.utils.o3d_utils import np2mesh, mesh2pcd, merge_mesh

        for link in self.bucket.get_links():
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
    
    def compute_dense_reward(self, action, state=None):
        actor = self.bucket.get_links()[0]
        ee_coords = np.array(self.agent.get_ee_coords())
        ee_mids = np.array([ee_coords[:2].mean(0), ee_coords[2:].mean(0)])
        ee_vels = np.array(self.agent.get_ee_vels())
        
        target_points = []
        for link in self.bucket.get_links():
            link_name = link.get_name()
            if link_name not in self.o3d_info:
                continue
            target_pcd = self.o3d_info[link_name][-1]
            target_points.append(apply_pose_to_points(np.asarray(target_pcd.points), link.get_pose()))
        target_points = np.concatenate(target_points, 0)

        dist_ee_actors = np.sqrt(((ee_coords[:, None] - target_points[None]) ** 2).sum(-1)).min(-1)
        dist_ee_actor = dist_ee_actors.mean()
        dist_robotroot_actor = np.linalg.norm(self.agent.get_base_link().get_pose().p[:2] - actor.get_pose().p[:2])

        bucket_mid = self.bucket_body_link.get_pose().transform(self.bucket_body_link.get_cmass_local_pose()).p
        bucket_mid[2] += self.bucket_center_offset
        v1 = ee_mids[0] - bucket_mid
        v2 = ee_mids[1] - bucket_mid
        ees_oppo = distance.cosine(v1, v2)
        z_axis_world = np.array([0,0,1])
        z_axis_bucket = quat2mat(self.root_link.get_pose().q) @ z_axis_world
        ees_height_diff = abs((quat2mat(self.root_link.get_pose().q).T @ (ee_mids[0] - ee_mids[1]))[2])
        log_ees_height_diff = np.log(ees_height_diff + 1e-5)
        
        # EE Part [ Approximate EE origin -> handle]
        log_dist_ee_actor = np.log(dist_ee_actor + 1e-5)

        ## EE Actor Part
        rel_vel_ee_actor = np.array([
            compute_relative_vel(
                actor.get_pose(),
                actor.get_velocity(),
                actor.get_angular_velocity(),
                ee_coord,
                ee_vel,
            )
            for ee_coord, ee_vel in zip(ee_coords, ee_vels)
        ])
        rel_vel_ee_actor_norm = np.linalg.norm(rel_vel_ee_actor, axis=-1).mean()

        # For reward
        dist_pos = actor.get_pose().p[:2] - self.target_xy
        dist_pos_norm = np.linalg.norm(dist_pos)
        bucket_height = self.bucket_body_link.get_pose().transform(self.bucket_body_link.get_cmass_local_pose()).p[2]
        dist_bucket_height = np.linalg.norm(bucket_height - self.init_bucket_height - 0.2)
        dist_ori = abs(angle_between_vec(z_axis_world, z_axis_bucket))
        log_dist_ori = np.log(dist_ori)

        ## Actor Part
        actor_vel = actor.get_velocity()
        actor_vel_norm = np.linalg.norm(actor_vel)
        actor_vel_dir = distance.cosine(actor_vel[:2], dist_pos)
        actor_ang_vel_norm = np.linalg.norm(actor.get_angular_velocity())
        action_norm = np.linalg.norm(action)
        actor_vel_up = actor_vel[2]

        info_dict = {
            'dist_ee_actor': dist_ee_actor,
            'dist_robotroot_actor': dist_robotroot_actor,
            'dist_pos': dist_pos_norm,
            'dist_ori': dist_ori,
            'bucket_height': bucket_height,
            'ees_oppo': ees_oppo,
            'ees_height_diff': ees_height_diff, 
            'actor_vel_up': actor_vel_up,
            'actor_vel_norm': actor_vel_norm,
            'actor_vel_dir': actor_vel_dir,
            'rel_vel_ee_actor_norm': rel_vel_ee_actor_norm,
            'action_norm': action_norm,
        }

        stage_reward = -20
        reward_scale = 1.0

        reward = - dist_ee_actor * 1 - np.clip(log_dist_ee_actor, -10, 0) * 1 - dist_ori * 0.2 - np.clip(log_ees_height_diff, -10, 0) * 0.2 - action_norm * 1E-6# + dist_robotroot_actor * 0.2
        if dist_ee_actor < 0.1:
            stage_reward += 2
            reward += ees_oppo * 2# - np.clip(log_ees_height_diff, -10, 0) * 0.2
            if dist_bucket_height < 0.03:
                stage_reward += 2
                reward -= np.clip(log_dist_ori, -4, 0)
                if dist_pos_norm <= 0.3:
                    stage_reward += 2
                    reward += np.exp(-actor_vel_norm * 10) * 2 #+ np.exp(-actor_ang_vel_norm) * 0.5
                    if (actor_vel_norm <= 0.1 and actor_ang_vel_norm <= 0.2):
                        stage_reward += 2
                        if dist_ori <= 0.1 * np.pi:
                            stage_reward += 2 
                else:
                    reward_vel = (actor_vel_dir - 1) * actor_vel_norm
                    reward += np.clip(1 - np.exp(-reward_vel), -1, np.inf) * 2 - dist_pos_norm * 2
            else:
                reward += np.clip(1 - np.exp(-actor_vel_up), -1, np.inf) * 2 - dist_bucket_height * 20
                
        if dist_ori > 0.4 * np.pi:
            stage_reward -= 2


        reward += stage_reward
        info_dict['stage_reward'] = stage_reward * reward_scale
        reward *= reward_scale
        return reward, info_dict

    def get_all_objects_in_state(self):
        return self.balls, [ (self.bucket, 2) ] # bucket max dof is 1 in our data
    
    def compute_eval_flag_dict(self):
        w2b = self.bucket_body_link.pose.inv().to_transformation_matrix() # world to bucket

        in_bucket = True
        for b in self.balls:
            p = w2b[:3, :3] @ b.pose.p + w2b[:3, 3]
            if not np.all((p > self.bb_local[0]) * (p < self.bb_local[1])):
                in_bucket = False
                break
        
        z_axis_world = np.array([0,0,1])
        z_axis_bucket = quat2mat(self.root_link.get_pose().q) @ z_axis_world

        flag_dict = {
            'balls_in_bucket': in_bucket,
            'bucket_above_platform': np.linalg.norm(self.root_link.get_pose().p[:2] - self.target_xy) < self.custom['target_radius'],
            'bucket_standing': abs(angle_between_vec(z_axis_world, z_axis_bucket)) < 0.1*np.pi,
            'bucket_static': self.check_actor_static(self.bucket_root, max_v=0.1, max_ang_v=0.2),
        }
        flag_dict['success'] = all(flag_dict.values())

        return flag_dict
