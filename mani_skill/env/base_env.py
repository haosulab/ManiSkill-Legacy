from sapien.utils import Viewer
import sapien.core as sapien
import numpy as np
import importlib
from gym import Env, spaces
import zipfile
import requests
import io
from sapien.core import Pose
from mani_skill.utils.config_parser import (
    preprocess,
    process_variables,
    process_variants,
)
import warnings
from copy import deepcopy
import os
from ..agent import CombinedAgent
from .camera import CombinedCamera, read_images_from_camera, read_pointclouds_from_camera
from collections import defaultdict
from mani_skill.utils.geometry import angle_distance
from mani_skill.utils.misc import get_actor_state, get_pad_articulation_state

from pathlib import Path


def download_data(model_id, directory=None):
    url = 'https://storage1.ucsd.edu/datasets/PartNetMobilityScrambled/{}.zip'.format(
        model_id
    )
    if not directory:
        directory = os.environ.get('PARTNET_MOBILITY_DATASET')
        if not directory:
            directory = 'partnet-mobility-dataset'
    urdf_file = os.path.join(directory, str(model_id), 'mobility.urdf')

    # return if file exists
    if os.path.exists(urdf_file):
        return urdf_file

    # download file
    r = requests.get(url, stream=True)
    if not r.ok:
        raise Exception(
            'Download PartNet-Mobility failed. '
            'Please check your token and IP address.'
            'Also make sure sure the model id is valid'
        )

    z = zipfile.ZipFile(io.BytesIO(r.content))

    os.makedirs(directory, exist_ok=True)
    z.extractall(directory)
    return urdf_file


_engine = sapien.Engine()
sapien.VulkanRenderer.set_log_level("off")
_renderer = sapien.VulkanRenderer(default_mipmap_levels=1)
_engine.set_renderer(_renderer)

_engine.set_log_level('off')

class BaseEnv(Env):
    def __init__(self, 
                config_file, 
                obs_mode='state', 
                reward_type='dense', 
                frame_skip=5, 
                max_episode_steps=200, 
                variant_config={}, 
                override_model_file=None,
        ):
        self.set_env_mode(obs_mode, reward_type)
        self._engine = _engine
        self._renderer = _renderer

        self._setup_main_rng(seed=np.random.seed())
        self.variant_config = variant_config
        self.frame_skip = frame_skip

        self.yaml_config = preprocess(config_file)
        if override_model_file is not None:
            art_name, file_name = override_model_file
            self.override_models(art_name, file_name)

        self.simulation_frequency = self.yaml_config['physics']['simulation_frequency']
        self.timestep = 1 / self.simulation_frequency

        self._viewer = None

        self.configure_env()
        obs = self.reset(level=0)

        self.action_space = spaces.Box(low=-1, high=1, shape=self.agent.action_range().shape)
        self.observation_space = self._observation_to_space(obs)
        self.time_per_env_step = frame_skip / self.agent.control_frequency

        self._max_episode_steps = max_episode_steps

    def set_env_mode(self, obs_mode=None, reward_type=None):
        if obs_mode is not None:
            assert obs_mode in ['pointcloud', 'state', 'rgbd', 'custom']
            self.obs_mode = obs_mode
            if hasattr(self, 'observation_space') and self.observation_space is not None:
                obs = self.get_obs()
                self.observation_space = self._observation_to_space(obs)
        if reward_type is not None:
            assert reward_type in ['dense', 'sparse']
            self.reward_type = reward_type

    def configure_env(self):
        pass

    def override_models(self, art_name, file_name):
        _this_file = Path(__file__).resolve()
        new_model_file = _this_file.parent.joinpath(
            "../assets/config_files/{:s}".format(file_name)
        )
        yaml_models = preprocess(new_model_file)
        flag = False
        for art_cfg in self.yaml_config['layout']['articulations']:
            if art_cfg['name'] == art_name:
                art_cfg['_variants']['options'] = yaml_models
                flag = True
                break
        if not flag:
            raise Exception('Override models failed: {:s} not found in the yaml file.'.format(art_name))

    def reset_level(self):
        self.agent.reset()
        for articulation in self.articulations.values():
            articulation['articulation'].unpack(articulation['init_state'])
        for actor in self.actors.values():
            actor['actor'].unpack(actor['init_state'])
        return self.get_obs()

    def reset(self, level=None):
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
        self.level_config, self.level_variant_config = process_variants(
            config, self._level_rng, self.variant_config
        )

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

        # Cannot return obs right now because something will be determined in derived class
        # return self.get_obs() 

    def _init_eval_record(self):
        self.keep_good_steps = defaultdict(int)
        keep_good_time = 0.5
        self.keep_good_steps_threshold = int( np.ceil(keep_good_time * self.agent.control_frequency / self.frame_skip) ) 

    def _setup_main_rng(self, seed):
        self._main_seed = seed
        self._main_rng = np.random.RandomState(seed)

    def _setup_renderer(self):
        self._scene.renderer_scene.set_ambient_light(
            self.level_config['render']['ambient_light']['color']
        )
        for pl in self.level_config['render']['point_lights']:
            self._scene.renderer_scene.add_point_light(pl['position'], pl['color'])
        for dl in self.level_config['render']['directional_lights']:
            self._scene.renderer_scene.add_directional_light(dl['direction'], dl['color'])

    def _load_camera(self, cam_info):
        cam_info = deepcopy(cam_info)
        if 'parent' in cam_info:
            if cam_info['parent'] == 'robot':
                parent = self.agent.get_base_link()
            else:
                parent = self.objects[cam_info['parent']]
                if isinstance(parent, sapien.Articulation):
                    parent = parent.get_links()[0]
            camera_mount_actor = parent
            del cam_info['parent']
        else:
            camera_mount_actor = self._scene.create_actor_builder().build_kinematic()
        pose = sapien.Pose(cam_info['position'], cam_info['rotation'])
        del cam_info['position'], cam_info['rotation']
        camera = self._scene.add_mounted_camera(
            actor=camera_mount_actor, pose=pose, **cam_info, fovx=0
        )
        return camera
    
    def _setup_cameras(self):
        cameras = []
        for cam_info in self.level_config['render']['cameras']:
            if 'sub_cameras' in cam_info:
                sub_cameras = [self._load_camera(sub_cam_info) for sub_cam_info in cam_info['sub_cameras']]
                combined_camera = CombinedCamera(cam_info['name'], sub_cameras)
                cameras.append(combined_camera)
            else:
                camera = self._load_camera(cam_info)
                cameras.append(camera)
        self.cameras = cameras

    def _setup_physical_materials(self):
        self.physical_materials = {}
        if 'surface_materials' in self.level_config['layout']:
            for material in self.level_config['layout']['surface_materials']:
                m = self._scene.create_physical_material(
                    material['static_friction'],
                    material['dynamic_friction'],
                    material['restitution'],
                )
                self.physical_materials[material['name']] = m

    def _setup_render_materials(self):
        self.render_materials = {}
        if 'materials' in self.level_config['render']:
            for material in self.level_config['render']['materials']:
                m = self._renderer.create_material()
                m.set_roughness(material['roughness'])
                m.set_specular(material['specular'])
                m.set_metallic(material['metallic'])
                m.set_base_color(material['base_color'])
                self.render_materials[material['name']] = m

    def _load_articulations(self):
        self.articulations = {}
        if 'articulations' not in self.level_config['layout']:
            return
        for articulation_config in self.level_config['layout']['articulations']:
            if 'urdf_file' in articulation_config:
                urdf = articulation_config['urdf_file']
            else:
                self.selected_id = articulation_config['partnet_mobility_id']
                urdf = download_data(
                    articulation_config['partnet_mobility_id'],
                    directory=None
                )
                vhacd_urdf = Path(urdf).parent.joinpath('mobility_fixed.urdf')
                if vhacd_urdf.exists():
                    urdf = str(vhacd_urdf)
            loader = self._scene.create_urdf_loader()
            if articulation_config.get('multiple_collisions', False):
                loader.load_multiple_collisions_from_file = True
            loader.scale = articulation_config.get('scale', 1)
            loader.fix_root_link = articulation_config.get('fix_base', True)

            config = {}
            if 'surface_material' in articulation_config:
                config['material'] = self.physical_materials[articulation_config['surface_material']]
            if 'density' in articulation_config:
                config['density'] = articulation_config['density']
            articulation = loader.load(urdf, config=config)
            if 'initial_qpos' in articulation_config:
                articulation.set_qpos(articulation_config['initial_qpos'])
            articulation.set_root_pose(
                Pose(articulation_config['position'], articulation_config['rotation'])
            )
            articulation.set_name(articulation_config['name'])
            self.articulations[articulation_config['name']] = {
                'articulation': articulation,
                'init_state': articulation.pack(),
            }

    def _load_actors(self):
        self.actors = {}
        if 'rigid_bodies' in self.level_config['layout']:
            for actor in self.level_config['layout']['rigid_bodies']:
                self._load_actor_from_config(actor)

    def _load_actor_from_config(self, actor):
        # special case for ground
        if actor['parts'] and actor['parts'][0]['type'] == 'ground':
            shape = actor['parts'][0]

            builder: sapien.ActorBuilder = self._scene.create_actor_builder()
            builder.add_box_visual(
                pose=Pose(p=(0,0,-1), q=(0,0,0,1)), 
                half_size=[50, 50, 1], 
                material=self.render_materials['ground'],
            )
            visual_ground: sapien.Actor = builder.build_static(name='visual_ground')

            a = self._scene.add_ground(
                shape['altitude'],
                False,
                self.physical_materials[shape['surface_material']]
                if 'surface_material' in shape
                else None,
                self.render_materials[shape['render_material']]
                if 'render_material' in shape
                else None,
            )
            a.set_name(actor['name'])
            self.actors[actor['name']] = {'actor': a, 'init_state': a.pack()}

            # add ignore group 30 to ground
            s = a.get_collision_shapes()[0]
            gs = s.get_collision_groups()
            gs[2] = gs[2] | 1 << 30
            s.set_collision_groups(*gs)

            return

        # all other actors
        builder = self._scene.create_actor_builder()
        is_kinematic = actor['kinematic'] if 'kinematic' in actor else False
        if 'mass' in actor:
            assert 'inertia' in actor and 'center_of_mass' in actor
            builder.set_mass_and_inertia(
                actor['mass'],
                Pose(
                    actor['center_of_mass']['position'],
                    actor['center_of_mass']['rotation'],
                ),
                actor['inertia'],
            )
        for shape in actor['parts']:
            position = shape.get('position', [0, 0, 0])
            rotation = shape.get('rotation', [1, 0, 0, 0])
            assert 'type' in shape
            if shape['type'] in ['box', 'sphere', 'capsule']:
                if shape['collision']:
                    shape_func = getattr(builder, 'add_{}_collision'.format(shape['type']))
                    shape_func(
                        Pose(position, rotation),
                        shape['size'],
                        material=self.physical_materials[shape['material']],
                        density=shape['physical_density'],
                    )
                if shape['visual']:
                    visual_func = getattr(
                        builder, 'add_{}_visual'.format(shape['type'])
                    )
                    if 'render_material' in shape:
                        render_mat = self.render_materials[shape['render_material']]
                    else:
                        render_mat = self._renderer.create_material()
                        if 'color' in shape:
                            render_mat.set_base_color(shape['color'])
                    visual_func(Pose(position, rotation), shape['size'], render_mat)
            elif shape['type'] == 'mesh':
                if shape['collision']:
                    builder.add_multiple_collisions_from_file(
                        shape['file'],
                        Pose(position, rotation),
                        scale=shape['scale'],
                        density=shape['physical_density'],
                    )
                if shape['visual']:
                    builder.add_visual_from_file(
                        shape['file'], Pose(position, rotation), scale=shape['scale']
                    )
            else:
                raise NotImplementedError

        if is_kinematic:
            a = builder.build_kinematic()
        else:
            a = builder.build()
        a.set_name(actor['name'])
        a.set_pose(Pose(actor['position'], actor['rotation']))
        self.actors[actor['name']] = {'actor': a, 'init_state': a.pack()}

    def _setup_objects(self):
        self.objects = {}
        for k, v in self.actors.items():
            self.objects[k] = v['actor']
        for k, v in self.articulations.items():
            self.objects[k] = v['articulation']

    def _load_agent(self):
        agent_config = self.level_config['agent']
        if isinstance(agent_config, list):
            agents = []
            for config in agent_config:
                module_name, class_name = config['agent_class'].rsplit('.', 1)
                module = importlib.import_module(module_name)
                AgentClass = getattr(module, class_name)
                agents.append(AgentClass(self._engine, self._scene, config))
            self.agent = CombinedAgent(agents)
        else:
            module_name, class_name = agent_config['agent_class'].rsplit('.', 1)
            module = importlib.import_module(module_name)
            AgentClass = getattr(module, class_name)
            self.agent = AgentClass(self._engine, self._scene, agent_config)


        if self.simulation_frequency % self.agent.control_frequency != 0:
            warnings.warn(
                'Simulation frequency does not divide agent control frequency. The number of simulation step per control step will be rounded'
            )
        self.n_simulation_per_control_step = self.simulation_frequency // self.agent.control_frequency
        self.robot_link_ids = self.agent.get_link_ids()

    def _load_custom(self):
        self.custom = self.level_config['custom'] if 'custom' in self.level_config else None

    def _observation_to_space(self, obs):
        if self.obs_mode == 'state' or self.obs_mode == 'custom':
            low = np.full(obs.shape, -float('inf'), dtype=np.float32)
            high = np.full(obs.shape, float('inf'), dtype=np.float32)
            return spaces.Box(low, high, dtype=obs.dtype)
        elif self.obs_mode in ['rgbd', 'pointcloud']:
            agent_space = spaces.Box(
                -float('inf'),
                float('inf'),
                shape=obs['agent'].shape,
                dtype=obs['agent'].dtype,
            )
            ob_space = {}
            if 'rgb' in obs[self.obs_mode]:
                view_dict = obs[self.obs_mode]
                for view_type, view in view_dict.items():
                    ob_space[view_type] = spaces.Box(
                        low=-float('inf'), high=float('inf'), 
                        shape=view.shape, dtype=view.dtype,
                    )
            else:
                for camera_name, view_dict in obs[self.obs_mode].items():
                    ob_space[camera_name] = {}
                    for view_type, view in view_dict.items():
                        ob_space[camera_name][view_type] = spaces.Box(
                            low=-float('inf'), high=float('inf'), 
                            shape=view.shape, dtype=view.dtype,
                        )
            return {
                'agent': agent_space.shape,
                self.obs_mode: ob_space,
            }
        else:
            raise Exception('Unknown obs mode.')

    def _setup_viewer(self):
        # self._viewer.paused = True
        self._viewer.paused = False
        self._viewer.set_scene(self._scene)
        self._viewer.set_camera_xyz(-2, 0, 3)
        self._viewer.set_camera_rpy(0, -0.8, 0)

    def get_all_model_ids(self):
        if 'partnet_mobility_id' in self.variant_config:
            return [self.variant_config['partnet_mobility_id']]
        else:
            return self.all_model_ids

    def get_state(self, with_controller_state=True):
        actors, arts = self.get_all_objects_in_state()

        actors_state = [get_actor_state(actor) for actor in actors]
        arts_state = [get_pad_articulation_state(art, max_dof) for art, max_dof in arts]

        return np.concatenate( actors_state + arts_state + [
            self.get_additional_task_info(obs_mode='state'),
            self.agent.get_state(with_controller_state=with_controller_state),
        ])

    def get_all_objects_in_state(self):
        # return [actor_1, ...], [(art_1, max_dof), ..]
        raise NotImplementedError()

    def set_state(self, state):
        # set actors
        actors, arts = self.get_all_objects_in_state()
        for actor in actors:
            actor.set_pose(pose=Pose(p=state[:3], q=state[3:7]))
            actor.set_velocity(state[7:10])
            actor.set_angular_velocity(state[10:13])
            state = state[13:]

        # set articulations
        for art, max_dof in arts:
            art.set_root_pose(Pose(state[0:3], state[3:7]))
            art.set_root_velocity(state[7:10])
            art.set_root_angular_velocity(state[10:13])
            state = state[13:]
            # import pdb; pdb.set_trace()
            art.set_qpos(state[0:art.dof])
            art.set_qvel(state[max_dof:max_dof+art.dof])
            state = state[2*max_dof:]
        
        # skip task info
        task_info_len = len(self.get_additional_task_info(obs_mode='state'))
        state = state[task_info_len:]

        # set robot state
        self.agent.set_state(state)

        return self.get_obs()

    def get_additional_task_info(self, obs_mode):
        return np.array([])

    def get_custom_observation(self):
        raise NotImplementedError()

    def compute_dense_reward(self, action, state=None):
        raise NotImplementedError()

    def compute_eval_flag_dict(self):
        raise NotImplementedError()

    def _eval(self):
        flag_dict = self.compute_eval_flag_dict()
        eval_result_dict = {}
        for key, value in flag_dict.items():
            if value:
                self.keep_good_steps[key] += 1
            else:
                self.keep_good_steps[key] = 0
            eval_result_dict[key] = self.keep_good_steps[key] >= self.keep_good_steps_threshold
        return eval_result_dict, eval_result_dict['success']

    def _clip_and_scale_action(self, action): # from [-1, 1] to real action range
        action = np.clip(action, -1, 1)
        t = self.agent.action_range()
        action = 0.5 * (t.high - t.low) * action + 0.5 * (t.high + t.low) 
        return action

    def step(self, action):
        self.step_in_ep += 1
        processed_action = self._clip_and_scale_action(action)
        for __ in range(self.frame_skip):
            self.agent.set_action(processed_action.copy()) # avoid action being changed
            for _ in range(self.n_simulation_per_control_step):
                self.agent.simulation_step()
                self._scene.step()
        
        info = {}
        info['eval_info'], done = self._eval()

        if self.reward_type == 'sparse':
            reward = int(info['eval_info']['success'])
        elif self.reward_type == 'dense':
            reward, more_info = self.compute_dense_reward(action)
            info.update(more_info)
        else:
            raise NotImplementedError()
        obs = self.get_obs()

        if self.step_in_ep >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True

        if hasattr(self, 'running_to_time_limit') and self.running_to_time_limit:
            done = False
        return obs, reward, done, info

    def enable_running_to_time_limit(self):
        self.running_to_time_limit = True

    def check_actor_static(self, actor, max_v=None, max_ang_v=None):
        if self.step_in_ep <= 1:
            flag_v = (max_v is None) or (np.linalg.norm(actor.get_velocity()) <= max_v)
            flag_ang_v = (max_ang_v is None) or (np.linalg.norm(actor.get_angular_velocity()) <= max_ang_v)
        else:
            pose = actor.get_pose()
            t = self.time_per_env_step
            flag_v = (max_v is None) or ( np.linalg.norm( pose.p - self._prev_actor_pose.p ) <= max_v * t)
            flag_ang_v = (max_ang_v is None) or (angle_distance(self._prev_actor_pose, pose) <= max_ang_v * t)
        self._prev_actor_pose = actor.get_pose()
        return flag_v and flag_ang_v
    
    def render(self, mode='color_image', depth=False, seg=None, camera_names=None):
        self._scene.update_render()
        if mode == 'human':
            if self._viewer is None:
                self._viewer = Viewer(self._renderer)
                self._setup_viewer()
            self._viewer.render()
            return self._viewer
        else:
            if seg is not None:
                if seg == 'visual':
                    seg_idx = 0
                elif seg == 'actor':
                    seg_idx = 1
                elif seg == 'both':
                    seg_idx = [0, 1]
                else:
                    raise NotImplementedError()
            else:
                seg_idx = None
            if camera_names is None:
                cameras = self.cameras
            else:
                cameras = []
                for camera in self.cameras:
                    if camera.get_name() in camera_names:
                        cameras.append(camera)
            if mode == 'color_image' or mode == 'pointcloud':
                views = {}
                get_view_func = read_images_from_camera if mode == 'color_image' else read_pointclouds_from_camera
                for cam in cameras:
                    cam.take_picture()
                for cam in cameras:
                    if isinstance(cam, CombinedCamera):
                        view = cam.get_combined_view(mode, depth, seg_idx) # list of dict for image, dict for pointcloud
                    else:
                        view = get_view_func(cam, depth, seg_idx) # dict
                    views[cam.get_name()] = view
                return views

    def _post_process_view(self, view_dict):
        actor_id_seg = view_dict['seg'] # (n, m, 1)
        mask = np.zeros(actor_id_seg.shape, dtype=np.bool)
        for actor_id in self.robot_link_ids:
            mask = mask | ( actor_id_seg == actor_id )

        view_dict['seg'] = mask

    def get_obs(self, seg='actor', **kwargs): # seg can be 'visual', 'actor', 'both'
        if self.obs_mode == 'custom':
            return self.get_custom_observation()
        if self.obs_mode == 'state':
            return self.get_state(with_controller_state=False)
        elif self.obs_mode == 'rgbd':
            obs = {
                'agent': self.agent.get_state(with_controller_state=False),
                'rgbd': self.render(mode='color_image', depth=True, camera_names=['robot'], seg=seg, **kwargs),
            }
        elif self.obs_mode == 'pointcloud':
            obs = {
                'agent': self.agent.get_state(with_controller_state=False),
                'pointcloud': self.render(mode='pointcloud', camera_names=['robot'], seg=seg, **kwargs),
            }
        # post processing
        if self.obs_mode == 'pointcloud' or self.obs_mode == 'rgbd':
            views = obs[self.obs_mode] # views is also a dict, keys including 'robot', 'world', ...
            for cam_name, view in views.items():
                if isinstance(view, list):
                    for view_dict in view:
                        self._post_process_view(view_dict)
                    combined_view = {}
                    for key in view[0].keys():
                        combined_view[key] = np.concatenate([view_dict[key] for view_dict in view], axis=-1)
                    views[cam_name] = combined_view
                else: # view is a dict
                    self._post_process_view(view)
            if len(views) == 1:
                view = next(iter(views.values()))
                obs[self.obs_mode] = view
        return obs


    def close(self):
        if hasattr(self, '_viewer') and self._viewer:
            self._viewer.close()
        self._viewer = None
        self._scene = None

    def seed(self, seed=None):
        if seed is None:
            return self._main_seed
        else:
            self._setup_main_rng(seed=seed)

    def __del__(self):
        self.close()
