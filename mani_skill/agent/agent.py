import numpy as np
from gym import spaces
from sapien.core import Pose, Engine, Scene, Articulation
from mani_skill.agent.controllers import LPFilter, PIDController, VelocityController, PositionController
import yaml
import transforms3d
from mani_skill.utils.geometry import rotate_2d_vec_by_angle

class CombinedAgent:
    def __init__(self, agents):
        self.agents = agents

        self.control_frequency = self.agents[0].control_frequency
        for a in self.agents[1:]:
            assert a.control_frequency == self.control_frequency

        self.action_dims = [len(a.action_range().low) for a in self.agents]
        self.action_indices = np.concatenate(([0], np.cumsum(self.action_dims)))

        self.state_dims = [len(a.get_state()) for a in self.agents]
        self.state_dim = sum(self.state_dims)
        self.state_indices = np.concatenate(([0], np.cumsum(self.state_dims)))

    def action_range(self) -> spaces.Box:
        lows = []
        highs = []
        for agent in self.agents:
            agent_range: spaces.Box = agent.action_range()
            lows.append(agent_range.low)
            highs.append(agent_range.high)
        lows = np.concatenate(lows)
        highs = np.concatenate(highs)
        return spaces.Box(lows, highs)

    def set_action(self, action: np.ndarray):
        for i, agent in enumerate(self.agents):
            agent.set_action(action[self.action_indices[i]:self.action_indices[i+1]])

    def simulation_step(self):
        for agent in self.agents:
            agent.simulation_step()

    def get_state(self, by_dict=False):
        if by_dict:
            return [agent.get_state(by_dict=True) for agent in self.agents]
        else:
            return np.concatenate([agent.get_state(by_dict=False) for agent in self.agents])

    def get_ee_coords(self):
        return np.concatenate([i.get_ee_coords() for i in self.agents], 0)
    
    def set_state(self, state, by_dict=False):
        if by_dict:
            assert len(state) == len(self.agents)
            for state_each_agent, agent in zip(state, self.agents):
                agent.set_state(state_each_agent, by_dict=True)
        else:
            for i, agent in enumerate(self.agents):
                agent.set_state(state[self.state_indices[i]:self.state_indices[i+1]], by_dict=False)

    def reset(self):
        for agent in self.agents:
            agent.reset()


class Agent:
    def __init__(self, engine: Engine, scene: Scene, config):
        if type(config) == str:
            with open(config, 'r') as f:
                config = yaml.safe_load(f)['agent']

        self.config = config
        self._engine = engine
        self._scene = scene

        self.control_frequency = config['control_frequency']

        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = config['fix_base']
        loader.scale = config['scale']

        self._physical_materials = {}
        if config['surface_materials']:
            for mat in config['surface_materials']:
                self._physical_materials[mat['name']] = scene.create_physical_material(
                    mat['static_friction'], mat['dynamic_friction'], mat['restitution']
                )

        urdf_config = {'link': {}}
        for link in self.config['links']:
            link_props = {}
            if 'surface_material' in link:
                link_props['surface_material'] = self._physical_materials[link['surface_material']]
            if 'patch_radius' in link:
                link_props['patch_radius'] = link['patch_radius']
            if 'min_patch_radius' in link:
                link_props['min_patch_radius'] = link['min_patch_radius']
            urdf_config['link'][link['name']] = link_props

        self.robot = loader.load(config['urdf_file'], urdf_config)
        self.robot.set_name(self.config['name'])

        self.active_joints = self.robot.get_active_joints()

        self.balance_passive_force = config['balance_passive_force']
        
        self._init_state = self.robot.pack()

        self.all_joint_indices = [
            [x.name for x in self.robot.get_active_joints()].index(name)
            for name in self.config['all_joints']
        ]
        self.controllable_joint_indices = [
            [x.name for x in self.robot.get_active_joints()].index(name)
            for name in self.config['controllable_joints']
        ]

        assert (
            len(self.config['initial_qpos']) == self.robot.dof
        ), 'initial_qpos does not match robot DOF'

        qpos_reordered = np.zeros(self.robot.dof)
        qpos_reordered[self.all_joint_indices] = self.config['initial_qpos']
        self.robot.set_qpos(qpos_reordered)
        self.robot.set_root_pose(Pose(self.config['base_position'], self.config['base_rotation']))

        name2pxjoint = dict((j.get_name(), j) for j in self.robot.get_joints())
        name2config_joint = dict((j['name'], j) for j in config['joints'])

        for joint in config['joints']:
            assert (
                joint['name'] in name2pxjoint
            ), 'Unrecognized name in joint configurations'
            j = name2pxjoint[joint['name']]

            stiffness = joint.get('stiffness', 0)
            damping = joint['damping']
            friction = joint['friction']
            j.set_drive_property(stiffness, damping)
            j.set_friction(friction)

        controllers = []
        all_action_range = []
        for name in self.config['controllable_joints']:
            assert (
                name in name2config_joint
            ), 'Controllable joints properties must be configured'
            joint = name2config_joint[name]
            action_type = joint['action_type']
            action_range = joint['action_range']

            all_action_range.append(action_range)

            velocity_filter = None
            if 'velocity_filter' in joint:
                velocity_filter = LPFilter(
                    self.control_frequency, joint['velocity_filter']['cutoff_frequency']
                )
            if action_type == 'velocity':
                controller = VelocityController(velocity_filter)
            elif action_type == 'position':
                kp = joint['velocity_pid']['kp']
                ki = joint['velocity_pid']['ki']
                kd = joint['velocity_pid']['kd']
                limit = joint['velocity_pid']['limit']
                controller = PositionController(
                    PIDController(kp, ki, kd, self.control_frequency, limit),
                    velocity_filter,
                )
            else:
                raise RuntimeError('Only velocity or position are valid action types')
            controllers.append(controller)

        self.controllers = controllers
        all_action_range = np.array(all_action_range, dtype=np.float32)
        self._action_range = spaces.Box(all_action_range[:, 0], all_action_range[:, 1])
        self.num_ee = None
        self.full_state_len = None

    def action_range(self):
        return self._action_range

    def set_action(self, action: np.ndarray):
        assert action.shape == self._action_range.shape

        qpos = self.robot.get_qpos()
        qvel = self.robot.get_qvel()

        for j_idx, controller, target in zip(
            self.controllable_joint_indices, self.controllers, action
        ):
            if type(controller) == PositionController:
                output = controller.control(qpos[j_idx], target)
            elif type(controller) == VelocityController:
                output = controller.control(qvel[j_idx], target)
            else:
                raise Exception('this should not happen, please report it')
            self.active_joints[j_idx].set_drive_velocity_target(output)

    def simulation_step(self):
        if self.balance_passive_force:
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True, external=False
            )
            self.robot.set_qf(qf)

    def get_ee_coords(self):
        raise NotImplementedError()
    
    def get_ee_vels(self):
        raise NotImplementedError()

    def get_state(self, by_dict=False, with_controller_state=True):
        state_dict = {}
        fingers_pos = self.get_ee_coords().flatten()
        fingers_vel = self.get_ee_vels().flatten()
        qpos = self.robot.get_qpos()[self.all_joint_indices]
        qvel = self.robot.get_qvel()[self.all_joint_indices]
        state_dict = {
            'fingers_pos': fingers_pos,
            'fingers_vel': fingers_vel,
            'qpos': qpos,
            'qvel': qvel,
        }
        if with_controller_state:
            controller_state = []
            for controller in self.controllers:
                if type(controller) == PositionController:
                    n = controller.velocity_pid._prev_err is not None
                    controller_state.append(n)
                    if n:
                        controller_state.append(controller.velocity_pid._prev_err)
                    else:
                        controller_state.append(0)
                    controller_state.append(controller.velocity_pid._cum_err)
                    controller_state.append(controller.lp_filter.y)
                elif type(controller) == VelocityController:
                    controller_state.append(controller.lp_filter.y)
            state_dict['controller_state'] = np.array(controller_state)

        if by_dict:
            return state_dict
        else:
            return np.concatenate(list(state_dict.values()))
    
    def set_state(self, state, by_dict=False):
        if not by_dict:
            assert len(state) == self.full_state_len,\
                'length of state is not correct, probably because controller states are missing'
            state = state[self.num_ee*12:] # remove fingers_pos and fingers_vel
            state_dict = {
                'qpos': state[:self.robot.dof],
                'qvel': state[self.robot.dof:2*self.robot.dof],
                'controller_state': state[2*self.robot.dof:],
            }
        else:
            state_dict = state
        if 'qpos' in state_dict:
            qpos = np.zeros(self.robot.dof)
            qpos[self.all_joint_indices] = state_dict['qpos']
            self.robot.set_qpos(qpos)
        if 'qvel' in state_dict:
            qvel = np.zeros(self.robot.dof)
            qvel[self.all_joint_indices] = state_dict['qvel']
            self.robot.set_qvel(qvel)
        if 'controller_state' in state_dict:
            # idx = 2*self.robot.dof
            state = state_dict['controller_state']
            idx = 0
            for controller in self.controllers:
                if type(controller) == PositionController:
                    if state[idx]:
                        controller.velocity_pid._prev_err = state[idx+1]
                    else:
                        controller.velocity_pid._prev_err = None
                    controller.velocity_pid._cum_err = state[idx+2]
                    controller.lp_filter.y = state[idx+3]
                    idx = idx + 4
                elif type(controller) == VelocityController:
                    controller.lp_filter.y = state[idx]
                    idx = idx + 1

    def reset(self):
        self.robot.unpack(self._init_state)

    def get_link_ids(self):
        return [ link.get_id() for link in self.robot.get_links() ]


def concat_vec_in_dict(d, key_list):
    return np.concatenate([
        d[key] if isinstance(d[key], np.ndarray) else np.array([d[key]]) for key in key_list
    ])

class DummyMobileAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.active_joints[0].name == 'root_x_axis_joint'
        assert self.active_joints[1].name == 'root_y_axis_joint'
        assert self.active_joints[2].name == 'root_z_rotation_joint'

    def action_range(self):
        # coincidently, the action_range does not change
        # generally, we need to rotate some dimensions
        return self._action_range

    def _get_base_orientation(self):
        # active_joints[2] is root_z_rotation_joint
        return self.robot.get_qpos()[self.all_joint_indices[2]]

    def set_action(self, action: np.ndarray):
        # better not to in-place change action
        new_action = action.copy()
        ego_xy = new_action[:2]
        world_xy = rotate_2d_vec_by_angle(ego_xy, self._get_base_orientation())
        new_action[:2] = world_xy
        # import pdb; pdb.set_trace()
        # print('action in world:', world_xy, ' , ', world_xy[0] / world_xy[1] )
        super().set_action(new_action)

    def get_pose(self):
        qpos = self.robot.get_qpos()[self.all_joint_indices]
        x, y, theta = qpos[0], qpos[1], qpos[2]
        return Pose([x,y,0], transforms3d.euler.euler2quat(0,0,theta))

    def get_base_link(self):
        return self.robot.get_links()[3] # this is the dummy mobile base

    def get_state(self, by_dict=False, with_controller_state=True):
        state_dict = super().get_state(by_dict=True, with_controller_state=with_controller_state)
        qpos, qvel = state_dict['qpos'], state_dict['qvel']
        base_pos, base_orientation, arm_qpos = qpos[:2], qpos[2], qpos[3:]
        base_vel, base_ang_vel, arm_qvel = qvel[:2], qvel[2], qvel[3:]

        state_dict['qpos'] = arm_qpos
        state_dict['qvel'] = arm_qvel
        state_dict['base_pos'] = base_pos
        state_dict['base_orientation'] = base_orientation
        state_dict['base_vel'] = base_vel
        state_dict['base_ang_vel'] = base_ang_vel
        if by_dict:
            return state_dict
        else:
            # return concat_values(state_dict.values())
            key_list = [
                    'fingers_pos', 'fingers_vel',
                    'base_pos', 'base_orientation', 'base_vel', 'base_ang_vel', 
                    'qpos', 'qvel',
                ]
            if with_controller_state:
                key_list.append('controller_state')
            return concat_vec_in_dict(state_dict, key_list)

    def set_state(self, state, by_dict=False):
        if not by_dict:
            # if input is not dict, we need to rearrange the order before passing to super().set_state()
            assert len(state) == self.full_state_len,\
                'length of state is not correct, probably because controller states are missing'
            state = state[self.num_ee*12:] # remove fingers_pos and fingers_vel
            arms_dof = self.robot.dof - 3
            state_dict = {
                'base_pos': state[:2],
                'base_orientation': state[2:3],
                'base_vel': state[3:5], 
                'base_ang_vel': state[5],
                'qpos': state[6:6+arms_dof],
                'qvel': state[6+arms_dof:6+2*arms_dof],
                'controller_state': state[6+2*arms_dof:],
            }
        else:
            state_dict = state

        # another way
        new_state_dict = self.get_state(by_dict=True)
        new_state_dict.update(state_dict)
        new_state_dict['qpos'] = concat_vec_in_dict(new_state_dict, ['base_pos', 'base_orientation', 'qpos'])
        new_state_dict['qvel'] = concat_vec_in_dict(new_state_dict, ['base_vel', 'base_ang_vel', 'qvel'])
        super().set_state(new_state_dict, by_dict=True)

def get_actor_by_name(actors, names):
    assert isinstance(actors, (list, tuple))
    # Actors can be joint and link
    if isinstance(names, str):
        names = [names]
        sign = True
    else:
        sign = False
    ret = [None for _ in names]
    for actor in actors:
        if actor.get_name() in names:
            ret[names.index(actor.get_name())] = actor
    return ret[0] if sign else ret

class DummyMobileAdjustableHeightAgent(DummyMobileAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        body = [link for link in self.robot.get_links() if link.name == 'adjustable_body'][0]
        s = body.get_collision_shapes()[0]
        gs = s.get_collision_groups()
        gs[2] = gs[2] | 1 << 30  # ignore collision with ground
        s.set_collision_groups(*gs)

    
class MobileA2DualArmAgent(DummyMobileAdjustableHeightAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'mobile_a2_dual_arm'
        self.rfinger1_joint, self.rfinger2_joint, self.lfinger1_joint, self.lfinger2_joint = \
            get_actor_by_name(self.robot.get_joints(), 
                            ['right_panda_finger_joint1', 
                            'right_panda_finger_joint2', 
                            'left_panda_finger_joint1', 
                            'left_panda_finger_joint2'])
        self.rfinger1_link, self.rfinger2_link, self.lfinger1_link, self.lfinger2_link = \
            get_actor_by_name(self.robot.get_links(), 
                            ['right_panda_leftfinger', 
                            'right_panda_rightfinger', 
                            'left_panda_leftfinger', 
                            'left_panda_rightfinger'])
        self.rhand, self.lhand = get_actor_by_name(self.robot.get_links(), ['right_panda_hand', 'left_panda_hand'])
        self.num_ee = 2
        self.full_state_len = len(self.get_state(by_dict=False, with_controller_state=True))

    def get_ee_coords(self):
        finger_tips = [
            self.rfinger2_joint
            .get_global_pose()
            .transform(Pose([0, 0.035, 0]))
            .p,
            self.rfinger1_joint
            .get_global_pose()
            .transform(Pose([0, -0.035, 0]))
            .p,
            self.lfinger2_joint
            .get_global_pose()
            .transform(Pose([0, 0.035, 0]))
            .p,
            self.lfinger1_joint
            .get_global_pose()
            .transform(Pose([0, -0.035, 0]))
            .p,
        ]
        return np.array(finger_tips)
    
    def get_ee_vels(self):
        finger_vels = [
            self.rfinger1_link.get_velocity(),
            self.rfinger2_link.get_velocity(),
            self.lfinger1_link.get_velocity(),
            self.lfinger2_link.get_velocity(),
        ]
        return np.array(finger_vels)
    
    def get_ee_coords_sample(self):
        l = 0.035
        r = 0.052
        ret = []
        for i in range(10):
            x = (l * i + (4 - i) * r) / 4
            finger_tips = [
                self.rfinger2_joint
                .get_global_pose()
                .transform(Pose([0, x, 0]))
                .p,
                self.rfinger1_joint
                .get_global_pose()
                .transform(Pose([0, -x, 0]))
                .p,
                self.lfinger2_joint
                .get_global_pose()
                .transform(Pose([0, x, 0]))
                .p,
                self.lfinger1_joint
                .get_global_pose()
                .transform(Pose([0, -x, 0]))
                .p,
            ]
            ret.append(finger_tips)
        return np.array(ret).transpose((1, 0, 2))

class MobileA2SingleArmAgent(DummyMobileAdjustableHeightAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'mobile_a2_single_arm'
        self.finger1_joint, self.finger2_joint = get_actor_by_name(self.robot.get_joints(),
                                                       ['right_panda_finger_joint1', 'right_panda_finger_joint2'])
        self.finger1_link, self.finger2_link = get_actor_by_name(self.robot.get_links(),
                                                       ['right_panda_leftfinger', 'right_panda_rightfinger'])
        self.hand = get_actor_by_name(self.robot.get_links(), 'right_panda_hand')
        self.num_ee = 1
        self.full_state_len = len(self.get_state(by_dict=False, with_controller_state=True))

    def get_ee_vels(self):
        finger_vels = [
            self.finger2_link.get_velocity(),
            self.finger1_link.get_velocity(),
        ]
        return np.array(finger_vels)

    def get_ee_coords(self):
        finger_tips = [
            self.finger2_joint
                .get_global_pose()
                .transform(Pose([0, 0.035, 0]))
                .p,
            self.finger1_joint
                .get_global_pose()
                .transform(Pose([0, -0.035, 0]))
                .p,
        ]
        return np.array(finger_tips)

    def get_body_link(self):
        return self.robot.get_links()[6]

    def get_ee_coords_sample(self):
        l = 0.0355
        r = 0.052
        ret = []
        for i in range(10):
            x = (l * i + (4 - i) * r) / 4
            finger_tips = [
                self.finger2_joint
                    .get_global_pose()
                    .transform(Pose([0, x, 0]))
                    .p,
                self.finger1_joint
                    .get_global_pose()
                    .transform(Pose([0, -x, 0]))
                    .p,
            ]
            ret.append(finger_tips)
        return np.array(ret).transpose((1, 0, 2))
