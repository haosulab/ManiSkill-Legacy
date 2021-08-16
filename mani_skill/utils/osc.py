import os.path as osp
import numpy as np
import sapien.core as sapien
import yaml
from scipy.linalg import null_space

__this_folder__ = osp.dirname(__file__)

def nullspace_method(J, delta, regularization_strength=0.0):
    """
    Find solution of JX = delta
    """
    hess_approx = J.T.dot(J)
    joint_delta = J.T.dot(delta)
    if regularization_strength > 0:
        hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
        return np.linalg.solve(hess_approx, joint_delta)
    else:
        return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]

def get_link_by_name(robot, names):
    if isinstance(names, str):
        names = [names, ]
        single = True
    else:
        single = False
    ret = {}
    for link in robot.get_links():
        link_name = link.get_name()
        if link_name in names:
            ret[link_name] = link
    ret = [ret.get(name, None) for name in names]
    return ret if not single else ret[0]


def get_link_name_by_joint_indices(robot, indices):
    joints = robot.get_active_joints()
    return [joints[index].get_child_link().get_name() for index in indices]

def get_joint_names():
    single_joints = osp.join(__this_folder__, '../', 'assets/config_files/robots/mobile_a2_single_arm.yml')
    dual_joints = osp.join(__this_folder__, '../', 'assets/config_files/robots/mobile_a2_dual_arm.yml')
    single_joint_names = yaml.safe_load(open(single_joints, 'r'))['controllable_joints']
    dual_joint_names = yaml.safe_load(open(dual_joints, 'r'))['controllable_joints']
    return single_joint_names, dual_joint_names

class OperationalSpaceControlInterface(object):
    """
    Operational Space:
        Dimensions are different for single-arm robot and dual-arm robot.
        n_dim = joints that control the base of the robot (4) + joint of the fingers * num of arms (2 * num) + 6 * num
    The control signal is a 6D velocity relative to the robot hand.
    """

    def __init__(self, env_name):
        if 'MoveBucket' in env_name or 'PushChair' in env_name:
            self.n_arms = 2
            joint_names = get_joint_names()[1]
        elif 'Cabinet' in env_name:
            self.n_arms = 1
            joint_names = get_joint_names()[0]
        else:
            raise NotImplementedError('Env name is not recognized')

        self._engine = sapien.Engine()
        scene_config = sapien.SceneConfig()
        self._scene = self._engine.create_scene(scene_config)
        loader = self._scene.create_urdf_loader()
        loader.scale = 1
        loader.fix_root_link = True
        left_panda_file = osp.abspath(osp.join(__this_folder__, '../assets/robot/sciurus/A2_left.urdf'))
        right_panda_file = osp.abspath(osp.join(__this_folder__, '../assets/robot/sciurus/A2_right.urdf'))

        self.left_panda = loader.load(left_panda_file)
        self.right_panda = loader.load(right_panda_file)
        self.left_hand_index = get_link_by_name(self.left_panda, 'left_panda_hand').get_index()
        self.right_hand_index = get_link_by_name(self.right_panda, 'right_panda_hand').get_index()
        self.left_model = self.left_panda.create_pinocchio_model()
        self.right_model = self.right_panda.create_pinocchio_model()

        if self.n_arms == 2:
            self.left_arm_joints = [joint_names.index(joint.get_name()) for joint in
                                    self.left_panda.get_active_joints()]
        else:
            self.left_arm_joints = []

        self.right_arm_joints = [joint_names.index(joint.get_name()) for joint in self.right_panda.get_active_joints()]
        self.osc_extra_joints = [i for i, name in enumerate(joint_names)
                                 if 'left_panda_joint' not in name and 'right_panda_joint' not in name]

        self.left_arm_joints = np.array(self.left_arm_joints, dtype=np.uint8)
        self.right_arm_joints = np.array(self.right_arm_joints, dtype=np.uint8)
        self.osc_extra_joints = np.array(self.osc_extra_joints, dtype=np.uint8)

        self.right_arm_dim = len(self.right_arm_joints)
        self.left_arm_dim = len(self.left_arm_joints)
        self.null_space_dim = self.right_arm_dim + self.left_arm_dim
        self.osc_extra_dim = len(self.osc_extra_joints)
        self.osc_dim = self.osc_extra_dim + 6 * self.n_arms

        assert self.right_arm_dim + self.left_arm_dim + self.osc_extra_dim == len(joint_names)

    def get_J(self, qpos, mode='right'):
        if mode == 'right':
            qpos = qpos[self.right_arm_joints]
            return self.right_model.compute_single_link_local_jacobian(qpos, self.right_hand_index).T
        else:
            qpos = qpos[self.left_arm_joints]
            return self.left_model.compute_single_link_local_jacobian(qpos, self.left_hand_index).T

    def joint_space_to_operational_space_and_null_space(self, qpos, joint_space_action):
        # joint_space_action is a 13-dim or 22-dim vector
        osc_extra_action = joint_space_action[self.osc_extra_joints]

        rJ = self.get_J(qpos)
        r_action = nullspace_method(rJ, joint_space_action[self.right_arm_joints])
        r_null = joint_space_action[self.right_arm_joints] - rJ @ r_action

        if self.n_arms == 2:
            lJ = self.get_J(qpos, 'left')
            l_action = nullspace_method(lJ, joint_space_action[self.left_arm_joints])
            l_null = joint_space_action[self.left_arm_joints] - lJ @ l_action
            osc_action = np.concatenate([osc_extra_action, r_action, l_action])
            null_action = np.concatenate([r_null, l_null])
        else:
            osc_action = np.concatenate([osc_extra_action, r_action])
            null_action = r_null
        return osc_action, null_action

    def operational_space_and_null_space_to_joint_space(self, qpos, operational_space_action, null_space_action,
                                                        do_projection=True):
        """
        Hint:
        do_projection means if null_space_action is needed to be projected into null space. If you can guarantee
        null_space_action lie in the null space or you do not want to do projection, you can set it False.
        """
        assert len(operational_space_action) == self.osc_dim
        assert len(null_space_action) == self.null_space_dim

        final_action = qpos * 0
        final_action[self.osc_extra_joints] = operational_space_action[: len(self.osc_extra_joints)]
        arms = operational_space_action[len(self.osc_extra_joints):]
        len_right_arm = len(self.right_arm_joints)

        rJ = self.get_J(qpos)
        r_null_base = null_space(rJ.T)
        r_null = null_space_action[:len_right_arm]

        if do_projection:
            r_null = r_null_base @ (r_null @ r_null_base)

        final_action[self.right_arm_joints] = rJ @ arms[:6] + r_null
        if self.n_arms == 2:
            lJ = self.get_J(qpos, 'left')
            l_null_base = null_space(lJ.T)
            l_null = null_space_action[len_right_arm:]
            if do_projection:
                l_null = l_null_base @ (l_null @ l_null_base)
            final_action[self.left_arm_joints] = lJ @ arms[6:] + l_null
        return final_action

    def get_robot_qpos_from_obs(self, obs):
        if isinstance(obs, dict):
            agent_state = obs['agent']
        elif isinstance(obs, np.ndarray):
            len_agent_state = (4 + self.n_arms * 9) * 2 + self.n_arms * 12
            agent_state = obs[-len_agent_state:]
        else:
            raise NotImplementedError()
        s = agent_state
        s = s[self.n_arms * 12:]  # remove ee_pos and ee_vel
        s = s[6:]  # remove base pos and vel
        s = s[:(1 + 9 * self.n_arms)]  # remove qvel
        s = np.concatenate([np.zeros(3), s])  # append dummy base qpos
        return s


def test():
    import gym
    import mani_skill.env

    env_name = 'OpenCabinetDoor-v0'
    # env_name = 'PushChair-v0'

    env = gym.make(env_name)
    osc_interface = OperationalSpaceControlInterface(env_name)

    env.set_env_mode(obs_mode='state', reward_type='sparse')
    print(env.observation_space)  # this shows the structure of the observation, openai gym's format
    print(env.action_space)  # this shows the action space, openai gym's format

    for level_idx in range(0, 5):  # level_idx is a random seed
        obs = env.reset(level=level_idx)
        print('#### Level {:d}'.format(level_idx))
        for i_step in range(100000):
            print(i_step)
            action = env.action_space.sample()
            qpos = osc_interface.get_robot_qpos_from_obs(obs)

            joint_action = action
            os_action, null_action = osc_interface.joint_space_to_operational_space_and_null_space(qpos, joint_action)
            joint_action_rec = osc_interface.operational_space_and_null_space_to_joint_space(qpos, os_action,
                                                                                             null_action)
            epsilon = 1E-6
            if np.max(np.abs(joint_action_rec - action)) > epsilon:
                print('Reconstruct Error!', joint_action_rec, action)
                exit(-1)

            '''
            # Example 1: Move robot arm in null space
            null_action = osc_interface.operational_space_and_null_space_to_joint_space(
                qpos, np.zeros(osc_interface.osc_dim), action[:osc_interface.null_space_dim])
            action = null_action
            '''

            # Example 2: Move end effector along a specific direction
            hand_forward = np.zeros(osc_interface.osc_dim)
            extra_dim = len(osc_interface.osc_extra_joints)

            dim = 1  # move along x direction in end effector's frame
            hand_forward[extra_dim + dim:extra_dim + dim + 1] = 0.1  # 0.1 is the target velocity in velocity controller
            # hand_forward[extra_dim + dim + 6:extra_dim + dim + 7] = 0.1  # this is left arm when the task needs two arms
            forward_action = osc_interface.operational_space_and_null_space_to_joint_space(
                qpos, hand_forward, action[:osc_interface.null_space_dim])
            action = forward_action

            obs, reward, done, info = env.step(action)  # take a random action
            env.render('human')  # a display is required to use this function, rendering will slower the running speed
            if done:
                break
    env.close()


if __name__ == '__main__':
    test()
