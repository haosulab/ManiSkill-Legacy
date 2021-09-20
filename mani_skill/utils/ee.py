import os.path as osp
import numpy as np
import sapien.core as sapien
import yaml

__this_folder__ = osp.dirname(__file__)


class EndEffectorInterface(object):
    def __init__(self, env_name):
        if 'MoveBucket' in env_name or 'PushChair' in env_name:
            self.n_arms = 2
        elif 'Cabinet' in env_name:
            self.n_arms = 1
        else:
            raise NotImplementedError('Env name is not recognized')

        self._engine = sapien.Engine()
        scene_config = sapien.SceneConfig()
        self._scene = self._engine.create_scene(scene_config)
        loader = self._scene.create_urdf_loader()
        loader.scale = 1
        loader.fix_root_link = True

        if self.n_arms == 1:
            self.robot = loader.load(osp.abspath(osp.join(__this_folder__, '../assets/robot/sciurus/A2_single.urdf')))
        else:
            self.robot = loader.load(osp.abspath(osp.join(__this_folder__, '../assets/robot/sciurus/A2.urdf')))

    def get_ee_pose_from_obs(self, obs):
        qpos = self.get_robot_qpos_from_obs(obs)
        self.robot.set_qpos(qpos)
        links = self.robot.get_links()
        if self.n_arms == 1:
            return links[15].pose
        else:
            return links[15].pose, links[26].pose # right, left

    def get_robot_qpos_from_obs(self, obs):
        if isinstance(obs, dict):
            agent_state = obs['agent']
        elif isinstance(obs, np.ndarray):
            len_agent_state = (4 + self.n_arms * 9) * 2 + self.n_arms * 12
            agent_state = obs[-len_agent_state:]
        else:
            raise NotImplementedError()
        s = agent_state
        s = s[self.n_arms * 12:]  # remove fingers_pos and fingers_vel
        qpos_mobile_base = s[:3]
        s = s[6:]  # remove base pos and vel
        s = s[:(1 + 9 * self.n_arms)]  # remove qvel
        s = np.concatenate([qpos_mobile_base, s])
        return s


def test():
    import gym
    import mani_skill.env

    env_name = 'OpenCabinetDoor-v0'
    # env_name = 'PushChair-v0'

    env = gym.make(env_name)
    ee_interface = EndEffectorInterface(env_name)

    env.set_env_mode(obs_mode='state', reward_type='sparse')
    print(env.observation_space)  # this shows the structure of the observation, openai gym's format
    print(env.action_space)  # this shows the action space, openai gym's format

    epsilon = 1E-6
    for level_idx in range(0, 5):  # level_idx is a random seed
        obs = env.reset(level=level_idx)
        print('#### Level {:d}'.format(level_idx))
        for i_step in range(100000):
            print(i_step)
            action = env.action_space.sample()

            ee_pose = ee_interface.get_ee_pose_from_obs(obs)
            # right_ee_pose, left_ee_pose = ee_interface.get_ee_pose_from_obs(obs) # dual-arm case
            gt_ee_pose = env.agent.robot.get_links()[15].pose # user cannot access env during inference time

            if np.max(np.abs(ee_pose.p - gt_ee_pose.p)) > epsilon or \
                ( np.max(np.abs(ee_pose.q - gt_ee_pose.q)) > epsilon and np.max(np.abs(ee_pose.q + gt_ee_pose.q)) > epsilon ):
                print('EE pose is not correct.')
                print(ee_pose)
                print(gt_ee_pose)
                exit(-1)

            obs, reward, done, info = env.step(action)  # take a random action
            # env.render('human')  # a display is required to use this function, rendering will slower the running speed
            if done:
                break
    env.close()


if __name__ == '__main__':
    test()
