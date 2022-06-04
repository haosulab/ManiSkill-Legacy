import pdb
import numpy as np


import gym
import mani_skill.env


env = gym.make('OpenCabinetDrawerMagic-v0')

env.set_env_mode(obs_mode='state', reward_type='sparse')
# obs_mode can be 'state', 'pointcloud' or 'rgbd'
# reward_type can be 'sparse' or 'dense'
print(env.observation_space) # this shows the structure of the observation, openai gym's format
print(env.action_space) # this shows the action space, openai gym's format
env.reset(level=0)
goal = np.array([0.1,0.1,0.4, 0.04,0.04])
for i_step in range(200):
    # pdb.set_trace()
    # env.render('rgb_array')  # a display is required to use this function, rendering will slower the running speed
    cur_qpos = env.agent.robot.get_qpos()
    vec_robot_to_goal = goal - cur_qpos
    action = np.zeros(5)
    ## normalize
    action[:3] = vec_robot_to_goal[:3] / env.agent.action_range().high[0]
    action[3:] = 1 if vec_robot_to_goal[-1] > 0 else -1
    dist = np.linalg.norm(vec_robot_to_goal)


    obs, reward, done, info=env.step(action)  # take a random action
    print('{:d}: dist {:.4f}, cur_qpos {}, target_qpos {}'.format(i_step, dist, cur_qpos, goal))
    if done:
        break
env.close()
