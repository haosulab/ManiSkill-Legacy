import gym
import mani_skill.env

env = gym.make('OpenCabinetDoor-v0')
# full environment list can be found in available_environments.txt

env.set_env_mode(obs_mode='state', reward_type='sparse')
# obs_mode can be 'state', 'pointcloud' or 'rgbd'
# reward_type can be 'sparse' or 'dense'
print(env.observation_space) # this shows the structure of the observation, openai gym's format
print(env.action_space) # this shows the action space, openai gym's format

for level_idx in range(0, 5): # level_idx is a random seed
    obs = env.reset(level=level_idx)
    print('#### Level {:d}'.format(level_idx))
    for i_step in range(100000):
        # env.render('human') # a display is required to use this function, rendering will slower the running speed
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action) # take a random action
        print('{:d}: reward {:.4f}, done {}'.format(i_step, reward, done))
        if done:
            break
env.close()
