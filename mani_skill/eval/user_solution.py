import numpy as np
import gym
import mani_skill.env
from gym.spaces import Box
import copy

class BasePolicy(object):
    def __init__(self, opts=None):
        self.obs_mode = 'pointcloud'

    def act(self, observation):
        raise NotImplementedError()

    def reset(self): # if you use an RNN-based policy, you need to implement this function
        pass

class RandomPolicy(BasePolicy):
    def __init__(self, env_name):
        super().__init__()
        env = gym.make(env_name)
        self.action_space = copy.copy(env.action_space)
        env.close()
        del env

    def act(self, state):
        return self.action_space.sample()

class UserPolicy(BasePolicy):
    def __init__(self, env_name):
        super().__init__()
        ##### Replace with your code
        env = gym.make(env_name)
        self.action_space = copy.copy(env.action_space)
        env.close()
        del env

        self.obs_mode = 'pointcloud' # remember to set this!
    
    def act(self, observation):
        ##### Replace with your code
        return self.action_space.sample()