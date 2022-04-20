import gym
import torch
import numpy as np

class BasicWrapper(gym.Wrapper):
    def __init__(self, env, function):
        super().__init__(env)
        self.env = env
        self.cur_state = None
        self.function = function

    def reset(self):
        obs = self.env.reset()
        self.cur_state = obs
        return obs

    def step(self, action):
        next_state, _, done, info = self.env.step(action)
        #combine action and state

        sa_pair = np.concatenate((self.cur_state, action))
        reward = -(self.function.forward(torch.tensor(sa_pair, dtype=torch.float)))
        self.cur_state = next_state

        return next_state, reward, done, info
