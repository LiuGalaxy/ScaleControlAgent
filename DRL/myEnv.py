import numpy as np
import gym
from gym import spaces
import os
import glob
from osgeo import gdal
from sklearn.metrics import jaccard_score, accuracy_score, f1_score
import cv2


class MinimalEnv(gym.Env):
    def __init__(self, n_actions, N_CHANNELS, THUMBSIZE):
        super(MinimalEnv, self).__init__()
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(N_CHANNELS + 1, THUMBSIZE, THUMBSIZE),
                                            dtype=np.float32)

        self.N_CHANNELS = N_CHANNELS
        self.THUMBSIZE = THUMBSIZE

    def step(self, action):
        return np.zeros(self.observation_space.shape), 0, False, {}

    def reset(self):
        return np.zeros(self.observation_space.shape)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
