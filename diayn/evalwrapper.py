import gym
import torch
import torch.nn as nn
import numpy as np

from diayn.common import resize_obs_space, augment_obs


class DIAYNAugmentationWrapper(gym.ObservationWrapper):
    def __init__(self, env, augment_with_z):
        super().__init__(env)
        self.observation_space = resize_obs_space(env.observation_space)
        self.augment_with_z = augment_with_z

    def observation(self, observation):
        return augment_obs(observation, self.augment_with_z)