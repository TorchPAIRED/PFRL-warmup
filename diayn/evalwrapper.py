import gym
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from diayn.common import resize_obs_space, augment_obs


class DIAYNAugmentationWrapper(gym.ObservationWrapper):
    def __init__(self, env, augment_with_z, oh_len, out_dir):
        super().__init__(env)
        self.observation_space = resize_obs_space(env.observation_space, oh_len)
        self.augment_with_z = augment_with_z
        self.oh_len = oh_len

        self.used_paths = []
        self.out_dir = out_dir

        # like a reset
        self.reset_writer()

    def reset_writer(self):
        i = 0
        postfix = f"{i}"
        while postfix in self.used_paths:
            i += 1
            postfix = f"{i}"

        self.used_paths.append(postfix)

        self.action_logger = SummaryWriter(self.out_dir)
        self.reward_logger = SummaryWriter(self.out_dir)

        self.n_steps = 0

    def step(self, action):
        obs, rew, don, inf = super().step(action)

        self.action_logger.add_scalar(f"/action-{self.used_paths[-1]}", action, self.n_steps)
        self.reward_logger.add_scalar(f"/reward-{self.used_paths[-1]}", rew, self.n_steps)

        self.n_steps += 1
        return obs, rew, don, inf

    def reset(self, **kwargs):
        self.reset_writer()
        return super().reset(**kwargs)

    def observation(self, observation):
        return augment_obs(observation, self.augment_with_z, self.oh_len)