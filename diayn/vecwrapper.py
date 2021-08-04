import random

import gym
import torch
import torch.nn as nn
import numpy as np

# https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/examples/mujoco_all_diayn.py#L218
# https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/value_functions/value_function.py#L47
# https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/misc/mlp.py#L161
# https://github.com/haarnoja/sac/blob/master/examples/mujoco_all_diayn.py#L218 sizes are in "layer_size"
# So i think it's 288 -> 32 -> RELU -> 32 -> RELU -> 1 -> TANH
# seems to be tanh? https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/misc/mlp.py#L88
from pfrl.wrappers.vector_frame_stack import VectorEnvWrapper

from diayn.common import resize_obs_space, augment_obs

class DIAYNWrapper(VectorEnvWrapper):
    def __init__(self, env, discriminator, n_skills, is_eval=False):
        # The next lines spoof the original env
        super().__init__(env)
        self.env = env
        self.is_eval = is_eval

        # need to augment by z, so this is one bigger than the original space
        self.n_skills = n_skills

        # NN stuff
        self.discriminator = discriminator

        # logging
        import logging
        self.logger = logging.getLogger(__name__)

        self._z = None
        self.reset()

        self.buffers = []
        for _ in range(self.n_skills):
            import collections
            self.buffers.append(collections.deque(maxlen=600))

    def step(self, action):
        obss, extrinsic_rews, dones, infos = self.env.step(action)
        # don't need rew, but keep it in info for logging.
        for i, (info, extrinsic_rew) in enumerate(zip(infos, extrinsic_rews)):
            info["extrinsic_reward"] = extrinsic_rew
            self.extrinsic_rew_counter[i] += extrinsic_rew

        for obs, z in zip(obss, self._z):
            self.buffers[z].append(obs)

        # see here for this logic https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L181
        # I think this is with no_grad() because the discriminator trains here https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L261
        reward_pls = self.discriminator.get_score(obss, self._z, self.n_skills)
        reward_pls = reward_pls.cpu().numpy()

        if not self.is_eval:
            self.discriminator.train_on_batch(obss, self._z, self.get_bad_obss())

        return obss, reward_pls, dones, infos

    def get_bad_obss(self):
        bad_obss = []
        debug_bad_zs = [] # only for debugging purposes

        bad_zs = np.arange(0, self.n_skills)
        np.random.shuffle(bad_zs)

        for i in range(self.env.num_envs):
            bad_zs = np.append(bad_zs, self._z)

        for i in range(self.n_skills + self.env.num_envs ** 2):
            z = bad_zs[i]

            if self._z[len(bad_obss)] == z:  # needs to NOT be the current z for this index
                continue

            buffer = self.buffers[z]
            maxlen = len(buffer)
            if maxlen == 0:
                continue

            index = np.random.randint(0, maxlen)
            bad_obss.append(buffer[index])
            debug_bad_zs.append(z)
            if len(bad_obss) == self.env.num_envs:
                break

        return np.array(bad_obss)

    def reset(self, mask=None):
        obs = self.env.reset(mask)

        must_reset = np.logical_not(mask)

        all_zs = np.arange(self.n_skills)
        np.random.shuffle(all_zs)
        selected_z = all_zs[:len(obs)]
        selected_z = torch.tensor(selected_z)

        if self._z is None or mask is None:
            self._z = selected_z
            self.extrinsic_rew_counter = np.zeros((self.n_skills,))
        elif True in must_reset:
            self.extrinsic_rew_counter[self._z[must_reset]] = 0
            self._z[must_reset] = selected_z[must_reset]    # note: might introduce same zs

        return obs

class ZAugmentationVecWrapper(VectorEnvWrapper):
    def __init__(self, env, augmentation_len):
        super().__init__(env)
        self.augmentation_len = augmentation_len
        self.observation_space = resize_obs_space(env.observation_space, augmentation_len)

    def augment(self, obs):
        return [augment_obs(ob, self.env._z[i], self.augmentation_len) for i, ob in enumerate(obs)]

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return self.augment(obs), rew, done, info

    def reset(self, mask=None):
        return self.augment(self.env.reset(mask))