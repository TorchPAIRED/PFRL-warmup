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
    def __init__(self, env, discriminator, n_skills, alpha, is_evaluator=False, oh_concat=False):
        # The next lines spoof the original env
        super().__init__(env)
        self.env = env
        self.is_evaluator = is_evaluator

        # need to augment by z, so this is one bigger than the original space
        self.oh_len = 1 if oh_concat is False else n_skills
        self.observation_space = resize_obs_space(env.observation_space, self.oh_len)
        self.n_skills = n_skills

        # NN stuff
        self.discriminator = discriminator
        self.alpha = alpha

        # logging
        import logging
        self.logger = logging.getLogger(__name__)   # todo unused for the moment, might make this into a tensorboard thingy
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
            self.buffers[self._z[i]].append(obss[i])

        # see here for this logic https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L181
        # I think this is with no_grad() because the discriminator trains here https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L261
        reward_pls = self.discriminator.get_score(obss, self._z, torch.log(torch.tensor(1.0/self.n_skills) + 1E-6))
        reward_pls = reward_pls.cpu().numpy()

        augmented_obss = [augment_obs(obs, self._z[i], self.oh_len) for i, obs in enumerate(obss)]

        if self.is_evaluator:   # when is an evaluator, just return the extrinsic rewards, no need for intrisic. Also, dont train
            return augmented_obss, extrinsic_rews, dones, infos

        self.train(obss)

        return augmented_obss, reward_pls, dones, infos

    def train(self, obss):
        bad_obss = []
        bad_zs = np.arange(0, self.n_skills) ; np.random.shuffle(bad_zs)
        for i in range(self.env.num_envs):
            bad_zs = np.append(bad_zs, self._z)

        for i in range(self.n_skills + self.env.num_envs**2):
            z = bad_zs[i]

            if self._z[len(bad_obss)] == z: # needs to NOT be the current z for this index
                continue

            buffer = self.buffers[z]
            maxlen = len(buffer)
            if maxlen == 0:
                continue

            index = np.random.randint(0, maxlen)
            bad_obss.append(buffer[index])
            if len(bad_obss) == len(obss):
                break

        bad_obss = np.array(bad_obss)

        self.discriminator.train_on_batch(obss, self._z, bad_obss)

    def reset(self, mask=None, force_z=False):
        obs = self.env.reset(mask)

        must_reset = np.logical_not(mask)

        z = torch.randint(0, self.n_skills, (len(obs),))
        if self._z is None or mask is None:
            self._z = z
        elif True in must_reset:
            self._z[must_reset] = z[must_reset]

        if force_z is not False and (True in (force_z != -1)):
            force_z = torch.tensor(force_z)
            self._z[force_z != -1] = force_z[force_z != -1]

        """
        # todo temp
        self._z[0] = 0
        self._z[1] = 1
        self._z[2] = 2
        self._z[3] = 3
        """

        obs = [augment_obs(ob, self._z[i], self.oh_len) for i, ob in enumerate(obs)]

        return obs


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


class DIAYNVecEval(VectorEnvWrapper):
    def __init__(self, env, n_skills, discriminator, oh_concat=False):
        # The next lines spoof the original env
        super().__init__(env)
        self.env = env
        self.discriminator = discriminator

        # need to augment by z, so this is one bigger than the original space
        self.oh_len = 1 if oh_concat is False else n_skills
        self.observation_space = resize_obs_space(env.observation_space, self.oh_len)
        self.n_skills = n_skills


        # logging
        import logging
        self.logger = logging.getLogger(
            __name__)  # todo unused for the moment, might make this into a tensorboard thingy
        self._z = None
        self.reset()

        self.rew_counter = np.zeros((n_skills,))



    def step(self, action):
        obss, extrinsic_rews, dones, infos = self.env.step(action)

        # don't need rew, but keep it in info for logging.
        for i, (info, extrinsic_rew) in enumerate(zip(infos, extrinsic_rews)):
            info["extrinsic_reward"] = extrinsic_rew

        augmented_obss = [augment_obs(obs, self._z[i], self.oh_len) for i, obs in enumerate(obss)]

        self.rew_counter += np.array(extrinsic_rews)

        reward_pls = self.discriminator.get_score(obss, self._z, torch.log(torch.tensor(1.0 / self.n_skills) + 1E-6))
        reward_pls = reward_pls.cpu().numpy()

        return augmented_obss, reward_pls, dones, infos

    def reset(self, mask=None):
        obs = self.env.reset(mask)

        if self._z is None or mask is None:
            self._z = torch.arange(self.n_skills)
            self.rew_counter = np.zeros((self.n_skills,))

        obs = [augment_obs(ob, self._z[i], self.oh_len) for i, ob in enumerate(obs)]

        return obs