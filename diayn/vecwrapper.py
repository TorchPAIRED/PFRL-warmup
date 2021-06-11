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
        self.daiyn_optimizer = torch.optim.Adam(discriminator.parameters(), lr=3E-4)  # todo pass LR etc
        self.daiyn_loss = nn.CrossEntropyLoss()
        self.alpha = alpha

        # logging
        import logging
        self.logger = logging.getLogger(__name__)   # todo unused for the moment, might make this into a tensorboard thingy
        self._z = None
        self.reset()
        """
        self.buffers = []
        for _ in range(self.n_skills):
            import collections
            self.buffers.append(collections.deque(maxlen=1000))
"""
    def train(self, obss):
       # obs

        # note: seems to do one training step for each env step https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L446
        # ==> link training to env.step
        self.daiyn_optimizer.zero_grad()

        logits = self.discriminator(obss)
        loss = self.daiyn_loss(logits, self._z.cuda())

        loss.backward()
        self.daiyn_optimizer.step()

    def step(self, action):
        obss, extrinsic_rews, dones, infos = self.env.step(action)

        # don't need rew, but keep it in info for logging.
        for i, (info, extrinsic_rew) in enumerate(zip(infos, extrinsic_rews)):
            info["extrinsic_reward"] = extrinsic_rew

        # see here for this logic https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L181
        # I think this is with no_grad() because the discriminator trains here https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L261
        with torch.no_grad():
            logits = self.discriminator(obss).cpu()

            from torch.nn.functional import cross_entropy
            reward_pls = -1 * cross_entropy(logits, self._z, reduction="none")

            log_p_z = torch.log(torch.tensor(1.0/self.n_skills) + 1E-6)  # EPS is a magic number https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L21
            reward_pls -= log_p_z
            reward_pls *= self.alpha

        reward_pls = reward_pls.cpu().numpy()

        augmented_obss = [augment_obs(obs, self._z[i], self.oh_len) for i, obs in enumerate(obss)]

        if self.is_evaluator:   # when is an evaluator, just return the extrinsic rewards, no need for intrisic. Also, dont train
            return augmented_obss, extrinsic_rews, dones, infos

        for obs, z in zip(obss, self._z):
            self.buffers[z].append(obs)

        self.train(obss)# if False else self.buffers)

        return augmented_obss, reward_pls, dones, infos

    def reset(self, mask=None):
        obs = self.env.reset(mask)

        must_reset = np.logical_not(mask)

        z = torch.randint(0, self.n_skills, (len(obs),))
        if self._z is None or mask is None:
            self._z = z
        elif True in must_reset:
            self._z[must_reset] = z[must_reset]

        obs = [augment_obs(ob, self._z[i], self.oh_len) for i, ob in enumerate(obs)]

        return obs