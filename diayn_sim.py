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


class Discriminator(nn.Module):
    def __init__(self, input_size, layers, n_skills, hidden_nonlinearity=nn.LeakyReLU, output_nonlinearity=nn.Tanh):
        super().__init__()

        input_size = np.array(list(input_size)).flatten()

        layers = list(input_size) + list(layers)
        layers = list(zip(layers[0:], layers[1:]))
        torch_layers = []
        for first, second in layers:
            torch_layers.append(nn.Linear(first, second))
            torch_layers.append(hidden_nonlinearity())

        self.seq = nn.Sequential(
            *torch_layers
        )

        self.out_layer = nn.Linear(layers[-1][-1], n_skills)
        self.out_nonlinearity = output_nonlinearity()

    # note: seems to do one training step for each env step https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L446
    # ==> link training to env.step

    def forward(self, input):
        with torch.no_grad():   # this is useful for big inputs, allows us to maxpool them etc
            # todo maybe add user-defined preprocess here
            input = torch.tensor(input).cuda()

        hidden_output = self.seq(input)
        linear_output = self.out_layer(hidden_output)
        logits = self.out_nonlinearity(linear_output)
        return logits

class DIAYNWrapper(VectorEnvWrapper):
    def __init__(self, env, discriminator, n_skills):

        # The next lines spoof the original env
        super().__init__(env)
        self.env = env

        # need to augment by z, so this is one bigger
        # todo for more complicated envs, will need a better way to do this..
        obs_space = env.observation_space
        obs_space_shape = np.array(obs_space.shape).copy()
        obs_space_shape[-1] = obs_space_shape[-1]+1

        self.observation_space = gym.spaces.Box(obs_space.low[0], obs_space.high[0], tuple(list(obs_space_shape)), dtype=obs_space.dtype)

        self.action_space = env.action_space
        self.n_skills = n_skills

        # NN stuff
        self.discriminator = discriminator
        self.daiyn_optimizer = torch.optim.Adam(discriminator.parameters())  # todo pass LR etc
        self.daiyn_loss = nn.CrossEntropyLoss()
        self.latest_loss = None
        self.reset()    # enforces stuff that we do in reset

        self.top_extrinsic = 0

        # logging
        import logging
        self.logger = logging.getLogger(__name__)

    def train(self, obss):
        logits = self.discriminator(obss)
        loss = self.daiyn_loss(logits, self._z.cuda())  # fixme stuff will break because numpy arrays arent torch tensors, duh

        self.latest_loss = loss
        self.latest_logits = logits
        #
        #
        self.daiyn_optimizer.zero_grad()
        loss.backward()
        self.daiyn_optimizer.step()

    def call_logging(self):
        #self.logger.info(f"disc logits: {self.latest_logits}")
        with torch.no_grad():
            preds = torch.argmax(self.latest_logits, dim=1)
            self.logger.info(f"true z: {preds}")
        self.logger.info(f"disc z: {self._z}")
        self.logger.info(f"disc loss: {self.latest_loss}")
        self.logger.info(f"top extrinsic: {self.top_extrinsic}")

    def augment_obs(self, obs_index, obs):
        obs = np.append(obs, self._z[obs_index])
        obs = obs.astype(np.float32)
        return obs

    def step(self, action):
        obss, extrinsic_rews, dones, infos = self.env.step(action)

        #import matplotlib.pyplot as plt
        #plt.imshow(np.squeeze(obs,axis=0))
        #plt.show()

        # don't need rew, but keep it in info for logging.
        for info, extrinsic_rew in zip(infos, extrinsic_rews):
            info["extrinsic_reward"] = extrinsic_rew
            if extrinsic_rew > self.top_extrinsic:
                self.top_extrinsic = extrinsic_rew

        # see here for this logic https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L181
        # I think this is with no_grad() because the discriminator trains here https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L261
        # todo should this be inside no_grad(), or is this actually just how we get gradients for the discriminator?
        # todo if yes, we need to add up all these, and probably train the disc during self.reset or something
        with torch.no_grad():
            logits = self.discriminator(obss).cpu()    # disc takes s_{t+1}
            #print(logits)
            from torch.nn.functional import cross_entropy
            reward_pls = -1 * cross_entropy(logits, self._z, reduction="none")   # todo probably pass one_hot as part of action, and split it from action before self.env.step
            #print(reward_pls)
            # todo wtf is self._p_z_pl?
            # https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L184
            # isnt p_z a static?
            #z_one_hot = torch.zeros((self.n_skills,))
            #z_one_hot[self._z] = 1
            #_p_z_vector = torch.full((self.n_skills,), 1.0 / self.n_skills)
            #p_z = torch.sum(_p_z_vector*z_one_hot, axis=1) # ?? shouldnt this just be reward_pl[z]
            log_p_z = torch.log(torch.tensor(1.0/self.n_skills) + 1E-6)  # EPS is a magic number https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L21

            reward_pls -= log_p_z
        reward_pls = reward_pls.cpu().numpy()

        self.train(obss)

        #print(obss)

        obss = [self.augment_obs(i, obs) for i, obs in enumerate(obss)]

        #print(obss)

        return obss, reward_pls, dones, infos

    def reset(self, mask=None):

        obs = self.env.reset(mask)

        self._z = np.random.randint(0, self.n_skills, (len(obs),))   # todo could actually explore more than one z at once?
        self._z = torch.tensor(self._z)
        #print("old:",obs[0])
        obs = [self.augment_obs(i, ob) for i, ob in enumerate(obs)]
        #print("new:",obs[0])

        #logging.getLogger(__name__)
        return obs

