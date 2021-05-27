import gym
import torch
import torch.nn as nn

from torch.nn.functional import cross_entropy

# https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/examples/mujoco_all_diayn.py#L218
# https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/value_functions/value_function.py#L47
# https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/misc/mlp.py#L161
# https://github.com/haarnoja/sac/blob/master/examples/mujoco_all_diayn.py#L218 sizes are in "layer_size"
# So i think it's 288 -> 32 -> RELU -> 32 -> RELU -> 1 -> TANH
# seems to be tanh? https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/misc/mlp.py#L88
class Discriminator(nn.Module):
    def __init__(self, layers, n_skills, hidden_nonlinearity=nn.ReLU, output_nonlinearity=nn.Tanh):
        super().__init__()

        layers = zip(layers[0:], layers[1:])
        torch_layers = []
        for first, second in layers:
            torch_layers.append(nn.Linear(first, second))
            torch_layers.append(hidden_nonlinearity())

        self.seq = nn.Sequential(
            *torch_layers
        )

        self.out_layer = nn.Linear(torch_layers[-1][-1], n_skills)
        self.out_nonlinearity = output_nonlinearity()

    # note: seems to do one training step for each env step https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L446
    # ==> link training to env.step

    def forward(self, input):
        hidden_output = self.seq(input)
        linear_output = self.out_layer(hidden_output)
        logits = self.out_nonlinearity(linear_output)
        return logits

class DIAYNEnv(gym.Env):
    def __init__(self,
                 env,
                 disc_hidden_layers,
                 n_skills,
                 all_the_random_discriminator_hparams=None #todo
                 ):

        # The next lines spoof the original env
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # NN stuff
        self.discriminator = Discriminator(disc_hidden_layers, n_skills)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters())  # todo pass LR etc
        self.loss_function = nn.CrossEntropyLoss()

    def train_discriminator(self, obs, one_hot_z):
        # todo will this do some weird stuff because of vectorized envs? should we train outside of the env instead?
        # as in, could just get the obs batch from the vectorized envs, slap the disc on it, backprop
        self.optimizer.zero_grad()
        logits = self.discriminator(obs)
        loss = self.loss_function(logits, one_hot_z)    # fixme stuff will break because numpy arrays arent torch tensors, duh
        loss.backward()
        self.optimizer.step()

    def step(self, action):
        obs, extrinsic_rew, done, info = self.env.step(action)

        # don't need rew, but keep it in info for logging.
        info["extrinsic_reward"] = extrinsic_rew

        # see here for this logic https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L181
        # I think this is with no_grad() because the discriminator trains here https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L261
        # todo should this be inside no_grad(), or is this actually just how we get gradients for the discriminator?
        # todo if yes, we need to add up all these, and probably train the disc during self.reset or something
        with torch.no_grad():
            logits = self.discriminator(obs)    # disc takes s_{t+1}
            reward_pl = -1 * cross_entropy(logits, one_hot_z)   # todo probably pass one_hot as part of action, and split it from action before self.env.step

            # todo wtf is self._p_z_pl?
            # https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L184
            # isnt p_z a static?
            p_z = torch.sum(self._p_z_pl*one_hot_z, axis=1) # ?? shouldnt this just be reward_pl[z]
            log_p_z = torch.log(p_z + 1E-6)  # EPS is a magic number https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L21

            reward_pl -= log_p_z

        reward_pl = reward_pl.cpu().numpy()[0]

        return obs, reward_pl, done, info

    def reset(self):
        return self.env.reset()

