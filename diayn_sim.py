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
class Discriminator(nn.Module):
    def __init__(self, input_size, layers, n_skills, hidden_nonlinearity=nn.ReLU, output_nonlinearity=nn.Tanh, preprocess=None):
        super().__init__()
        self.preprocess = preprocess

        layers = [input_size] + list(layers)
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
            if self.preprocess is not None:
                input = self.preprocess(input)
            input = torch.tensor(input).cuda()

        hidden_output = self.seq(input)
        linear_output = self.out_layer(hidden_output)
        logits = self.out_nonlinearity(linear_output)
        return logits

class DIAYNWrapper(gym.RewardWrapper):
    def __init__(self, env, discriminator, n_skills, augment_obs_func=None):

        # The next lines spoof the original env
        super().__init__(env)
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_skills = n_skills
        self.augment_obs_func = augment_obs_func

        # NN stuff
        self.discriminator = discriminator
        self.optimizer = torch.optim.Adam(self.discriminator.parameters())  # todo pass LR etc
        self.loss_function = nn.CrossEntropyLoss()

        self.reset()    # enforces stuff that we do in reset

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

        import matplotlib.pyplot as plt
        plt.imshow(np.squeeze(obs,axis=0))
        plt.show()

        # don't need rew, but keep it in info for logging.
        info["extrinsic_reward"] = extrinsic_rew

        # see here for this logic https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L181
        # I think this is with no_grad() because the discriminator trains here https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L261
        # todo should this be inside no_grad(), or is this actually just how we get gradients for the discriminator?
        # todo if yes, we need to add up all these, and probably train the disc during self.reset or something
        with torch.no_grad():


            logits = self.discriminator(obs)    # disc takes s_{t+1}
            from torch.nn.functional import cross_entropy
            z_tensor = torch.tensor([self._z])
            reward_pl = -1 * cross_entropy(logits, z_tensor)   # todo probably pass one_hot as part of action, and split it from action before self.env.step

            # todo wtf is self._p_z_pl?
            # https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L184
            # isnt p_z a static?
            z_one_hot = torch.zeros((self.n_skills,))
            z_one_hot[self._z] = 1
            #_p_z_vector = torch.full((self.n_skills,), 1.0 / self.n_skills)
            #p_z = torch.sum(_p_z_vector*z_one_hot, axis=1) # ?? shouldnt this just be reward_pl[z]
            log_p_z = torch.log(z_tensor[0] + 1E-6)  # EPS is a magic number https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L21

            reward_pl -= log_p_z

        reward_pl = reward_pl.cpu().numpy()

        if self.augment_obs_func is not None:
            obs = self.augment_obs_func(obs, self._z)
            import matplotlib.pyplot as plt
            plt.imshow(np.squeeze(obs,axis=0))
            plt.show()

        return obs, reward_pl, done, info

    def reset(self):
        self._z = np.random.randint(0, self.n_skills)
        return self.env.reset()

