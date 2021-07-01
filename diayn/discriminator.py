import collections

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

from utils import make_n_hidden_layers


class CrossEntropyDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_channels, hidden_layers, n_skills, hidden_nonlinearity=nn.LeakyReLU, output_nonlinearity=nn.Tanh):
        super().__init__()

        input_size = np.array(list(input_size)).flatten()[0]

        self.seq = nn.Sequential(
            nn.Linear(input_size, hidden_channels),
            hidden_nonlinearity(),
            *make_n_hidden_layers(hidden_layers, hidden_channels, hidden_nonlinearity)
        )

        self.out_layer = nn.Linear(hidden_channels, n_skills)
        self.out_nonlinearity = output_nonlinearity()
        self.trainer = CrossEntropyTrainer(self)
        self.n_skills = n_skills
        print(self)

    def forward(self, input):
        with torch.no_grad():   # preprocess doesn't have grads
            input = torch.tensor(input).cuda()

        hidden_output = self.seq(input)
        linear_output = self.out_layer(hidden_output)
        logits = self.out_nonlinearity(linear_output)
        return logits

    def get_score(self, input, labels, add_to):
        with torch.no_grad():
            logits = self(input).cpu()

            from torch.nn.functional import cross_entropy
            reward_pls = -1 * cross_entropy(logits, labels, reduction="none")
            reward_pls += add_to
        return reward_pls

    def train_on_batch(self, obss, labels, bad_obss):
        self.trainer.train(obss, labels, bad_obss)

class CrossEntropyTrainer():
    def __init__(self, discriminator):
        # NN stuff
        self.discriminator = discriminator
        self.daiyn_optimizer = torch.optim.Adam(discriminator.parameters(), lr=3E-4)  # todo pass LR etc
        self.daiyn_loss = nn.CrossEntropyLoss()
        self.losses = collections.deque(maxlen=600)

    def train(self, obss, labels, bad_obss):
        zs = labels.cuda()

        self.daiyn_optimizer.zero_grad()

        logits = self.discriminator(obss)
        loss = self.daiyn_loss(logits, zs)

        loss.backward()
        self.daiyn_optimizer.step()

        with torch.no_grad():
            self.losses.append(loss.cpu().numpy())

class BinaryEntropyDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_channels, hidden_layers, n_skills, hidden_nonlinearity=nn.Tanh, output_nonlinearity=nn.Sigmoid):
        super().__init__()

        input_size = np.array(list(input_size)).flatten()[0] + n_skills     # because we need to augment with skill

        self.seq = nn.Sequential(
            nn.Linear(input_size, hidden_channels),
            hidden_nonlinearity(),
            *make_n_hidden_layers(hidden_layers, hidden_channels, hidden_nonlinearity)
        )

        self.out_layer = nn.Linear(hidden_channels, 1)
        self.out_nonlinearity = nn.Identity() # output_nonlinearity() no sigmoid output because we're using BCEWithLogits
        print(self)
        self.trainer = BinaryCrossEntropyTrainer(self)

        self.n_skills = n_skills

    def forward(self, input):
        hidden_output = self.seq(input)
        linear_output = self.out_layer(hidden_output)
        logits = self.out_nonlinearity(linear_output)
        return logits

    def preprocess(self, obss, labels):
        from diayn.common import augment_obs
        obss = [augment_obs(obs, labels[i], self.n_skills) for i, obs in enumerate(obss)]
        obss = torch.tensor(obss).cuda().float()
        return obss

    def get_score(self, input, labels, add_to):
        with torch.no_grad():
            preprocessed = self.preprocess(input, labels)
            logits = self(preprocessed)
            scores = torch.sigmoid(logits) #torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones((len(labels),1)).cuda(), reduction="none")
            # "if the disciminator can tell who you are, you get a very bad reward, and if he can't, you get a better reward"
            reward_pls = torch.log(scores)

            reward_pls = reward_pls.squeeze()
            reward_pls += add_to

        return reward_pls

    def train_on_batch(self, obss, labels, bad_obss):
        good_obss = self.preprocess(obss, labels)
        bad_obss = self.preprocess(bad_obss, labels)
        self.trainer.train(good_obss, bad_obss)

class BinaryCrossEntropyTrainer():
    def __init__(self, discriminator):
        # NN stuff
        self.discriminator = discriminator
        self.daiyn_optimizer = torch.optim.Adam(discriminator.parameters(), lr=3E-4)  # todo pass LR etc
        self.daiyn_loss = nn.BCEWithLogitsLoss()
        self.losses = collections.deque(maxlen=600)

    def train(self, good_obss, bad_obss):
        # note: seems to do one training step for each env step https://github.com/haarnoja/sac/blob/8258e33633c7e37833cc39315891e77adfbe14b2/sac/algos/diayn.py#L446
        # ==> link training to env.step
        self.daiyn_optimizer.zero_grad()

        good_logits = self.discriminator(good_obss)
        bad_logits = self.discriminator(bad_obss)

        loss = self.daiyn_loss(good_logits, torch.ones((len(good_logits), 1)).cuda()) + \
               self.daiyn_loss(bad_logits, torch.zeros((len(bad_logits), 1)).cuda())

        loss.backward()
        self.daiyn_optimizer.step()
        with torch.no_grad():
            self.losses.append(loss.cpu().numpy())