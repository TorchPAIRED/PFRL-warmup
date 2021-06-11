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


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_channels, hidden_layers, n_skills, hidden_nonlinearity=nn.LeakyReLU, output_nonlinearity=nn.Tanh):
        super().__init__()

        input_size = np.array(list(input_size)).flatten()[0]

        self.seq = nn.Sequential(
            nn.Linear(input_size, hidden_channels),
            nn.ReLU(),
            *make_n_hidden_layers(hidden_layers, hidden_channels)
        )

        self.out_layer = nn.Linear(hidden_channels, n_skills)
        self.out_nonlinearity = output_nonlinearity()
        print(self)

    def forward(self, input):
        with torch.no_grad():   # preprocess doesn't have grads
            input = torch.tensor(input).cuda()

        hidden_output = self.seq(input)
        linear_output = self.out_layer(hidden_output)
        logits = self.out_nonlinearity(linear_output)
        return logits