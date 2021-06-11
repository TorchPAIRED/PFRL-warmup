from torch import nn


def make_n_hidden_layers(n, size):
    hiddens = []
    for i in range(n):
        hiddens.append(nn.Linear(size, size))
        hiddens.append(nn.ReLU())
    return hiddens