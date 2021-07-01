from torch import nn


def make_n_hidden_layers(n, size, non_linearity):
    hiddens = []
    for i in range(n):
        hiddens.append(nn.Linear(size, size))
        hiddens.append(non_linearity())
    return hiddens