import torch
import numpy as np
from torch import nn

def weights_init(m):
    if isinstance(m, nn.Conv3d):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.xavier_uniform(m.bias.data)

    if isinstance(m, nn.Linear):
        y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,1/np.sqrt(y))
        # m.bias.data should be 0
        m.bias.data.fill_(0.01)
