import torch
from torch.nn import Module

class Randomize(Module):
    def __init__(self):
        super(Randomize, self).__init__()


        pass

    def forward(self, input):
        # input.shape [batch_num, width, height]
        output = torch.randint(low=0, high=1, size=input.shape[0])
        return output