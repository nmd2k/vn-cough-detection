import torch
import torch.nn as nn
from torch.nn import Module

class Randomize(Module):
    def __init__(self):
        super(Randomize, self).__init__()


        pass

    def forward(self, input):
        # input.shape [batch_num, width, height]
        output = torch.randint(low=0, high=1, size=input.shape[0])
        return output

class BaseLineRNN(Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=2, batch_first=True):
        super(BaseLineRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size=input_size, 
                        hidden_size=hidden_size, 
                        num_layers=num_layers,
                        batch_first=batch_first)
        
        self.classifier = nn.Sequential(
            nn.Linear(
                num_layers*hidden_size, 1
            ),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        h0 = torch.zeros(self.num_layers, inputs.shape[0], self.hidden_size).to(inputs.device)
        _, hn = self.rnn(inputs, h0)
        # # output(b x l x (h x num_layers)) ; hn(num_layers x b x h)
        
        flatten_hn = hn.reshape(hn.shape[1], -1) # Reshape hn to (b x (num_layers x h))
        output = self.classifier(flatten_hn)
        
        return output[:,0]
        
        