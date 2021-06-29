import torch
from torch import nn
from torch.nn import Module
from torchvision import models

class Randomize(Module):
    def __init__(self):
        super(Randomize, self).__init__()
        pass

    def forward(self, input):
        # >> input.shape 
        # [batch_num, width, height]
        output = torch.rand(input.shape[0], dtype=torch.float)
        return output

class SimpleCNN(Module):
    def __init__(self): 
        super(SimpleCNN, self).__init__()
        self.dropout    = nn.Dropout(0.2)
        self.maxpool    = nn.MaxPool2d(kernel_size=(2,2))
        self.relu       = nn.ReLU()

        self.conv1      = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2      = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3      = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)

        self.fc1        = nn.Linear(in_features=14*14*128, out_features=4096)
        self.fc2        = nn.Linear(in_features=4096, out_features=256)
        self.fc3        = nn.Linear(in_features=256, out_features=1)
        self.softmax    = nn.LogSoftmax(dim=1)


    def forward(self, input):
        out = self.relu(self.conv1(input))
        out = self.maxpool(out)
        out = self.relu(self.conv2(out))
        out = self.maxpool(out)
        out = self.relu(self.conv3(out))
        out = self.maxpool(out)

        out = out.view(-1, 14*14*128)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.relu(self.fc3(out))
        out = self.softmax(out)

        return out

def initialize_model(model_name, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft        = models.resnet18(pretrained=use_pretrained)
        if feature_extract:
            for param in model_ft.parameters():
                param.requires_grad = False 

        model_ft.conv1  = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model_ft.fc = nn.Sequential(
                        nn.Linear(512, 128),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, 1),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.LogSoftmax(dim=1))

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft