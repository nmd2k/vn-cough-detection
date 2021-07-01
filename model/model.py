import torch
from torch import nn
from torch.nn import Module
from torchvision import models
import torch.nn.functional as F
from model.common import weights_init

class Randomize(Module):
    def __init__(self):
        super(Randomize, self).__init__()
        pass

    def forward(self, input):
        # >> input.shape 
        # [batch_num, width, height]
        output = torch.rand(input.shape[0], dtype=torch.float)
        return output

class TwoResnetWay(Module):
    def __init__(self, model_name, pretrain=False, dropout_rate=0.2):
        super(TwoResnetWay, self).__init__()
        
        if '18' in model_name:
            res1        = models.resnet18(pretrained=pretrain)
            res2        = models.resnet18(pretrained=pretrain)
            
        if '34' in model_name:
            res1        = models.resnet34(pretrained=pretrain)
            res2        = models.resnet34(pretrained=pretrain)
            

        if '50' in model_name:
            res1        = models.resnet50(pretrained=pretrain)
            res2        = models.resnet50(pretrained=pretrain)
            
        self.ftrs   = res1.fc.in_features
        self.res1   = nn.Sequential(*(list(res1.children())[:-1]))
        self.res2   = nn.Sequential(*(list(res2.children())[:-1]))

        if pretrain:
            for param in self.res1.parameters():
                param.requires_grad = False 
            for param in self.res2.parameters():
                param.requires_grad = False 
                
        avgpool     = nn.Sequential(
                        nn.BatchNorm2d(num_features=self.size),
                        nn.AdaptiveAvgPool2d((1,1)))

        self.res1.avgpool = avgpool
        self.res2.avgpool = avgpool
        self.dropout      = nn.Dropout(dropout_rate)

        self.batch1 = nn.BatchNorm1d(self.ftrs//2)
        self.fc1    = nn.Linear(in_features=self.ftrs*2, out_features=self.ftrs//2)
        self.fc2    = nn.Linear(in_features=self.ftrs//2, out_features=1)

        if self.ftrs >= 1024:
            self.batch1  = nn.BatchNorm1d(self.ftrs)
            self.batch2  = nn.BatchNorm1d(self.ftrs//2)
            self.fc1     = nn.Linear(in_features=self.ftrs*2, out_features=self.ftrs)
            self.fc2     = nn.Linear(in_features=self.ftrs, out_features=self.ftrs//2)
            self.fc3     = nn.Linear(in_features=self.ftrs//2, out_features=1)

    def forward(self, mel, mfcc):
        out_mel  = self.res1(mel)
        out_mfcc = self.res2(mfcc)
        out      = torch.cat((out_mel, out_mfcc), dim=1)

        out      = out.view(-1, self.ftrs*2)
        out      = F.relu(self.fc1(out))
        out      = self.batch1(out)

        out      = self.dropout(out)
        out      = self.fc2(out)

        if self.ftrs >= 1024:
            out  = F.relu(out)
            out  = self.batch2(out)
            out  = self.dropout(out)
            out  = self.fc3 (out)
        
        return torch.sigmoid(out)

class SimpleCNN(Module):
    def __init__(self): 
        super(SimpleCNN, self).__init__()
        self.dropout    = nn.Dropout(0.2)
        self.maxpool    = nn.MaxPool2d(kernel_size=(2,2))

        self.conv1      = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2      = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3      = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)

        self.fc1        = nn.Linear(in_features=14*14*128, out_features=4096)
        self.fc2        = nn.Linear(in_features=4096, out_features=256)
        self.fc3        = nn.Linear(in_features=256, out_features=1)

        self.sigmoid    = nn.Sigmoid()

    def forward(self, input):
        out = F.relu(self.conv1(input))
        out = self.maxpool(out)
        out = F.relu(self.conv2(out))
        out = self.maxpool(out)
        out = F.relu(self.conv3(out))
        out = self.maxpool(out)

        out = out.view(-1, 14*14*128)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)

        return torch.sigmoid(out)

def initialize_model(model_name, weight=None):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    use_pretrained  = False

    if weight == None:
        use_pretrained  = True

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft        = models.resnet18(pretrained=use_pretrained)
        if use_pretrained:
            for param in model_ft.parameters():
                param.requires_grad = False 

        model_ft.avgpool = nn.Sequential(
                            nn.BatchNorm2d(num_features=512),
                            nn.AdaptiveAvgPool2d((1,1)),
                            )

        model_ft.fc      = nn.Sequential(
                            nn.Linear(512, 1),
                            nn.Sigmoid())

        print("Resnet18 initializing")

    elif model_name == "resnet34":
        """ Resnet34
        """
        model_ft        = models.resnet34(pretrained=use_pretrained)
        if use_pretrained:
            for param in model_ft.parameters():
                param.requires_grad = False 
        model_ft.avgpool = nn.Sequential(
                            nn.BatchNorm2d(num_features=512),
                            nn.AdaptiveAvgPool2d((1,1)),
                            )

        model_ft.fc      = nn.Sequential(
                            nn.Linear(512, 1),
                            nn.Sigmoid())

        print("Resnet34 initializing")

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft        = models.resnet50(pretrained=use_pretrained)
        if use_pretrained:
            for param in model_ft.parameters():
                param.requires_grad = False 
        model_ft.fc = nn.Sequential(
                        nn.Linear(2048, 1024),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(1024, 128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, 1),
                        nn.Sigmoid())

        print("Resnet50 initializing")

    elif model_name == "simplecnn":
        """Simple CNN
        """
        model_ft        = SimpleCNN()
        print("Simple RCNN initializing")

    elif model_name =="tworesnet34":
        """Two resnet neck"""
        model_ft        = TwoResnetWay(backbone_name=model_name, pretrain=use_pretrained)
        print("Two Resnet neck")

    else:
        print("Invalid model name, exiting...")

    if weight == None:
        if 'resnet' not in model_name:
            model_ft.apply(weights_init)

    else: 
        model_ft.load_state_dict(torch.load(weight))
        print('Weight being loaded')
    
    return model_ft