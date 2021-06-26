import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import utils.Audioutils as utils


class AICoughDataset(Dataset):
    """
    AI Cough Dataset
    Args:
        root_path (string): Path to dataset directory
        is_train (bool): Train dataset or test dataset
        transform (function): whether to apply the data augmentation scheme
                mentioned in the paper. Only applied on the train split.
    """

    def __init__(self, root_path, is_train=True, transform=None):
        self.path      = root_path
        self.train     = is_train
        if transform == None:
            trans = transforms.ToTensor()
        else:
            self.transform = transform

        if is_train:
            csv = pd.read_csv(os.path.join(root_path, 'train', '*.csv'))
        else:
            csv = pd.read_csv(os.path.join(root_path, 'test', '*.csv'))

        self.ids       = csv['uuid']
        self.target    = csv['assessment_result']
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]

        if not self.train:
            # testset return id and input
            img = Image.open(os.path.join(self.path, 'test', id + '.jpg'))
            img = self.transform(img)
            return id, img
        
        # trainset return input and target
        img    = Image.open(os.path.join(self.path, 'train', id + '.jpg'))
        target = float(self.target[id])

        img    = self.transform(img)
        target = self.transform(target)

        return img, target
        