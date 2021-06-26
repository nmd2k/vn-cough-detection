from logging import root
import torch
from torch.utils.data import Dataset

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
        self.transform = transform

        pass
    
    def __len__(self):
        pass

    def __getitem__(self, index):
        if not self.train:
            # testset return id and input
            pass
        
        # trainset return input and target
        return None
        