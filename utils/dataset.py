import os
import torch
import torchaudio
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from utils.data_tools import AudioUtil

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
            self.transform = transforms.ToTensor()
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
        
class FixedLenDataset(Dataset):
    def __init__(self, root_path, mfcc=True, is_train=True, transform=None):
        """
        Fixed length cough covid dataset
        Args:
            root_path (string): Path to dataset directory
            mfcc (bool): Return MFCC image if True, otherwise Mel spectrogram
            is_train (bool): Train dataset or test dataset
            transform (function): whether to apply the data augmentation scheme
                    mentioned in the paper. Only applied on the train split.
        """

        self.root = root_path
        self.mfcc = mfcc

        if is_train:
            self.df = pd.read_csv(os.path.join(root_path, 'metadata_train_challenge.csv'))
        else: 
            self.df = pd.read_csv(os.path.join(root_path, 'metadata_test_challenge.csv'))

        if transform == None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        aud   = torchaudio.load(self.root + self.df.loc[idx, 'file_path'])
        label = self.df.loc[idx, 'assessment_result']
        aud   = AudioUtil.pad_trunc(aud, max_ms = 10000)
        spec  = AudioUtil.get_spectrogram(aud, n_mels=64, n_fft=1024, hop_len=None)
        mfcc  = AudioUtil.get_mfcc(aud)

        # transform
        label = self.transform(label)
        mfcc  = self.transform(mfcc)
        spec  = self.transform(spec)

        if self.mfcc:
            return mfcc, label

        return spec, label