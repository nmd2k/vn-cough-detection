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
        

        # Read csv and random shuffle rows
        df = pd.read_csv(os.path.join(root_path, 'train', 'metadata_train_challenge.csv'))
        # df = df.sample(frac=1) 

        # Get train and test set from the data
        split_pos = int(len(df) * 0.8)
        train_csv = df.iloc[:split_pos,:]
        val_csv = df.iloc[split_pos:,:]

        if is_train:
            csv = train_csv
        else:
            csv = val_csv

        self.ids       = csv['uuid']
        self.target    = csv['assessment_result']


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        audio_id = self.ids[index]
        target = float(self.target[index])
        
        return audio_id, target
        
"""
The dataset where each item is the amplitude array of a specific audio, processed by torchaudio
"""
class RawAudioAmplitudeDataset(Dataset):
    def __init__(self,  root_path, is_train=True):
        dataset_dir = "train" if is_train else "test"

        self.is_train = is_train

        self.metadata = pd.read_csv(os.path.join(root_path, dataset_dir, "metadata.csv"))
        self.items = torch.load(os.path.join(root_path, dataset_dir, "amplitude.pt"))
        
        if is_train:
            audio_id_to_labels = {}
            for idx, audio_id in enumerate(self.metadata["file_path"]):
                audio_id_to_labels[audio_id] = (self.metadata["assessment_result"][idx])
            self.labels = [audio_id_to_labels[item[0]] for item in self.items]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if self.is_train:
            return self.items[idx], float(self.labels[idx])
        else:
            return self.items[idx]

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