from torch.utils.data import DataLoader,Dataset
import torchaudio
import utils.Audioutils as utils
from torch.utils.data import random_split
import pandas as pd

# Path needed
train_root = "aicv115m_public_train/train_audio_files_8k/"
test_root  = "aicv115m_public_test/public_test_audio_files_8k/"
metadata_train = pd.read_csv("aicv115m_public_train/metadata_train_challenge.csv")
metadata_test = pd.read_csv("aicv115m_public_test/metadata_public_test.csv")

class FixedLenDataset(Dataset):
    def __init__(self, root, df):
        self.root = root
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        aud = torchaudio.load(self.root + self.df.loc[idx, 'file_path'])
        label = self.df.loc[idx, 'assessment_result']
        aud = utils.pad_trunc(aud, max_ms = 10000)
        spec = utils.get_spectrogram(aud, n_mels=64, n_fft=1024, hop_len=None)
        mfcc = utils.get_mfcc(aud)
        return spec, label
    
datasets = FixedLenDataset(train_root, metadata_train)
num_train = round(len(metadata_train) * 0.8)
num_valid = len(metadata_train) - num_train
train_ds, valid_ds = random_split(datasets, [num_train, num_valid])

#Create train and valid loader
train_loader = DataLoader(train_ds, batch_size = 16, shuffle = True)
valid_loader = DataLoader(valid_ds, batch_size = 16, shuffle = True)