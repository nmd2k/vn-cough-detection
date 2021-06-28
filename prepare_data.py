import os
import wandb
import torch
import time
import torchaudio
import numpy as np
from model.config import *
from torchvision import transforms
from utils.mlops_tools import *
from utils.data_tools import AudioUtil, plot_spectrogram, unsilence_dir, mel_spectrogram_generator, mfcc_spectrogram_generator

if __name__ == '__main__':
    config = dict(
        sample_rate  = SR,
        n_fft        = N_FFT,
        n_mfcc       = N_MFCC,
        hop_length   = HOP_LENGTH_FFT,
        n_mels       = N_MELS,
        duration     = DURATION,)

    # init wandb run
    run = wandb.init(project=PROJECT, entity='uet-coughcovid', config=config)

    use_data_wandb(run, data_name='unsilence-warm-up-8k', data_type='UNSILENCE DATASET', download=False)

    mfcc_spectrogram_generator(root_path=os.path.join(DATA_PATH, 'train'))
    mfcc_spectrogram_generator(root_path=os.path.join(DATA_PATH, 'test'))

    log_data_wandb(run, data_name='mfcc-warm-up-8k', data_type='SPECTROGRAM DATASET')
