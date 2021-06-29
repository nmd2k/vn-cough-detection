import os
import torch
import random
import numpy as np
import torchaudio
from tqdm import tqdm
import pandas as pd
from model.config import *
from unsilence import Unsilence
import torchaudio.functional as F
import torchaudio.transforms as T

def validate_submission(sample_file, submission_file):
    """
    Function to validate submission file
    Args:
        sample_file (string): Path to submission file
        submission_file (string): Path to sample submission file
    """
    df_sample = pd.read_csv(sample_file)
    df_submission = pd.read_csv(submission_file)

    if len(df_sample["uuid"]) != len(df_submission["uuid"]):
        print("Invalid size")
        return

    submission_ids = {}

    for idx, uuid in enumerate(df_submission["uuid"]):
        submission_ids[uuid] = df_submission["assessment_result"][idx]
    
    for idx, uuid in enumerate(df_sample["uuid"]):
        if not uuid in submission_ids:
            print("Invalid uuid", uuid)
            return

    print("All ids are valid. Aligning to sample file")

    data = {
        "uuid": [],
        "assessment_result": []
    }

    for idx, uuid in enumerate(df_sample["uuid"]):
        data["uuid"].append(uuid)
        data["assessment_result"].append(submission_ids[uuid])

    df = pd.DataFrame.from_dict(data)
    df.to_csv('results.csv', index=False)

    print("Done. Result saved")

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    import matplotlib.pyplot as plt
    import librosa
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show()
    
class AudioUtil():
    def rechannel(aud, new_channel):
        """
        Convert the given audio to the desired number of channels
        """
        sig, sr = aud

        if (sig.shape[0] == new_channel):
            # Nothing to do
            return aud

        if (new_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])

        return ((resig, sr))

    def resample(aud, newsr):
        """
        Resample the given audio to have original sr = newsr
        """
        sig, sr = aud

        if (sr == newsr):
             # Nothing to do
            return aud

        num_channels = sig.shape[0]
            # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))

    def pad_trunc(aud, max_ms):
        """
        Pad or truncate a given audio to have length in miliseconds = max_ms
        """
        signal, sr = aud
        num_rows, sig_len = signal.shape
        max_len = int(sr/1000 * max_ms)
        if (sig_len > max_len):
            #Truncate the signal 
            signal = signal[:, :max_len]
        elif (sig_len < max_len):
            #Length of padding to add at the begin and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = int(max_len - sig_len - pad_begin_len)

            #Pad 0s
            begin_sig = torch.zeros(num_rows, pad_begin_len)
            end_sig = torch.zeros(num_rows, pad_end_len)
            signal = torch.cat([begin_sig, signal, end_sig], dim=1)
        return (signal, sr)

    def get_spectrogram(aud, n_mels=64, n_fft=1024, hop_len=None):
        signal, sr = aud
        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        get_spec =  T.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)
        spec = get_spec(signal)
        #print(spec.shape)
        return spec

    def get_mfcc(aud):
        signal, sr = aud
        get_mfcc_func = T.MFCC(sample_rate = sr)
        mfcc = get_mfcc_func(signal)
        #print(mfcc.shape)
        return mfcc 

def unsilence_dir(root_path):
    filename = [x for x in os.listdir(root_path) if 'wav' in x]
    dir_length = len(filename)

    pbar = tqdm(range(dir_length))
    for idx in pbar:
        pbar.set_description(filename[idx])
        read_file = Unsilence(os.path.join(root_path, filename[idx]))
        read_file.detect_silence()
        if not os.path.exists(os.path.join(root_path, 'audio')):
            os.mkdir(os.path.join(root_path, 'audio'))
        try:
            read_file.render_media(os.path.join(root_path, 'audio', filename[idx]))
        except Exception as e: 
            print(filename[idx], e)
            from shutil import copyfile
            copyfile(os.path.join(root_path, filename[idx]), os.path.join(root_path, 'audio', filename[idx]))

def mel_spectrogram_generator(sample_rate=SR, 
                                n_fft=N_FFT, 
                                win_length=None, 
                                hop_length=HOP_LENGTH_FFT, 
                                n_mels=N_MELS,
                                root_path=DATA_PATH):
    
    filename = os.listdir(os.path.join(root_path, 'audio'))
    pbar = tqdm(range(len(filename)))

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode='reflect',
        power=2.0,
        onesided=True,
        n_mels=n_mels,
    )

    for idx in pbar:
        pbar.set_description(filename[idx], refresh=False)
        aud   = torchaudio.load(os.path.join(root_path, 'audio', filename[idx]))
        aud   = AudioUtil.pad_trunc(aud, max_ms = DURATION)

        spec  = mel_spectrogram(aud[0])

        if not os.path.exists(os.path.join(root_path, 'mel_spectrogram')):
            os.mkdir(os.path.join(root_path, 'mel_spectrogram'))
        
        torch.save(spec, os.path.join(root_path, 'mel_spectrogram', filename[idx][:-3]+'pt'))

def mfcc_spectrogram_generator(sample_rate=SR, 
                                n_mfcc=N_MFCC,
                                n_fft=N_FFT, 
                                hop_length=HOP_LENGTH_FFT, 
                                n_mels=N_MELS,
                                root_path=DATA_PATH):
    
    filename = os.listdir(os.path.join(root_path, 'audio'))
    pbar = tqdm(range(len(filename)))

    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
        'n_fft': n_fft,
        'n_mels': n_mels,
        'hop_length': hop_length,}
    )

    for idx in pbar:
        pbar.set_description(filename[idx], refresh=False)
        aud   = torchaudio.load(os.path.join(root_path, 'audio', filename[idx]))
        aud   = AudioUtil.pad_trunc(aud, max_ms = DURATION)

        spec  = mfcc_transform(aud[0])

        if not os.path.exists(os.path.join(root_path, 'mfcc_spectrogram')):
            os.mkdir(os.path.join(root_path, 'mfcc_spectrogram'))
        
        torch.save(spec, os.path.join(root_path, 'mfcc_spectrogram', filename[idx][:-3]+'pt'))
