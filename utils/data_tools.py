import os
import torch
import random
import torchaudio
from tqdm import tqdm
from unsilence import Unsilence
import torchaudio.functional as F
import torchaudio.transforms as T

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
        signal,sr = aud
        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        get_spec =  T.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)
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
        read_file.render_media(os.path.join(root_path, 'unsilenced', filename[idx]))