import torch
import torchaudio
import argparse
import glob
import os
from multiprocessing import Pool

def load_amplitude_from_file(filepath):
    audio_id = os.path.split(filepath)[1]
    audio_tensor, sampling_rate = torchaudio.load(filepath)

    return audio_id, audio_tensor, sampling_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir")
    parser.add_argument("--output_dir")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    audio_paths = glob.glob(os.path.join(args.audio_dir,"*"))

    p = Pool(8)
    items = p.map(load_amplitude_from_file, audio_paths)

    torch.save(items, os.path.join(args.output_dir, "amplitude.pt"))