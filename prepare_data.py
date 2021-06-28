import os
import wandb
import torch
import time
import torchaudio
import numpy as np
from model.config import *
from utils.data_tools import AudioUtil, unsilence_dir

def log_data_wandb(run, data_name=DATASET, data_type=None, root_path=DATA_PATH):
    if data_type == None:
        data_type = 'DATASET'

    artifact = wandb.Artifact(data_name, data_type)

    if os.path.isdir(root_path):
        artifact.add_dir(root_path)
    elif os.path.isfile(root_path):
        artifact.add_file(root_path)

    else:
        print("Can not log data dir/file into wandb, please double check root_path")

    run.log_artifact(artifact)

def use_data_wandb(run, data_name=DATASET, data_ver=DVERSION, data_type=None, root_path=DATA_PATH, download=True):
    if data_type == None:
        artifact = run.use_artifact(data_name+':'+data_ver)
    else:
        artifact = run.use_artifact(data_name+':'+data_ver, data_type)

    if download:
        artifact.download(root_path)

if __name__ == '__main__':
    # init wandb run
    run = wandb.init(project=PROJECT, entity='uet-coughcovid')
    
    use_data_wandb(run, data_name='warm-up-8k', data_type='RAW DATASET', download=False)

    

    log_data_wandb(run, data_name="raw-amplitude-warm-up-8k", data_type="RAW DATASET")
