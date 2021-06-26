from logging import root
import os
import torch
import wandb
import numpy as np
from model.config import *

def log_data_wandb(run, data_name, data_type, root_path=DATA_PATH):
    artifact = wandb.Artifact(data_name, data_type)
    
    if os.path.isdir(root_path):
        artifact.add_dir(root_path)
    elif os.path.isfile(root_path):
        artifact.add_file(root_path)

    else:
        print("Can not log data dir/file into wandb, please double check root_path")

    run.log_artifact(artifact)

if __name__ == '__main__':
    # init wandb run
    run = wandb.init(PROJECT)

    log_data_wandb(run, DATASET, 'RAW DATASET')