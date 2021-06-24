import wandb
import torch
import argparse
import time
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from model.config import *
from data.dataset import AICoughDataset
from model.model import Randomize

def parse_args():
    """
    Parse command line arguments
    """
    
    parser = argparse.ArgumentParser(description='COVID-19 Detection through Cough')
    parser.add_argument('--run', type=str, default=RUN_NAME, help='run name')
    parser.add_argument('--dataset', type=str, default=DATASET, help='dataset name for call W&B api')
    parser.add_argument('--model', type=str, default=None, help='init model')
    parser.add_argument('--epoch', type=int, default=EPOCH, help='number of epoch')
    parser.add_argument('--size', type=int, default=INPUT_SIZE, help='input size')
    parser.add_argument('--lr', type=float, default=LR, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='size of each batch')
    parser.add_argument('--num_worker', type=int, default=NUM_WORKER, help='how many subprocesses to use for data loading. (default = 0)')
    
    args = parser.parse_args()
    return args

def train(model, device, trainloader, optimizer, loss_function):
    """
    Train model on train dataset

    Args:
        model (function): deep learning model 
        device (string): run on 'cuda' or 'cpu'
        trainloader (iterator): Iterator for iterate through train dataset
        optimizer (function): optimizer function for update gradient descent
        loss_function (function): function for calucate model loss
    """
    pass

def eval(model, device, validloader, loss_function, best_acc):
    """
    Evaluate model on valid dataset

    Args:
        model (function): deep learning model 
        device (string): run on 'cuda' or 'cpu'
        trainloader (iterator): Iterator for iterate through valid dataset
        loss_function (function): function for calucate model loss
        best_acc (float): checkpoint for save current model
    """
    pass

if __name__ == '__main__':
    args = parse_args()
    
    # log run into wandb
    config = dict(
        run         = args.run,
        model       = args.model,
        size        = args.size,
        batch_size  = args.batch_size,
        num_worker  = args.num_worker,

        lr          = args.lr,
        epoch       = args.epoch,
    )

    run = wandb.init(project=PROJECT, config=config)

    # select dataset version + download it if need
    artifact     = run.use_artifact(args.dataset+DVERSION, type='RAW DATASET')
    artifact_dir = artifact.download(DATA_PATH)

    # load dataset
    train_set = AICoughDataset(root_path=DATA_PATH, is_train=True)
    valid_set = AICoughDataset(root_path=DATA_PATH, is_train=False)

    trainloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_worker)
    validloader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_worker)

    # set device to train on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Current device", torch.cuda.get_device_name(torch.cuda.current_device()))

    # define model + optimizer + criterion
    # TODO: replace dummy model, criterion, optimizer
    model = Randomize().to(device)

    criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # start training
    run.watch(models=model, criterion=criterion, log='all', log_freq=10) # call wandb to track weight and bias

    best_acc = 0 # if valid acc bigger than best_acc then save version
    epochs = args.epoch
    
    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train(model, device, trainloader, optimizer, criterion)
        t1 = time.time()
        # log train experiment
        print(f'{epoch}/epochs | Train loss: {train_loss:.3f} | Train accuracy: {train_acc:.3f} | t: {(t1-t0):.1f}s')
        
        test_loss, test_acc = eval(model, device, validloader, criterion, best_acc)
        # log eval experiment
        print(f'{epoch}/epochs | Valid loss: {train_loss:.3f} | Valid accuracy: {train_acc:.3f}')

    # TODO: log weight into wandb

    