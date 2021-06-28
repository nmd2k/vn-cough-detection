import wandb
import torch
import argparse
import time
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence

from model.config import *
from utils.dataset import AICoughDataset, RawAudioAmplitudeDataset
from utils.mlops_tools import use_data_wandb
from utils.metric import binary_acc
from model.model import Randomize, BaseLineRNN

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
    model.train()

    running_loss, total_count, total_acc = 0, 0, 0
    for i, (inputs, sampling_rates, targets) in enumerate(trainloader):
        # load data into cuda
        inputs, targets = inputs.to(device), targets.to(device)

        # forward
        predict = model(inputs)
        # print("MODEL PREDICTION", predict)
        # print("MODEL TARGETS", targets)
        loss = loss_function(predict, targets)

        # metric
        running_loss    += loss.item()
        total_acc       += binary_acc(predict, targets)
        total_count     += predict.shape[0]

        # zero gradient + back propagation + step
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    total_loss = running_loss/len(trainloader)
    accuracy   = total_acc/total_count

    wandb.log({'Train loss': total_loss, 'Train accuracy': accuracy})
    return total_loss, accuracy

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
    model.eval()
    running_loss, total_count, total_acc = 0,0,0
    with torch.no_grad():
        for i, (inputs, sampling_rates, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # forward
            predict = model(inputs)
            loss    = loss_function(predict, targets)

            # metric
            running_loss    += loss.item()
            total_acc       += binary_acc(predict, targets)
            total_count     += predict.shape[0]

        total_loss = running_loss/len(validloader)
        accuracy   = total_acc/total_count

        # export weight
        if accuracy>best_acc:
            try:
                torch.onnx.export(model, inputs, SAVE_PATH+RUN_NAME+'.onnx')
                torch.save(model.state_dict(), SAVE_PATH+RUN_NAME+'.pth')
            except:
                print('Can export weights')

        wandb.log({'Valid loss': total_loss, 'Valid accuracy': accuracy})
        return total_loss, accuracy

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

    run = wandb.init(project=PROJECT, config=config, entity='uet-coughcovid')

    print("Artifact name", args.dataset+DVERSION)
    # select dataset version + download it if need
    use_data_wandb(run, data_name=DATASET, data_ver=DVERSION, data_type=None, root_path=DATA_PATH, download=True)

    # load dataset
    # TODO: custom dataset
    dataset   = RawAudioAmplitudeDataset(root_path=DATA_PATH, is_train=True)

    valid_set_size = int(0.2*len(dataset))

    train_set, valid_set = random_split(dataset, [valid_set_size, len(dataset)-valid_set_size])

    def custom_collate_fn(batch):
        ids = [item[0][0] for item in batch]
        tensors = pad_sequence([item[0][1].reshape(-1, 1)[::100] for item in batch], batch_first=True)
        sampling_rates = torch.tensor([item[0][2] for item in batch])

        targets = torch.tensor([item[1] for item in batch])

        return tensors, sampling_rates, targets

    trainloader = DataLoader(train_set, 
                        batch_size=args.batch_size, 
                        num_workers=args.num_worker, 
                        collate_fn=custom_collate_fn)
    validloader = DataLoader(valid_set, 
                        batch_size=args.batch_size, 
                        num_workers=args.num_worker,
                        collate_fn=custom_collate_fn)

    # set device to train on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("Current device", torch.cuda.get_device_name(torch.cuda.current_device()))

    # define model + optimizer + criterion
    model = BaseLineRNN().to(device)

    criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # import IPython
    # IPython.embed()

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

    # # TODO: log weight into wandb
    # trained_weight = wandb.Artifact(RUN_NAME, type='weights')
    # trained_weight.add_file(SAVE_PATH+RUN_NAME+'.onnx')
    # trained_weight.add_file(SAVE_PATH+RUN_NAME+'.pth')
    # wandb.log_artifact(trained_weight)