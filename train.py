import os
import wandb
import torch
import argparse
import time
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from model.config import *
from utils.dataset import AICoughDataset, Mel_Mfcc_Dataset
from utils.mlops_tools import create_exp_dir, use_data_wandb
from utils.metric import binary_acc, plot_roc_auc
from model.common import weights_init
from utils.data_tools import eval_validset
from model.model import Randomize, SimpleCNN, initialize_model

def parse_args():
    """
    Parse command line arguments
    """
    
    parser = argparse.ArgumentParser(description='COVID-19 Detection through Cough')
    parser.add_argument('--run', type=str, default=RUN_NAME, help='run name')
    parser.add_argument('--dataset', type=str, default=DATASET, help='dataset name for call W&B api')
    parser.add_argument('--model', type=str, default=MODEL, help='init model')
    parser.add_argument('--weight', type=str, default=None, help='path to pretrained weight')
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

    running_loss, total_count, total_acc, auc = 0, 0, 0, 0
    for i, (input, target) in enumerate(trainloader):
        # load data into cuda
        input, target   = input.to(device), target.unsqueeze(1).to(device, dtype=torch.float)
        
        # zero gradient
        optimizer.zero_grad()

        if 'tworesnet' in args.model:
            mel, mfcc = input[0], mfcc[1]
            
            # forward
            predict = model(mel, mfcc)

        else:
            predict = model(input)

        loss = loss_function(predict, target)
    
        # back propagation + step
        loss.backward()
        optimizer.step()

        # metric
        running_loss    += loss.item()
        total_acc       += binary_acc(predict, target)
        total_count     += predict.shape[0]
        auc             += plot_roc_auc(y_pred=predict, y_true=target, save_dir=save_dir)

    epoch_auc  = auc/len(trainloader)
    total_loss = running_loss/len(trainloader)
    accuracy   = total_acc/total_count

    return total_loss, accuracy, epoch_auc

def eval(model, device, validloader, loss_function, best_acc):
    """
    Evaluate model on valid dataset

    Args:
        model (function): deep learning model 
        device (string): run on 'cuda' or 'cpu'
        validloader (iterator): Iterator for iterate through valid dataset
        loss_function (function): function for calucate model loss
        best_acc (float): checkpoint for save current model
    """
    model.eval()
    running_loss, total_count, total_acc, auc = 0, 0, 0, 0
    with torch.no_grad():
        for i, (input, target) in enumerate(validloader):
            # load data into cuda
            input, target   = input.to(device), target.unsqueeze(1).to(device, dtype=torch.float)
            
            # zero gradient
            optimizer.zero_grad()

            if 'tworesnet' in args.model:
                mel, mfcc = input[0], mfcc[1]
                
                # forward
                predict = model(mel, mfcc)

            else:
                predict = model(input)

            loss        = loss_function(predict, target)

            # metric
            running_loss    += loss.item()
            total_acc       += binary_acc(predict, target)
            total_count     += predict.shape[0]

            auc             += plot_roc_auc(y_pred=predict, y_true=target, save_dir=save_dir)

        total_loss = running_loss/len(validloader)
        epoch_auc  = auc/len(validloader)
        accuracy   = total_acc/total_count

        # export weight
        if accuracy>best_acc:
            torch.save(model.state_dict(), os.path.join(save_dir,'weight.pth'))
        return total_loss, accuracy, epoch_auc

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

    # select dataset version + download it if need
    use_data_wandb(run, data_name=DATASET, data_ver=DVERSION, data_type=None, root_path=DATA_PATH, download=False)

    # load dataset
    if 'tworesnet' in args.model:
        train_set   = Mel_Mfcc_Dataset(root_path=DATA_PATH, is_train=True)
    else:
        train_set   = AICoughDataset(root_path=DATA_PATH, is_train=True)

    valid_size  = int(VALID_RATE*len(train_set))
    train_set, valid_set = random_split(train_set, [len(train_set)-valid_size, valid_size])

    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
    validloader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.num_worker)

    # set device to train on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Current device", torch.cuda.get_device_name(torch.cuda.current_device()))

    # create dir for save weight
    epochs = args.epoch
    save_dir = create_exp_dir()

    # define model + optimizer + criterion
    model = initialize_model(args.model, args.weight).to(device)

    criterion = nn.BCELoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4,9,14,19], gamma=0.7)

    # start training
    run.watch(models=model, criterion=criterion, log='all', log_freq=10) # call wandb to track weight and bias

    best_acc = 0 # if valid acc bigger than best_acc then save version
    
    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc, train_auc = train(model, device, trainloader, optimizer, criterion)
        t1 = time.time()
        # log train experiment
        wandb.log({'Train loss': train_loss, 'Train accuracy': train_acc}, step=epoch)
        print(f'{epoch+1}/{epochs} | Train loss: {train_loss:.3f} | Train accuracy: {train_acc:.3f} | AUC: {train_auc:.3f} | {(t1-t0):.1f}s')
        
        t0 = time.time()
        test_loss, test_acc, test_auc = eval(model, device, validloader, criterion, best_acc)
        t1 = time.time()
        # log eval experiment
        wandb.log({'Valid loss': test_loss, 'Valid accuracy': test_acc}, step=epoch)
        print(f'{epoch+1}/{epochs} | Valid loss: {test_loss:.3f} | Valid accuracy: {test_acc:.3f} | AUC: {test_auc:.3f} | {(t1-t0):.1f}s')

        wandb.log({"lr": scheduler.get_last_lr()[0]}, step=epoch)
        
        # decrease lr
        scheduler.step()
    
    # load best model
    model.load_state_dict(torch.load(os.path.join(save_dir, 'weight.pth')))
    auc, pred, target = eval_validset(model, device, args.model, valid_set, save_dir)
    wandb.run.summary['AUC'] = auc
    
    print(f'\n===========================================\nSUMMARY: Area under the ROC curve = 0.00000 {auc:.5f}')

    trained_weight = wandb.Artifact(RUN_NAME, type='weights')
    # trained_weight.add_file(os.path.join(SAVE_PATH,RUN_NAME+'.onnx'))
    trained_weight.add_file(os.path.join(save_dir,'weight.pth'))
    wandb.log_artifact(trained_weight)