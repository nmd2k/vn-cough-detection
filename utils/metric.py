import torch

def binary_acc(y_pred, y_true):
    round_y_pred    = torch.round(y_true)
    total_true      = (round_y_pred == y_pred).sum().float()

    return total_true