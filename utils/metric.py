import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def binary_acc(y_pred, y_true):
    round_y_pred    = torch.round(y_pred)
    total_true      = (round_y_pred == y_true).sum().float()

    return total_true

def plot_roc_auc(y_pred, y_true, save_img=False, save_dir='./model/exp1'):
    # convert to numpy
    y_pred, y_true = y_pred.data.cpu().numpy(), y_true.data.cpu().numpy()

    # n_classes = 1 # fixed
    fpr, tpr, _  = roc_curve(y_true, y_pred)
    roc_auc   = auc(fpr, tpr)
    
    if save_img:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")

        plt.savefig(os.path.join(save_dir, 'roc_curve_summary.png'))

    return roc_auc

