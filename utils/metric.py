import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def binary_acc(y_pred, y_true):
    round_y_pred    = torch.round(y_pred)
    total_true      = (round_y_pred == y_true).sum().float()

    return total_true

def plot_roc_auc(y_pred, y_true):
    # convert to numpy
    y_pred, y_true = y_pred.data.cpu().numpy(), y_true.data.cpu().numpy()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    n_classes = y_true.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i]  = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i]      = auc(fpr[i], tpr[i])

    pos_label = 1
    
    plt.figure()
    lw = 2
    plt.plot(fpr[pos_label], tpr[pos_label], color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[pos_label])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc



