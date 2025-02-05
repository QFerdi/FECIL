import torch
from torch.nn import functional as F

def KD_loss(pred, soft, T):
    """
        Knowledge distillation loss used in incremental learning
    """
    pred = torch.log_softmax(pred/T, dim=1)
    with torch.no_grad():
        soft = torch.softmax(soft/T, dim=1)
    #return -1 * torch.mul(soft, pred).sum()/pred.shape[0]
    return -torch.mean(torch.sum(soft*pred, dim=1))

def BKD(pred, soft, T, per_cls_weights):
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    soft = soft*per_cls_weights
    soft = soft / soft.sum(1)[:, None]
    return -1*torch.mul(soft, pred).sum()/pred.shape[0]

def KLdiv_loss(pred, soft, T):
    """
        Knowledge distillation loss used in transfer learning
    """
    pred = torch.log_softmax(pred/T, dim=1)
    soft = torch.softmax(soft/T, dim=1)
    return F.kl_div(pred, soft, reduction="batchmean")

def CEsoft_loss(logits, targets):
        """
            o : output of model [bsz, nbC]
            T : onehot target [bsz, nbC]
            performs standard cross-entropy computation but with onehot vectors (supports soft targets)
        """
        bsz = logits.size(0)
        logprobs = F.log_softmax(logits, dim=1)
        L = (-targets * logprobs).sum()/bsz #batchmean reduction in original CrossEntropyLoss
        return L