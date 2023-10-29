import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, pos_weight=1):
        super(FocalLoss, self).__init__()
        self.gamma = torch.tensor(gamma, dtype=torch.float32)
        self.eps = 1e-6
        self.pos_weight = torch.tensor(pos_weight)

    def forward(self, probs, target):
        # Predicted probabilities for the negative class
        q = 1 - probs
        p = probs
        # For numerical stability (so we don't inadvertently take the log of 0)
        p = p.clamp(self.eps, 1.0 - self.eps)
        q = q.clamp(self.eps, 1.0 - self.eps)
        self.gamma = self.gamma.to(target.device)
        self.pos_weight = self.gamma.to(target.device)
        # Loss for the positive examples
        pos_loss = -(q**self.gamma) * torch.log(p)
        if self.pos_weight is not None:
            pos_loss *= self.pos_weight

        # Loss for the negative examples
        neg_loss = -(p**self.gamma) * torch.log(q)

        loss = target * pos_loss + (1 - target) * neg_loss

        return loss.sum()
    
    
def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1