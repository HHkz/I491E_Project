import torch
import torch.nn.functional as F


class CELoss(torch.nn.Module):
    """
    CE: Cross Entropy
    """
    def __init__(self, ):
        super(CELoss, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, preds, labels):
        ce = self.cross_entropy(preds, labels.long())
        return ce


class MAELoss(torch.nn.Module):
    """
    MAE: Mean Absolute Error
    2017 AAAI | Robust Loss Functions under Label Noise for Deep Neural Networks
    Ref: https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
    """
    def __init__(self, num_classes=2):
        super(MAELoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, preds, labels):
        pred = F.softmax(preds, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_oh = F.one_hot(labels.long(), self.num_classes).float()
        loss = 1. - torch.sum(label_oh * pred, dim=1)
        return loss.mean()


class RCELoss(torch.nn.Module):
    """
    RCELoss: Reverse Cross Entropy
    2018 NIPS | Towards Robust Detection of Adversarial Examples
    Ref: https://arxiv.org/abs/1706.00633
    """
    def __init__(self, num_classes, scale=1.0):
        super(RCELoss, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().cuda()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * rce.mean()


class NCELoss(torch.nn.Module):
    """
    NCELoss: Normalized Cross Entropy
    2020 ICML | Normalized loss functions for deep learning with noisy labels
    Ref: https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
    """
    def __init__(self, num_classes=2):
        super(NCELoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, preds, labels):
        pred = F.log_softmax(preds, dim=1)
        label_one_hot = F.one_hot(labels.long(), self.num_classes).float()
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return nce.mean()



class NFLLoss(torch.nn.Module):
    """
    NFLLoss: Normalized Focal Loss
    2020 ICML | Normalized loss functions for deep learning with noisy labels
    Ref: https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
    """
    def __init__(self, gamma=0, num_classes=2, size_average=True):
        super(NFLLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.num_classes = num_classes

    def forward(self, preds, labels):
        target = labels.view(-1, 1)
        logpt = F.log_softmax(preds, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target.long())
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss / normalizor

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class NCEandMAE(torch.nn.Module):
    """
    NCEandMAE: APL - Normalized Cross Entropy + MAE Loss
    2020 ICML | Normalized loss functions for deep learning with noisy labels
    Ref: https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
    """    
    def __init__(self, alpha, beta, num_classes=2):
        super(NCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.nce = NCELoss(num_classes=num_classes)
        self.mae = MAELoss(num_classes=num_classes)

    def forward(self, preds, labels):
        return self.alpha * self.nce(preds, labels) + self.beta * self.mae(preds, labels)


class NCEandRCE(torch.nn.Module):
    """
    NCEandRCE: APL - Normalized Cross Entropy + Reverse Cross Entropy
    2020 ICML | Normalized loss functions for deep learning with noisy labels
    Ref: https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
    """
    def __init__(self, alpha, beta, num_classes=2):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.nce = NCELoss(num_classes=num_classes)
        self.rce = RCELoss(num_classes=num_classes)

    def forward(self, preds, labels):
        return self.alpha * self.nce(preds, labels) + self.beta * self.rce(preds, labels)


class NFLandMAE(torch.nn.Module):
    """
    NFLandMAE: APL - Normalized Focal Loss + MAE Loss
    2020 ICML | Normalized loss functions for deep learning with noisy labels
    Ref: https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
    """   
    def __init__(self, alpha, beta, num_classes=2, gamma=0.5):
        super(NFLandMAE, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.nfl = NFLLoss(gamma=gamma, num_classes=num_classes)
        self.mae = MAELoss(num_classes=num_classes)

    def forward(self, preds, labels):
        return self.alpha * self.nfl(preds, labels) + self.beta * self.mae(preds, labels)


class NFLandRCE(torch.nn.Module):
    """
    NFLandRCE: APL - Normalized Focal Loss + Reverse Cross Entropy
    2020 ICML | Normalized loss functions for deep learning with noisy labels
    Ref: https://github.com/HanxunH/Active-Passive-Losses/blob/master/loss.py
    """   
    def __init__(self, alpha, beta, num_classes=2, gamma=0.5):
        super(NFLandRCE, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.nfl = NFLLoss(gamma=gamma, num_classes=num_classes)
        self.rce = RCELoss(num_classes=num_classes)

    def forward(self, preds, labels):
        return self.alpha * self.nfl(preds, labels) + self.beta * self.rce(preds, labels)