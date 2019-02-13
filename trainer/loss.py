import torch.nn as nn

def nll_loss(output, target):
    return nn.functional.nll_loss(output, target)

def cross_entropy_loss(output, target):
    return nn.CrossEntropyLoss()(output, target)
