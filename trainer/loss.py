import torch.nn as nn

# categorical cross entropy = softmax + nll

def nll_loss(output, target):
    return nn.NLLLoss()(output, target)

def bce_loss(output, target):
    return nn.BCELoss()(output, target)

def logsoftmax_nll_loss(output, target):
    return nn.NLLLoss()(nn.LogSoftmax()(output), target.max(1)[1].long())

def softmax_bce_loss(output, target):
    return nn.BCELoss()(nn.Softmax()(output), target)

def sigmoid_bce_loss(output, target):
    return nn.BCELoss()(nn.Sigmoid()(output), target)