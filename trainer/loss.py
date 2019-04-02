import torch.nn as nn

# categorical cross entropy = softmax + nll

# exp 1 : softmax in network vs loss function

def nll_loss(output, target):
    return nn.functional.nll_loss(output, target)

def nll_softmax_loss(output, target):
    return nn.NLLLoss()(nn.Softmax()(output), target.long())

# exp 2 : sigmoid vs softmax

def bce_sigmoid_loss(output, target):
    return nn.BCELoss()(nn.Sigmoid()(output), target)

def bce_softmax_loss(output, target):
    return nn.BCELoss()(nn.Softmax()(output), target)

def bce_loss(output, target):
    return nn.BCELoss()(output, target)
