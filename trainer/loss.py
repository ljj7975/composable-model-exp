import torch.nn as nn

def nll_loss(output, target):
    return nn.functional.nll_loss(output, target)

def cross_entropy_loss(output, target):
    return nn.CrossEntropyLoss()(output, target)

def bce_logits_loss(output, target):
    return nn.BCEWithLogitsLoss()(output, target)

def ce_sigmoid_loss(output, target):
    return nn.CrossEntropyLoss()(nn.Sigmoid()(output), target.long())

def ce_log_sigmoid_loss(output, target):
    return nn.CrossEntropyLoss()(nn.LogSigmoid()(output), target.long())

def nll_softmax_loss(output, target):
    return nn.NLLLoss()(nn.Softmax()(output), target.long())

def nll_log_softmax_loss(output, target):
    return nn.NLLLoss()(nn.LogSoftmax()(output), target.long())

def nll_sigmoid_loss(output, target):
    return nn.NLLLoss()(nn.Sigmoid()(output), target.long())

def kl_div_loss(output, target):
    return nn.KLDivLoss()(nn.Softmax()(output), target)
