import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

# --------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------- #

class STEFunction(autograd.Function):
    """ Straight Through Estimator hard function for Binary Quantization"""
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

# --------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------- #

class StraightThroughEstimator(nn.Module):
    """ Straight Through Estimator implementation for Binary Quantization"""
    
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()
    def forward(self, x):
        x = STEFunction.apply(x)
        return x
    
# --------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------- #