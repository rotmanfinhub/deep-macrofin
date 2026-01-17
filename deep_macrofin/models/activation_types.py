from collections import OrderedDict
from enum import Enum

import torch
import torch.nn as nn

class ActivationType(str, Enum):
    ReLU="relu"
    SiLU="silu"
    Sigmoid="sigmoid"
    Softplus="softplus"
    Tanh="tanh"
    Wavelet="wavelet"

class Wavelet(nn.Module):
    def __init__(self):
        super(Wavelet, self).__init__() 
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x)+ self.w2 * torch.cos(x)


activation_function_mapping = {
    ActivationType.ReLU: nn.ReLU,
    ActivationType.SiLU: nn.SiLU,
    ActivationType.Sigmoid: nn.Sigmoid,
    ActivationType.Softplus: nn.Softplus,
    ActivationType.Tanh: nn.Tanh,
    ActivationType.Wavelet: Wavelet
}