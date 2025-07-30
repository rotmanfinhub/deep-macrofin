from collections import OrderedDict
from typing import Any, Dict, List

import torch
import torch.nn as nn

from torch.nn.init import xavier_uniform_

from .activation_types import *


class DGMLayer(nn.Module):
    '''
    Implementation of a LSTM-like layer for the neural network proposed by
    J. Sirignano and K. Spiliopoulos, "DGM: A deep learning algorithm for
    solving partial differential equations", 2018.

    From https://github.com/gdetor/differential_equations_dnn/blob/main/neural_networks.py
    '''
    def __init__(self, input_size=1, output_size=1, act_func=ActivationType.ReLU):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Calculate the gain for the Xavier weight initialization
        try:
            gain = nn.init.calculate_gain(act_func)
        except:
            gain = nn.init.calculate_gain(act_func.value)

        # Initialize all the parameters
        self.Uz = nn.Parameter(xavier_uniform_(torch.ones([input_size,
                                                           output_size]),
                                               gain=gain))
        self.Ug = nn.Parameter(xavier_uniform_(torch.ones([input_size,
                                                           output_size]),
                                               gain=gain))
        self.Ur = nn.Parameter(xavier_uniform_(torch.ones([input_size,
                                                           output_size]),
                                               gain=gain))
        self.Uh = nn.Parameter(xavier_uniform_(torch.ones([input_size,
                                                           output_size]),
                                               gain=gain))

        self.Wz = nn.Parameter(xavier_uniform_(torch.ones([output_size,
                                                           output_size]),
                                               gain=gain))
        self.Wg = nn.Parameter(xavier_uniform_(torch.ones([output_size,
                                                           output_size]),
                                               gain=gain))
        self.Wr = nn.Parameter(xavier_uniform_(torch.ones([output_size,
                                                           output_size]),
                                               gain=gain))
        self.Wh = nn.Parameter(xavier_uniform_(torch.ones([output_size,
                                                           output_size]),
                                               gain=gain))

        self.bz = nn.Parameter(torch.zeros(output_size))
        self.bg = nn.Parameter(torch.zeros(output_size))
        self.br = nn.Parameter(torch.zeros(output_size))
        self.bh = nn.Parameter(torch.zeros(output_size))

        # Set the non-linear activation functions
        self.act1 = activation_function_mapping[act_func]()
        self.act2 = activation_function_mapping[act_func]()

    def forward(self, x, s):
        Z = self.act1(torch.matmul(x, self.Uz) + torch.matmul(s, self.Wz) +
                      self.bz)
        G = self.act1(torch.matmul(x, self.Ug) + torch.matmul(s, self.Wg) +
                      self.bg)
        R = self.act1(torch.matmul(x, self.Ur) + torch.matmul(s, self.Wr) +
                      self.br)

        H = self.act2(torch.matmul(x, self.Uh) + torch.matmul(s*R, self.Wh) +
                      self.bh)

        s_new = (torch.ones_like(G) - G) * H + Z * s
        return s_new


class DGM(nn.Module):
    '''
    DGM LSTM-like neural network.
    '''
    def __init__(self, configs: Dict[str, Any]):
        '''
        Configs: specifies number of layers/hidden units/activation function of the neural network.
            - device: **str**, the device to run the model on (e.g., "cpu", "cuda"), default will be chosen based on whether or not GPU is available
            - hidden_units: **List[int]**, number of units in each layer, default: [30, 30, 30, 30]. For DGM, only the first element in the list is used, but the length will be used as the number of layers.
            - output_size: **int**, number of output units, default: 1 for MLP, and last hidden unit size for KAN and MultKAN
            - activation_type: *str**, a selection from the ActivationType enum, default: ActivationType.Tanh
            - positive: **bool**, apply softplus to the output to be always positive if true, default: false (This has no effect for KAN.)
        '''

        super(DGM, self).__init__()
        self.configs = configs
        self.build_model()
    
    def build_model(self):
        activation_type = self.configs["activation_type"]
        input_size = self.configs["input_size"]
        hidden_size = self.configs["hidden_units"][0]
        output_size = self.configs["output_size"]
        positive = self.configs["positive"]

        # Input layer
        self.x_in = nn.Linear(input_size, hidden_size)

        # DGM layers
        self.dgm1 = DGMLayer(input_size, hidden_size, act_func=activation_type)
        self.layers = nn.Sequential(*[DGMLayer(input_size, hidden_size)
                                     for _ in range(len(self.configs["hidden_units"]))])

        # Output layer
        output_layers = [nn.Linear(hidden_size, output_size)]
        if positive:
            output_layers.append(nn.Softplus())
        self.x_out = nn.Sequential(*output_layers)

        # Non-linear activation function
        self.sigma = activation_function_mapping[activation_type]()

        # Initialize input and output layers
        xavier_uniform_(self.x_in.weight)
        xavier_uniform_(self.x_out[0].weight)

    def forward(self, x):
        '''
        Forward method.

        @param x Input tensor of shape (*, input_dim)

        @note The input_dim is the number of covariates (independent variables)
        and output_dim is the number of dependent variables.

        @return s Output tensor of shape (*, output_dim)
        '''
        s = self.sigma(self.x_in(x))
        for layer in self.layers:
            s = layer(x, s)
        s = self.x_out(s)
        return s