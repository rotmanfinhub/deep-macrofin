from collections import OrderedDict
from enum import Enum

import torch
import torch.nn as nn

from .kan import KAN


class LearnableModelType(str, Enum):
    Agent="Agent"
    EndogVar="EndogVar"

class LayerType(str, Enum):
    MLP="MLP"
    KAN="KAN"

class ActivationType(str, Enum):
    ReLU="relu"
    SiLU="silu"
    Sigmoid="sigmoid"
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
    ActivationType.Tanh: nn.Tanh,
    ActivationType.Wavelet: Wavelet
}

def get_MLP_layers(config):
    act_func = activation_function_mapping.get(config["activation_type"], nn.Tanh)
    input_size = config["input_size"]
    hidden_sizes = config["hidden_units"]
    output_size = config["output_size"]
    positive = config["positive"]

    layers = OrderedDict()

    for i in range(len(hidden_sizes)):
        layers[f"linear_{i}"] = nn.Linear(input_size, hidden_sizes[i])
        layers[f"activation_{i}"] = act_func()
        input_size = hidden_sizes[i]
    layers["final_layer"] = nn.Linear(input_size, output_size)
    
    if positive:
        layers["positive_act"] = nn.Softplus()
    
    return nn.Sequential(layers)

def get_KAN(config):
    base_fun = activation_function_mapping.get(config["base_fun_type"], nn.SiLU)()
    width = config["width"]
    device = config["device"]
    grid = config.get("grid", 3)
    k = config.get("k", 3)
    grid_range = config.get("grid_range", [-1, 1])

    return KAN(width=width,
               base_fun=base_fun,
               grid=grid,
               k=k,
               grid_range=grid_range,
               device=device)