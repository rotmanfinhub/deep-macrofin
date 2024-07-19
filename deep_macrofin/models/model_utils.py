from collections import OrderedDict
from enum import Enum

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

activation_function_mapping = {
    ActivationType.ReLU: nn.ReLU,
    ActivationType.SiLU: nn.SiLU,
    ActivationType.Sigmoid: nn.Sigmoid,
    ActivationType.Tanh: nn.Tanh
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

    return KAN(width=width,
               base_fun=base_fun,
               device=device)