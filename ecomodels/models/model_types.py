from collections import OrderedDict
from enum import Enum

import torch.nn as nn


class ModelType(str, Enum):
    MLP="MLP"
    KAN="KAN"

class ActivationType(str, Enum):
    ReLU="relu"
    Tanh="tanh"

activation_function_mapping = {
    ActivationType.ReLU: nn.ReLU,
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