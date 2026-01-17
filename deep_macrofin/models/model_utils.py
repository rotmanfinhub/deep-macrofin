from collections import OrderedDict
from enum import Enum

import torch
import torch.nn as nn

from .activation_types import *
from .deepset import DeepSet
from .dgm import DGM
from .kan import KAN
from .multkan import MultKAN
from .resnet import ResNet


class LearnableModelType(str, Enum):
    Agent="Agent"
    EndogVar="EndogVar"

class LayerType(str, Enum):
    MLP="MLP"
    KAN="KAN"
    MultKAN="MultKAN"
    DeepSet="DeepSet"
    DGM="DGM"
    ResNet="ResNet"

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
    
    if config.get("sigmoid", False):
        layers["final_act"] = nn.Sigmoid()
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

def get_MultKAN(config):
    base_fun = activation_function_mapping.get(config["base_fun_type"], nn.SiLU)()
    width = config["width"]
    device = config["device"]
    grid = config.get("grid", 3)
    k = config.get("k", 3)
    grid_range = config.get("grid_range", [-1, 1])

    return MultKAN(width=width,
               base_fun=base_fun,
               grid=grid,
               k=k,
               grid_range=grid_range,
               device=device,
               auto_save=False)

def get_DeepSet(config):
    return DeepSet(config)

def get_DGM(config):
    return DGM(config)

def get_ResNet(config):
    return ResNet(config)