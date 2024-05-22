import torch
import torch.nn as nn
from .derivative_utils import *
from typing import List, Dict, Any
import os
import random
import numpy as np


class Agent(nn.Module):
    def __init__(self, name: str, state_variables: List[str], config: Dict[str, Any]):
        '''
        Initialize the Agent model.

        Parameters:
            - name (str): The name of the model.
            - state_variables (List[str]): List of state variables.
        config:
            - "output_size": int, the size of the output layer
            - "num_layers": int, the number of layers in the neural network
            - "model_type": str, type of the model (e.g., "linear", "active learning", "kan")
            - "positive": bool, default false, apply softplus to the output if true
            - "test_derivatives": bool, default false, check derivative computations during forward pass if true
            - "device": str, the device to run the model on (e.g., "cpu", "cuda")
        '''
        super(Agent, self).__init__()

        self.name = name
        self.state_variables = state_variables
        self.config = config
        self.input_size = len(state_variables)
        self.output_size = config['output_size']
        self.num_layers = config['num_layers']
        self.model_type = config['model_type']
        self.positive = config['positive']
        self.test_derivatives = config['test_derivatives']
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        layers = [nn.Linear(self.input_size, self.output_size), nn.Tanh()]
        for i in range(1, self.num_layers):
            layers.append(nn.Linear(self.output_size, self.output_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(self.output_size, self.output_size))

        self.net = nn.Sequential(*layers)
        nn.init.xavier_normal_(self.net[0].weight)

        self.derives_template = get_all_derivs(name, state_variables)
        self.__construct_all_derivatives()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the model.

        Parameters:
        X (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Output tensor after passing through the network.
        '''
        output = self.net(X)
        if self.positive:
            output = nn.functional.softplus(output)
        return output

    def compute_derivative(self, derivative_seq: List[str], x: torch.Tensor) -> torch.Tensor:
        '''
        Compute derivatives with respect to state variables.

        Parameters:
        derivative_seq (List[str]): Sequence of derivatives to compute.
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Computed derivatives tensor.
        '''
        x = x.clone().requires_grad_()
        y = self.forward(x)
        deriv_func = self.derivatives[derivative_seq[0]]
        for d in derivative_seq[1:]:
            deriv_func = self.derivatives[d]
        return deriv_func(x)

    def __get_derivative(self, X: torch.Tensor, target_derivative: str) -> torch.Tensor:
        X = X.clone().requires_grad_()
        y = self.forward(X)
        return self.derives_template[target_derivative](y, X)

    def __construct_all_derivatives(self):
        self.derivatives = {}
        self.derivatives[self.name] = self.forward
        for deriv_name in self.derives_template:
            self.derivatives[deriv_name] = lambda x, target_deriv=deriv_name: self.__get_derivative(x, target_deriv)

    def check_input(self, config: Dict[str, Any]):
        '''
        Validate the configuration dictionary.

        Parameters:
        config (Dict[str, Any]): Configuration dictionary.
        '''
        required_keys = ["output_size", "num_layers", "model_type", "positive", "test_derivatives"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Config must contain key: {key}")

        if "device" not in config:
            config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    def save_weights(self, optimizer: torch.optim.Optimizer, model_dir: str, filename: str):
        '''
        Save the model weights and optimizer state.

        Parameters:
        optimizer (torch.optim.Optimizer): The optimizer.
        model_dir (str): Directory to save the model.
        filename (str): Filename to save the model.
        '''
        dict_to_save = {
            "model": self.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": self.config,
            "system_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.random.get_rng_state(),
        }
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(dict_to_save, f"{model_dir}/{filename}.pt")

    def load_weights(self, f: str = None, dict_to_load: Dict[str, Any] = None):
        '''
        Load the model weights and optimizer state.

        Parameters:
        f (str): Path to the saved state dictionary.
        dict_to_load (Dict[str, Any]): A dictionary loaded from the saved file.
        '''
        assert (f is not None) or (dict_to_load is not None), "One of file path or dict_to_load must not be None"
        if f is not None:
            dict_to_load = torch.load(f)
        self.load_state_dict(dict_to_load["model"])
