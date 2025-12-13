import random
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.func import hessian, jacrev, vmap

from .derivative_utils import *
from .model_utils import *


class LearnableVar(nn.Module):
    def __init__(self, name: str, state_variables: List[str], config: Dict[str, Any]):
        '''
        Build a neural network representation of a learnable variable.

        Record the forward (compute) function and all derivatives w.r.t. state variables. 

        This is the base class for Agent and EndogVar.

        Inputs:
            - name (str): The name of the model.
            - state_variables (List[str]): List of state variables.

        Config: specifies number of layers/hidden units of the neural network and highest order of derivatives to take.
            - device: **str**, the device to run the model on (e.g., "cpu", "cuda"), default will be chosen based on whether or not GPU is available
            - hidden_units: **List[int]**, number of units in each layer, default: [30, 30, 30, 30]
            - output_size: **int**, number of output units, default: 1 for MLP, and last hidden unit size for KAN and MultKAN
            - layer_type: **str**, a selection from the LayerType enum, default: LayerType.MLP
            - activation_type: *str**, a selection from the ActivationType enum, default: ActivationType.Tanh
            - positive: **bool**, apply softplus to the output to be always positive if true, default: false (This has no effect for KAN.)
            - hardcode_function: a lambda function for hardcoded forwarding function, default: None
            - derivative_order: int, an additional constraint for the number of derivatives to take, so for a function with one state variable, we can still take multiple derivatives, default: number of state variables
            - batch_jac_hes: **bool**, whether to use batch jacobian or hessian for computing derivatives, default: False (When True, only name_Jac and name_Hess are included in the derivatives dictionary, and derivative_order is ignored; When False, all derivatives name_x, name_y, etc are included.)
        '''
        super(LearnableVar, self).__init__()
        self.name = name
        self.sv_subset_idx = config.get("sv_subset", list(range(len(state_variables))))
        self.state_variables = list(np.array(state_variables)[self.sv_subset_idx])
        config["input_size"] = len(self.state_variables)
        self.config = self.check_inputs(config)
        self.device = self.config["device"]
        self.build_network()

        self.derives_template = get_all_derivs(name, self.state_variables, self.config["derivative_order"])
        self.get_all_derivatives()
        self.to(self.device)

    def check_inputs(self, config: Dict[str, Any]):
        if "device" not in config:
            config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        
        if "layer_type" not in config:
            config["layer_type"] = LayerType.MLP
        
        if "activation_type" not in config:
            config["activation_type"] = ActivationType.Tanh
        
        if "positive" not in config:
            config["positive"] = False
        
        if "hardcode_function" not in config and "hidden_units" not in config:
            config["hidden_units"] = [30, 30, 30, 30]
        
        if "output_size" not in config:
            config["output_size"] = 1

        if "derivative_order" not in config:
            config["derivative_order"] = config["input_size"]

        if "batch_jac_hes" not in config:
            config["batch_jac_hes"] = False

        if config["output_size"] > 1:
            config["batch_jac_hes"] = True

        if config["layer_type"] in [LayerType.KAN, LayerType.MultKAN ]:
            config = self.check_inputs_KAN(config)
        return config
    
    def check_inputs_KAN(self, config: Dict[str, Any]):
        config["width"] = config["hidden_units"]
        config["base_fun_type"] = config["activation_type"]
        return config
    
    def build_network(self):
        if "hardcode_function" in self.config:
            self.model = self.config["hardcode_function"]
        elif self.config["layer_type"] == LayerType.MLP:
            self.model = get_MLP_layers(self.config)
        elif self.config["layer_type"] == LayerType.KAN:
            self.model = get_KAN(self.config)
        elif self.config["layer_type"] == LayerType.MultKAN:
            self.model = get_MultKAN(self.config)
        elif self.config["layer_type"] == LayerType.DeepSet:
            self.model = get_DeepSet(self.config)
        elif self.config["layer_type"] == LayerType.DGM:
            self.model = get_DGM(self.config)
        elif self.config["layer_type"] == LayerType.ResNet:
            self.model = get_ResNet(self.config)
        else:
            required_model_type = self.config["layer_type"]
            raise NotImplementedError(f"Model type: {required_model_type} is not implemented")
    
    def forward(self, X: torch.Tensor):
        X = X.to(self.device)[..., self.sv_subset_idx]
        # The if check will never be true.
        # if len(X.shape) == 1: 
        #     # always have the shape (B, num_inputs)
        #     X = X.unsqueeze(0) 
        return self.model(X)

    def compute_derivative(self, X: torch.Tensor, target_derivative: str):
        '''
        A helper function for computing derivatives, 
        useful for constructing a dictionary representation for all derivatives, 
        take derivatives w.r.t. state variables in derivative_seq. 
        e.g. if target_derivative=et, this function should compute 
        name_{et} (e first, then t), at the specific state x.
        '''
        X = X.clone().requires_grad_()
        if len(X.shape) == 1: 
            # always have the shape (B, num_inputs)
            X = X.unsqueeze(0)
        y = self.forward(X)
        return self.derives_template[target_derivative](y, X)
    
    def get_all_derivatives(self):
        '''
        Returns a dictionary of derivative functional mapping 
        e.g. if name="qa", state_variables=["e", "t"], derivative_order=2, it will return 
        {
            "qa_e": lambda x:self.compute_derivative(x, "e")
            "qa_t": lambda x:self.compute_derivative(x, "t"),
            "qa_ee": lambda x:self.compute_derivative(x, "ee"),
            "qa_tt": lambda x:self.compute_derivative(x, "tt"),
            "qa_et": lambda x:self.compute_derivative(x, "et"),
            "qa_te": lambda x:self.compute_derivative(x, "te"),
        }
        
        If "batch_jac_hes" is True, it will only include the name_Jac and name_Hess in the dictionary. 
        {
            "qa_Jac": lambda x:vmap(jacrev(self.forward))(x),
            "qa_Hess": lambda x:vmap(hessian(self.forward))(x),
        }

        Note that the last two will be the same for C^2 functions, 
        but we keep them for completeness. 
        '''
        self.derivatives = {}
        self.derivatives[self.name] = self.forward
        if self.config["batch_jac_hes"]:
            self.derivatives[self.name + "_Jac"] = lambda x: vmap(jacrev(self.forward))(x)
            self.derivatives[self.name + "_Hess"] = lambda x: vmap(hessian(self.forward))(x)
        else:
            for deriv_name in self.derives_template:
                self.derivatives[deriv_name] = lambda x, target_deriv=deriv_name: self.compute_derivative(x, target_deriv)

    def to_dict(self):
        '''
        Save all the configurations and weights to a dictionary.
        '''
        dict_to_save = {
            "name": self.name,
            "model": self.state_dict(),
            "model_config": self.config,
            "system_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.random.get_rng_state(),
        }
        return dict_to_save
    
    def from_dict(self, dict_to_load: Dict[str, Any]):
        '''
        Load all the configurations and weights from a dictionary.
        '''
        self.load_state_dict(dict_to_load["model"])

    def plot(self, target: str, domain: Dict[str, List[np.float32]]={}, ax=None):
        '''
        Inputs:
            target: name for the original function, or the associated derivatives to plot
            domain: the range of state variables to plot. 
            If state_variables=["x", "y"] domain = {"x": [0,1], "y":[-1,1]}, it will be plotted on the region [0,1]x[-1,1].
            If one of the variable is not provided in the domain, [0,1] will be taken as the default
            ax: a matplotlib.Axes object to plot on, if not provided, it will be plotted on a new figure

        This function is only supported for 1D or 2D state_variables.
        '''
        assert len(self.state_variables) <= 2, "Plot is only supported for problems with no more than 2 state variables"
        X = []
        for sv in self.state_variables:
            x_lims = domain.get(sv, [0, 1])
            X.append(np.linspace(x_lims[0], x_lims[1], 100))
        X = np.stack(X).T
        if len(self.state_variables) == 1:
            y = self.derivatives[target](torch.Tensor(X).to(self.device)).detach().cpu().numpy()
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot()
            ax.plot(X.reshape(-1), y.reshape(-1), label=target)
            ax.set_xlabel(self.state_variables[0])
            ax.set_ylabel(target)
            ax.set_title(f"{target} vs {self.state_variables[0]}")
            ax.legend()
        elif len(self.state_variables) == 2:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            X1, X2 = np.meshgrid(X[:, 0], X[:, 1])
            y = self.derivatives[target](torch.Tensor(np.array([X1.reshape(-1), X2.reshape(-1)])).T.to(self.device)).detach().cpu().numpy()
            ax.plot_surface(X1, X2, y.reshape(100, 100), label=target)
            ax.set_xlabel(self.state_variables[0])
            ax.set_ylabel(self.state_variables[1])
            ax.set_zlabel(target)
            ax.set_title(f"{target} vs {self.state_variables[0]}, {self.state_variables[1]}")
            ax.legend()

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())