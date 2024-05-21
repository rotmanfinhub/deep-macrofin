from typing import Any, Dict, List

import torch
import torch.nn as nn

from .derivative_utils import *
from .model_types import *


class EndogVar(nn.Module):
        def __init__(self, name, state_variables: List[str], config: Dict[str, Any]):
            '''
            build a neural network representation of agent state. 
            Record the forward (compute) function and all derivatives w.r.t. state variables. 
            Inputs:
            - name: the name of the endogenous variable, will be used in the derivatives
            - state_variables: list of names of all state variables, will be used in the derivatives

            Config: specifies number of layers/hidden units of the neural network.
            - device: the device for the model, cpu or cuda, default will be chosen based on whether or not GPU is available
            - hidden_units: number of units in each layer, list of integers
            - layer_type: a selection from the ModelType enum, default: ModelType.MLP
            - activation_type: a selection from the ActivationType enum, default: ActivationType.Tanh
            - positive: whether or not constraint the output to be always positive, default: false
            - test_derivatives: whether or not use some hard coded function to test the derivatives instead of neural network forward, default: false
            - hardcode_function: a lambda function for hardcoded forwarding function.
            '''
            super(EndogVar, self).__init__()
            self.name = name
            self.state_variables = state_variables
            self.config = self.check_inputs(config)
            self.config["input_size"] = len(self.state_variables)
            config["output_size"] = 1
            self.device = self.config["device"]
            self.build_network()

            self.derives_template = get_all_derivs(name, self.state_variables)
            self.get_all_derivatives()
            self.to(self.device)

        def check_inputs(self, config):
            if "device" not in config:
                config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            
            if "layer_type" not in config:
                config["layer_type"] = ModelType.MLP
            
            if "activation_type" not in config:
                config["activation_type"] = ActivationType.Tanh
            
            if "positive" not in config:
                config["positive"] = False
            
            if "test_derivatives" not in config:
                config["test_derivatives"] = False
            
            if not config["test_derivatives"]:
                for key in ["hidden_units"]:
                    assert key in config, f"Missing required configuration: {key}"
            
            if config["test_derivatives"] and "hardcode_function" not in config:
                # return the identity function
                config["hardcode_function"] = lambda x: x

            return config
        
        def build_network(self):
            if self.config["test_derivatives"]:
                self.model = self.config["hardcode_function"]
            elif self.config["layer_type"] == ModelType.MLP:
                self.model = get_MLP_layers(self.config)
            else:
                required_model_type = self.config["layer_type"]
                raise NotImplementedError(f"Model type: {required_model_type} is not implemented")
        
        def forward(self, X: torch.Tensor):
            X = X.to(self.device)
            if len(X.shape) == 1: 
                # always have the shape (B, num_inputs)
                X = X.unsqueeze(0) 
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
            e.g. if name="qa", state_variables=["e", "t"], it will return 
            {
                "qa_e": lambda x:self.compute_derivative(x, "e")
                "qa_t": lambda x:self.compute_derivative(x, "t"),
                "qa_ee": lambda x:self.compute_derivative(x, "ee"),
                "qa_tt": lambda x:self.compute_derivative(x, "tt"),
                "qa_et": lambda x:self.compute_derivative(x, "et"),
                "qa_te": lambda x:self.compute_derivative(x, "te"),
            }

            Note that the last two will be the same for C^2 functions, 
            but we keep them for completeness. 
            '''
            self.derivatives = {}
            self.derivatives[self.name] = self.forward
            for deriv_name in self.derives_template:
                self.derivatives[deriv_name] = lambda x, target_deriv=deriv_name: self.compute_derivative(x, target_deriv) 

        def to_dict(self):
            '''
            Save all the configurations and weights to a dictionary.
            '''
            dict_to_save = {
                "model": self.state_dict(),
                "model_config": self.config,
            }
            return dict_to_save
        
        def from_dict(self, dict_to_load: Dict[str, Any]):
            '''
            Load all the configurations and weights from a dictionary.
            '''
            self.load_state_dict(dict_to_load["model"])

