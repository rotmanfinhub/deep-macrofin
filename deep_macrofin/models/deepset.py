from collections import OrderedDict
from typing import Any, Dict, List

import torch
import torch.nn as nn

from .activation_types import *


class DeepSet(nn.Module):
    def __init__(self, configs: Dict[str, Any]):
        '''
        This implements the deepset architecture which allows for permutation equivariance

        http://papers.nips.cc/paper/6931-deep-sets.pdf

        Configs: specifies number of layers/hidden units/activation function of the neural network.
        - input_size_sym: int, number of dimensions that should impose symmetry
        - input_size_ext: int, number of extra dimensions
        - hidden_units_phi: List[int], number of hidden units in phi
        - hidden_units_rho: List[int], number of hidden units in rho
        - output_size: int, number of output dimensions
        - activation_type
        '''

        super(DeepSet, self).__init__()
        self.configs = self.check_configs(configs)
        self.build_model()

    def check_configs(self, configs: Dict[str, Any]):
        '''
        By default assume full symmetry

        This function is expected to be called after LearnableVar.check_inputs is called. 
        Other parameters should be imposed already.

        '''
        if "input_size_sym" not in configs:
            configs["input_size_sym"] = configs["input_size"]

        if "input_size_ext" not in configs:
            configs["input_size_ext"] = 0

        if "hidden_units_phi" not in configs:
            configs["hidden_units_phi"] = [30, 30]

        if "hidden_units_rho" not in configs:
            configs["hidden_units_rho"] = [30, 30]

        if configs["output_size"] < configs["input_size_sym"]:
            output_size = configs["output_size"]
            input_size_sym = configs["input_size_sym"]
            raise f"output_size={output_size} is smaller than input_size_sym={input_size_sym}. Impossible to permute."
        
        return configs
    
    def build_model(self):
        self.input_size_sym = self.configs["input_size_sym"]
        self.input_size_ext = self.configs["input_size_ext"]
        self.output_size = self.configs["output_size"]

        hidden_units_phi = self.configs["hidden_units_phi"]
        hidden_units_rho = self.configs["hidden_units_rho"]
        act_func = activation_function_mapping.get(self.configs["activation_type"], nn.Tanh)

        input_size = 1
        phi_layers = OrderedDict()
        for i in range(len(hidden_units_phi) - 1):
            phi_layers[f"linear_{i}"] = nn.Linear(input_size, hidden_units_phi[i])
            phi_layers[f"activation_{i}"] = act_func()
            input_size = hidden_units_phi[i]
        phi_layers["final_layer"] = nn.Linear(input_size, hidden_units_phi[-1])
        self.phi = nn.Sequential(phi_layers)

        # input_size = hidden_units_phi[-1] + self.input_size_ext
        input_size = 2 * hidden_units_phi[-1] + self.input_size_ext
        rho_layers = OrderedDict()
        for i in range(len(hidden_units_rho)):
            rho_layers[f"linear_{i}"] = nn.Linear(input_size, hidden_units_rho[i])
            rho_layers[f"activation_{i}"] = act_func()
            input_size = hidden_units_rho[i]
        # rho_layers["final_layer"] = nn.Linear(input_size, self.output_size)
        rho_layers["final_layer"] = nn.Linear(input_size, 1)
        if self.configs.get("positive", False):
            rho_layers["positive_act"] = nn.Softplus()
        self.rho = nn.Sequential(rho_layers)

        if self.output_size > self.input_size_sym:
            '''
            Only when output_size == input_size_sym, we can achieve global permutation equivariance.

            If output_size > input_size_sym, we can only achieve partial permutation equivariance. The rest of the dimensions are invariant, and we can use a different neural network (pooler) to compute them.
            '''
            input_size = hidden_units_phi[-1] + self.input_size_ext
            pooler_layers = OrderedDict()
            for i in range(len(hidden_units_rho)):
                pooler_layers[f"linear_{i}"] = nn.Linear(input_size, hidden_units_rho[i])
                pooler_layers[f"activation_{i}"] = act_func()
                input_size = hidden_units_rho[i]
            # output all remaining dimensions
            pooler_layers["final_layer"] = nn.Linear(input_size, self.output_size - self.input_size_sym)
            if self.configs.get("positive", False):
                pooler_layers["positive_act"] = nn.Softplus()
            self.pooler = nn.Sequential(pooler_layers)
    
    def forward(self, x: torch.Tensor):
        x_sym = x[...,:self.input_size_sym]
        x_ext = x[...,self.input_size_sym:]
        x_sym = x_sym.unsqueeze(-1) # (B, input_size_sym, 1)

        # step 1: generate a global representation of the set using encoder phi
        phi_out = self.phi(x_sym) # (B, input_size_sym, hidden)
        phi_agg = torch.mean(phi_out, dim=-2, keepdim=True) # use -2 instead of 1 to ensure compatibility with vmap operation

        # step 2: compute the permutation equivariant part using decoder rho
        if len(x.shape) == 1:
            # vmapped
            phi_agg = phi_agg.expand(self.input_size_sym, -1) # (B, input_size_sym, hidden)
            combined = torch.cat([phi_out, phi_agg], dim=-1)  # (B, input_size_sym, 2*hidden)
            if x_ext.shape[-1] > 0:
                combined = torch.cat([combined, x_ext.unsqueeze(-2).expand(self.input_size_sym, -1)], dim=-1) # (B, input_size_sym, 2*hidden+ext)
                pooler_input = torch.cat([phi_agg[0, :], x_ext], dim=-1) # (B, hidden+ext)
            else:
                pooler_input = phi_agg[0, :]  # (B, hidden+ext)
        else:
            phi_agg = phi_agg.expand(-1, self.input_size_sym, -1) # (B, input_size_sym, hidden)
            combined = torch.cat([phi_out, phi_agg], dim=-1)  # (B, input_size_sym, 2*hidden)
            if x_ext.shape[-1] > 0:
                combined = torch.cat([combined, x_ext.unsqueeze(-2).expand(-1, self.input_size_sym, -1)], dim=-1) # (B, input_size_sym, 2*hidden+ext)
                pooler_input = torch.cat([phi_agg[:, 0, :], x_ext], dim=-1) # (B, hidden+ext)
            else:
                pooler_input = phi_agg[:, 0, :]  # (B, hidden+ext)
        out = self.rho(combined).squeeze(-1) # (B, input_size_sym)

        # step 3: compute the remaining dimensions using the pooler
        if self.output_size > self.input_size_sym:
            pooler_out = self.pooler(pooler_input) # (B, output_size - input_size_sym)
            out = torch.cat([out, pooler_out], dim=-1) # (B, output_size)
        return out
