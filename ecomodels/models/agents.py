import torch
import torch.nn as nn
from .derivative_utils import *

class Net1(nn.Module):
    def __init__(self, output_size, nn_width, nn_num_layers, input_names=["x"], model_name="f", positive=False):
        super(Net1, self).__init__()

        self.input_names = input_names
        self.input_size = len(input_names)

        layers = [nn.Linear(self.input_size, nn_width), nn.Tanh()]
        for i in range(1, nn_num_layers):
            layers.append(nn.Linear(nn_width, nn_width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(nn_width, output_size))
        self.positive = positive
        self.net = nn.Sequential(*layers)
        nn.init.xavier_normal_(self.net[0].weight)  # Initialize the first linear layer weights

        self.model_name = model_name
        self.derives_template = get_all_derivs(model_name, input_names)
        self.__construct_all_derivatives()

    def forward(self, X: torch.Tensor):
        output = self.net(X)
        if self.positive: output = nn.functional.softplus(output)  # Apply softplus to the output
        return output
    
    def __get_derivative(self, X: torch.Tensor, target_derivative: str):
        X = X.clone().requires_grad_()
        y = self.forward(X)
        return self.derives_template[target_derivative](y, X)

    
    def __construct_all_derivatives(self):
        self.derivatives = {}
        self.derivatives[self.model_name] = self.forward
        for deriv_name in self.derives_template:
            self.derivatives[deriv_name] = lambda x, target_deriv=deriv_name: self.__get_derivative(x, target_deriv) 