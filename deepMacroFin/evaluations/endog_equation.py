import re
from enum import Enum
from typing import Callable, Dict, List, Union

import torch

from .formula import Formula


class EndogEquation:
    '''
    Given a string representation of an endogenuous equation, 
    and a set of variables (state, value, prices) in a model, 
    parse the equation to a pytorch function that can be evaluated.

    This is used to define the loss functions

    The formula classes should be able to parse the equation.
    '''
    def __init__(self, eq, label, latex_var_mapping: Dict[str, str] = {}):
        '''
            Parse the equation LHS and RHS of `eq` separately,
        '''
        pass 
        

    def eval(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        '''
            evaluate LHS and RHS, compute MSE between them, return the value
        '''
        pass