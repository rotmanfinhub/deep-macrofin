import re
from enum import Enum
from typing import Callable, Dict, List, Union

import torch

from .formula import Formula


class HJBEquation:
    '''
    Given a string representation of a Hamilton-Jacobi-Bellman equation, 
    and a set of variables in a model, 
    parse the equation to a pytorch function that can be evaluated.

    This is used to define the loss function, in the form of a maximization/minimization problem.

    The formula classes should be able to parse the equation
    '''
    def __init__(self, eq, label, latex_var_mapping: Dict[str, str] = {}):
        pass

    def eval(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        '''
            evaluate the function, compute MSE with 0, return the value
        '''
        pass