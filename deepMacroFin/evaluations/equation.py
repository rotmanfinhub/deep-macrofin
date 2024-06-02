import re
from enum import Enum
from typing import Callable, Dict, List, Union

import torch

from .formula import Formula


class Equation:
    '''
    Given a string representation of new variable definition, 
    properly evaluate it with agent, endogenous variables and constants. 
    
    The formula classes should be able to parse the equation.
    '''
    def __init__(self, eq, label, latex_var_mapping: Dict[str, str] = {}):
        '''
            Parse the equation LHS and RHS of `eq` separately, 
            firstly make sure non-latex version can be parsed correctly.
        '''
        self.label = label
        self.lhs = None
        self.rhs = None
        pass 

    def eval(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        '''
            evaluate RHS, assign value to LHS, and return the value.
        '''
        pass