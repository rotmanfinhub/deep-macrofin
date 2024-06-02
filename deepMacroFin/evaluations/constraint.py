import re
from typing import Callable, Dict, List, Union

import torch

from .comparators import Comparator
from .formula import Formula


class Constraint:
    '''
    Given a string representation of a constraint (equality or inequality), 
    and a set of variables (state, value, prices) in a model, 
    parse the equation to a pytorch function that can be evaluated. 

    Label is used to identify the corresponding system to use 
    when the constraint is satisfied.

    If the constraint is an inequality, loss should be penalized 
    whenever the inequality is not satisfied. 
    e.g. if eq is L1 > L2, it is not satisfied when L2 - L1 >= 0, 
    so we should formulate the Loss as ReLU(L2-L1).
    '''
    def __init__(self, lhs, comparator: Comparator, rhs, label, latex_var_mapping: Dict[str, str] = {}):
        pass 

    def eval(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        '''
            compute the loss based on the inequality constraint
        '''
        pass