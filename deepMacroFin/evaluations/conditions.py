import re
from enum import Enum
from typing import Callable, Dict, List, Union

import torch

from .comparators import Comparator
from .formula import Formula


class BaseConditions:
    '''
    Define specific boundary/initial conditions for a specific agent. 
    e.g. x(0)=0 or x(0)=x(1). 
    May also be an inequality, but it is very rare.

    The difference between a constraint and a condition is:
    - a constraint can be satisfied at any state
    - a condition must be satisfied at a specific given state

    Parse to a loss function
    '''
    def __init__(self, lhs, lhs_state: Dict[str, torch.Tensor], 
                 comparator, 
                 rhs, rhs_state: Dict[str, torch.Tensor], 
                 label, latex_var_mapping: Dict[str, str] = {}):
        pass


class MinMaxConditions:
    '''
        Todo
    '''
    def __init__(self, lhs, lhs_state: Dict[str, torch.Tensor], 
                 comparator, 
                 rhs, rhs_state: Dict[str, torch.Tensor], 
                 label, latex_var_mapping: Dict[str, str] = {}):
        pass

class AgentConditions(BaseConditions):
    '''
    Define specific boundary/initial conditions for a specific agent. 
    e.g. x(0)=0 or x(0)=x(1). 
    May also be an inequality, but it is very rare.

    The difference between a constraint and a condition is:
    - a constraint can be satisfied at any state
    - a condition must be satisfied at a specific given state

    Parse to a loss function
    '''
    def __init__(self, agent_name: str, lhs, lhs_state: Dict[str, torch.Tensor], 
                 comparator, 
                 rhs, rhs_state: Dict[str, torch.Tensor], 
                 label, latex_var_mapping: Dict[str, str] = {}):
        super().__init__(lhs, lhs_state, comparator, rhs, rhs_state, label, latex_var_mapping)
        self.agent_name = agent_name

class EndogVarConditions(BaseConditions):
    '''
    Define specific boundary/initial conditions for an endogenous variable. 
    e.g. e(0)=0 or e(0)=e(1). 
    May also be an inequality, but it is very rare.

    Parse to a loss function
    '''
    def __init__(self, endog_name: str, lhs, lhs_state: Dict[str, torch.Tensor], 
                 comparator, 
                 rhs, rhs_state: Dict[str, torch.Tensor], 
                 label, latex_var_mapping: Dict[str, str] = {}):
        super().__init__(lhs, lhs_state, comparator, rhs, rhs_state, label, latex_var_mapping)
        self.endog_name = endog_name