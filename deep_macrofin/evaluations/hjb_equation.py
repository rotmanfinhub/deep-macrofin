import re
from enum import Enum
from typing import Callable, Dict, List, Union

import torch

from .formula import EvaluationMethod, Formula


class HJBEquation:
    '''
    Given a string representation of a Hamilton-Jacobi-Bellman equation,
    and a set of variables in a model,
    parse the equation to a pytorch function that can be evaluated.

    This is used to define the loss function, in the form of a maximization/minimization problem.

    The formula classes should be able to parse the equation
    '''
    def __init__(self, eq: str, label: str, latex_var_mapping: Dict[str, str] = {}):
        '''
        Parse the equation LHS and RHS of `eq` separately,
        '''
        self.label = label
        self.eq = eq
        self.parsed_eq = Formula(eq, EvaluationMethod.Eval, latex_var_mapping)

    def eval(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        '''
        Evaluate the function, compute MSE with 0, return the value
        '''
        eq_eval = self.parsed_eq.eval(available_functions, variables)
        return torch.mean(torch.square(eq_eval))

    def __str__(self):
        str_repr = f"{self.label}: \n"
        str_repr += f"Raw input: {self.eq};\n"
        str_repr += f"Parsed: {self.parsed_eq.formula_str}"
        return str_repr
