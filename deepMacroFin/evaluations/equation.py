import re
from enum import Enum
from typing import Callable, Dict, List, Union

import torch
from .formula import Formula
from .endog_equation import EndogEquation


class EvaluationMethod(str, Enum):
    Eval = "eval"
    Sympy = "sympy"
    AST = "ast"


class Equation:
    '''
    Given a string representation of new variable definition,
    properly evaluate it with agent, endogenous variables, and constants.

    The formula classes should be able to parse the equation.
    '''
    def __init__(self, eq: str, label: str, latex_var_mapping: Dict[str, str] = {}):
        '''
        Parse the equation LHS and RHS of `eq` separately,
        firstly make sure non-latex version can be parsed correctly.
        '''
        assert "=" in eq, f"The equation ({eq}) does not contain '='."
        self.label = label
        self.eq = eq.replace("==", "=")  # Ensure single equals for assignment
        eq_splitted = self.eq.split("=")
        self.lhs = Formula(eq_splitted[0], EvaluationMethod.Eval, latex_var_mapping)
        self.rhs = Formula(eq_splitted[1], EvaluationMethod.Eval, latex_var_mapping)

    def eval(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        '''
        Evaluate LHS and RHS, compute MSE between them, return the value
        '''
        lhs_eval = self.lhs.eval(available_functions, variables)
        rhs_eval = self.rhs.eval(available_functions, variables)
        return torch.mean(torch.square(lhs_eval - rhs_eval))

    def __str__(self):
        str_repr = f"{self.label}: \n"
        str_repr += f"Raw input: {self.eq};\n"
        str_repr += f"Parsed: {self.lhs.formula_str}={self.rhs.formula_str}"
        return str_repr
