import re
from typing import Callable, Dict, Union

import torch
from .comparators import Comparator
from .formula import Formula, EvaluationMethod


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

    def __init__(self, lhs: str, comparator: Comparator, rhs: str, label: str, latex_var_mapping: Dict[str, str] = {}):
        '''
        Parse the constraint LHS and RHS separately, and initialize the comparator.
        '''
        self.label = label
        self.lhs = Formula(lhs, EvaluationMethod.Eval, latex_var_mapping)
        self.rhs = Formula(rhs, EvaluationMethod.Eval, latex_var_mapping)
        self.comparator = comparator

    def eval(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        '''
        Compute the loss based on the inequality constraint
        '''
        lhs_eval = self.lhs.eval(available_functions, variables)
        rhs_eval = self.rhs.eval(available_functions, variables)

        if self.comparator == Comparator.EQ:
            return torch.mean(torch.square(lhs_eval - rhs_eval))
        elif self.comparator == Comparator.GT:
            return torch.mean(torch.relu(rhs_eval - lhs_eval))
        elif self.comparator == Comparator.LT:
            return torch.mean(torch.relu(lhs_eval - rhs_eval))

    def __str__(self):
        str_repr = f"{self.label}: \n"
        str_repr += f"LHS: {self.lhs.formula_str}\n"
        str_repr += f"Comparator: {self.comparator}\n"
        str_repr += f"RHS: {self.rhs.formula_str}"
        return str_repr
