import re
from enum import Enum
from typing import Callable, Dict, List, Union

import torch

from .formula import EvaluationMethod, Formula
from .loss_compute_methods import LOSS_REDUCTION_MAP, LossReductionMethod


class EndogEquation:
    '''
    Given a string representation of an endogenuous equation, 
    and a set of variables (state, value, prices) in a model, 
    parse the equation to a pytorch function that can be evaluated.

    This is used to define the loss functions

    The formula classes should be able to parse the equation.
    '''
    def __init__(self, eq: str, label: str, latex_var_mapping: Dict[str, str] = {}):
        '''
        Parse the equation LHS and RHS of `eq` separately,
        '''
        assert "=" in eq, f"The endogenous equation ({self.eq}) does not contain =."
        self.eq = eq.replace("==", "=")
        self.label = label
        eq_splitted = self.eq.split("=")
        self.lhs = Formula(eq_splitted[0], EvaluationMethod.Eval, latex_var_mapping)
        self.rhs = Formula(eq_splitted[1], EvaluationMethod.Eval, latex_var_mapping)

    def eval(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor], 
                loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
        '''
        evaluate LHS and RHS, compute MSE between them, return the value
        '''
        lhs_eval = self.lhs.eval(available_functions, variables)
        rhs_eval = self.rhs.eval(available_functions, variables)
        return LOSS_REDUCTION_MAP[loss_reduction](lhs_eval - rhs_eval)
    
    def eval_no_loss(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        return self.eval(available_functions, variables, loss_reduction=LossReductionMethod.NONE)
    
    def eval_with_mask(self,  available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor], 
                       mask: torch.Tensor, loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
        '''
        evaluate LHS and RHS, for batch elements selected by the mask, compute MSE between them, return the value
        mask should be set by a system only, it is to detect which loss should be triggered by the system constraint
        '''
        lhs_eval = self.lhs.eval(available_functions, variables)
        rhs_eval = self.rhs.eval(available_functions, variables)
        return LOSS_REDUCTION_MAP[loss_reduction]((lhs_eval - rhs_eval)[mask.squeeze(-1).to(torch.bool)])

    def __str__(self):
        str_repr = f"{self.label}: \n"
        str_repr += f"Raw input: {self.eq}\n" 
        str_repr += f"Parsed: {self.lhs.formula_str}={self.rhs.formula_str}"
        # str_repr += "-" * 80
        return str_repr