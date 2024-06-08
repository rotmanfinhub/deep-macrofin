import re
from typing import Callable, Dict, Union

import torch
import torch.nn.functional as F

from .comparators import Comparator
from .formula import EvaluationMethod, Formula


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

    def __init__(self, 
                 lhs: str, comparator: Comparator, rhs: str, 
                 label: str, latex_var_mapping: Dict[str, str] = {}):
        '''
        Parse the constraint LHS and RHS separately, and initialize the comparator.

        if reduce is False, eval will return lhs-rhs (used for system constraints), 
        otherwise, eval will return a single loss value
        '''
        self.label = label
        self.lhs = Formula(lhs, EvaluationMethod.Eval, latex_var_mapping)
        self.rhs = Formula(rhs, EvaluationMethod.Eval, latex_var_mapping)
        self.comparator = comparator

        if comparator == Comparator.EQ:
            self.eval = self.eval_eq
        elif comparator == Comparator.LT:
            self.eval = self.eval_lt
        elif comparator == Comparator.LEQ:
            self.eval = self.eval_leq
        elif comparator == Comparator.GT:
            self.eval = self.eval_gt
        elif comparator == Comparator.GEQ:
            self.eval = self.eval_geq

    def eval_no_reduce(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        lhs_eval = self.lhs.eval(available_functions, variables)
        rhs_eval = self.rhs.eval(available_functions, variables)
        return lhs_eval - rhs_eval

    def eval_eq(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        '''
        The condition is LHS=RHS.
        Compute the MSE between LHS and RHS.
        '''
        lhs_eval = self.lhs.eval(available_functions, variables)
        rhs_eval = self.rhs.eval(available_functions, variables)
        return torch.mean(torch.square(lhs_eval - rhs_eval))

    def eval_lt(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        '''
        The condition is LHS<RHS.
        Evaluate ReLU(LHS-RHS+eps), it will only contribute to loss when LHS>RHS-eps

        reduce: whether or not compute the MSE, if False, return lhs_eval - rhs_eval
        '''
        lhs_eval = self.lhs.eval(available_functions, variables)
        rhs_eval = self.rhs.eval(available_functions, variables)
        relu_res = F.relu(lhs_eval - rhs_eval + 1e-8)
        return torch.mean(torch.square(relu_res))

    def eval_leq(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        '''
        The condition is LHS<=RHS.
        Evaluate ReLU(LHS-RHS), it will only contribute to loss when LHS>RHS

        reduce: whether or not compute the MSE, if False, return lhs_eval - rhs_eval
        '''
        lhs_eval = self.lhs.eval(available_functions, variables)
        rhs_eval = self.rhs.eval(available_functions, variables)
        relu_res = F.relu(lhs_eval - rhs_eval)
        return torch.mean(torch.square(relu_res))
    
    def eval_gt(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        '''
        The condition is LHS>RHS.
        Evaluate ReLU(RHS-LHS+eps), it will only contribute to loss when RHS>LHS-eps

        reduce: whether or not compute the MSE, if False, return lhs_eval - rhs_eval
        '''
        lhs_eval = self.lhs.eval(available_functions, variables)
        rhs_eval = self.rhs.eval(available_functions, variables)
        relu_res = F.relu(rhs_eval - lhs_eval + 1e-8)
        return torch.mean(torch.square(relu_res))

    def eval_geq(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        '''
        The condition is LHS>=RHS.
        Evaluate ReLU(RHS-LHS), it will only contribute to loss when RHS>LHS

        reduce: whether or not compute the MSE, if False, return lhs_eval - rhs_eval
        '''
        lhs_eval = self.lhs.eval(available_functions, variables)
        rhs_eval = self.rhs.eval(available_functions, variables)
        relu_res = F.relu(rhs_eval - lhs_eval)
        return torch.mean(torch.square(relu_res))

    def __str__(self):
        str_repr = f"{self.label}: "
        str_repr += self.lhs.formula_str + self.comparator + self.rhs.formula_str
        return str_repr
