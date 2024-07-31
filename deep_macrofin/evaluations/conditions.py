import re
from enum import Enum
from typing import Callable, Dict, List, Union

import torch
import torch.nn.functional as F

from .comparators import Comparator
from .formula import EvaluationMethod, Formula
from .loss_compute_methods import LOSS_REDUCTION_MAP, LossReductionMethod


class BaseConditions:
    '''
    Define specific boundary/initial conditions for a specific agent. 
    e.g. x(0)=0 or x(0)=x(1) (Periodic conditions). 
    May also be an inequality, but it is very rare.

    The difference between a constraint and a condition is:
    - a constraint must be satisfied at any state
    - a condition is satisfied at a specific given state

    Parse to a loss function
    '''
    def __init__(self, 
                 lhs: str, lhs_state: Dict[str, torch.Tensor], 
                 comparator: Comparator, 
                 rhs: str, rhs_state: Dict[str, torch.Tensor], 
                 label: str, latex_var_mapping: Dict[str, str] = {}):
        '''
        Inputs:
        - lhs: the string expression for lhs formula, latex expression not supported, should be functions of specific format agent_name(SV), endog_name(SV), or simply a constant value
        - lhs_state: the specific value of SV to evaluate lhs at for the agent/endogenous variable
        - comparator: 
        - rhs: the string expression for lhs formula, latex expression not supported, should be functions of specific format agent_name(SV), endog_name(SV), or simply a constant value
        - rhs_state: the specific value of SV to evaluate rhs at for the agent/endogenous variable, if rhs is a constant, this can be an empty dictionary
        - label: label for the condition
        - latex_var_mapping: Not implemented. only used if the formula_str is in latex form, the keys should be the latex expression, and the values should be the corresponding python variable name. 

        E.g. if agent_name is "f", lhs="f(SV)", and lhs_state={"SV": torch.zeros((1,1))}, comparator="=" rhs="1", the condition is f(0)=1;
        if agent_name is "f", lhs="f(SV)", and lhs_state={"SV": torch.zeros((1,1))}, comparator="=" rhs="f(SV)", rhs_state={"SV": torch.ones((1,1))}, the condition is f(0)=f(1)
        '''
        if "$" in lhs or "$" in rhs:
            raise NotImplementedError("Latex expression not yet supported for conditions")
        self.label = label
        
        self.lhs = Formula(lhs, EvaluationMethod.Eval)
        self.rhs = Formula(rhs, EvaluationMethod.Eval)
        self.lhs_state = lhs_state
        self.rhs_state = rhs_state

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
    
    def eval_eq(self, available_functions: Dict[str, Callable], 
                loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
        '''
        The condition is LHS=RHS.
        Compute the MSE between LHS and RHS
        '''
        lhs_eval = self.lhs.eval(available_functions, self.lhs_state)
        rhs_eval = self.rhs.eval(available_functions, self.rhs_state)
        return LOSS_REDUCTION_MAP[loss_reduction](lhs_eval - rhs_eval)

    def eval_lt(self, available_functions: Dict[str, Callable], 
                loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
        '''
        The condition is LHS<RHS.
        Evaluate ReLU(LHS-RHS+eps), it will only contribute to loss when LHS>RHS-eps
        '''
        lhs_eval = self.lhs.eval(available_functions, self.lhs_state)
        rhs_eval = self.rhs.eval(available_functions, self.rhs_state)
        relu_res = F.relu(lhs_eval - rhs_eval + 1e-8)
        return LOSS_REDUCTION_MAP[loss_reduction](relu_res)

    def eval_leq(self, available_functions: Dict[str, Callable], 
                loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
        '''
        The condition is LHS<=RHS.
        Evaluate ReLU(LHS-RHS), it will only contribute to loss when LHS>RHS
        '''
        lhs_eval = self.lhs.eval(available_functions, self.lhs_state)
        rhs_eval = self.rhs.eval(available_functions, self.rhs_state)
        relu_res = F.relu(lhs_eval - rhs_eval)
        return LOSS_REDUCTION_MAP[loss_reduction](relu_res)
    
    def eval_gt(self, available_functions: Dict[str, Callable], 
                loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
        '''
        The condition is LHS>RHS.
        Evaluate ReLU(RHS-LHS+eps), it will only contribute to loss when RHS>LHS-eps
        '''
        lhs_eval = self.lhs.eval(available_functions, self.lhs_state)
        rhs_eval = self.rhs.eval(available_functions, self.rhs_state)
        relu_res = F.relu(rhs_eval - lhs_eval + 1e-8)
        return LOSS_REDUCTION_MAP[loss_reduction](relu_res)

    def eval_geq(self, available_functions: Dict[str, Callable], 
                loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
        '''
        The condition is LHS>=RHS.
        Evaluate ReLU(RHS-LHS), it will only contribute to loss when RHS>LHS
        '''
        lhs_eval = self.lhs.eval(available_functions, self.lhs_state)
        rhs_eval = self.rhs.eval(available_functions, self.rhs_state)
        relu_res = F.relu(rhs_eval - lhs_eval)
        return LOSS_REDUCTION_MAP[loss_reduction](relu_res)
    
    def __str__(self):
        cond_str = f"{self.label}: "
        cond_str += self.lhs.formula_str + self.comparator + self.rhs.formula_str
        lhs_state_value_str = ""
        for k, v in self.lhs_state.items():
            if isinstance(v, torch.Tensor):
                lhs_state_value_str += k + "=" + str(v.tolist())
            else:
                lhs_state_value_str += k + "=" + str(v)
        if len(lhs_state_value_str) > 0:
            cond_str += " with LHS evaluated at " + lhs_state_value_str

        rhs_state_value_str = ""
        for k, v in self.rhs_state.items():
            if isinstance(v, torch.Tensor):
                rhs_state_value_str += k + "=" + str(v.tolist())
            else:
                rhs_state_value_str += k + "=" + str(v)
        if len(rhs_state_value_str) > 0:
            cond_str += " and RHS evaluated at " + rhs_state_value_str
        return cond_str


class MinMaxConditions:

    '''
    Todo
    '''
    def __init__(self, 
                 lhs: str, lhs_state: Dict[str, torch.Tensor], 
                 comparator: Comparator, 
                 rhs: str, rhs_state: Dict[str, torch.Tensor], 
                 label: str, latex_var_mapping: Dict[str, str] = {}):
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
    def __init__(self, agent_name: str, 
                 lhs: str, lhs_state: Dict[str, torch.Tensor], 
                 comparator: Comparator, 
                 rhs: str, rhs_state: Dict[str, torch.Tensor], 
                 label: str, latex_var_mapping: Dict[str, str] = {}):
        super().__init__(lhs, lhs_state, comparator, rhs, rhs_state, label, latex_var_mapping)
        self.agent_name = agent_name

class EndogVarConditions(BaseConditions):
    '''
    Define specific boundary/initial conditions for an endogenous variable. 
    e.g. e(0)=0 or e(0)=e(1). 
    May also be an inequality, but it is very rare.

    Parse to a loss function
    '''
    def __init__(self, endog_name: str, 
                 lhs: str, lhs_state: Dict[str, torch.Tensor], 
                 comparator: Comparator, 
                 rhs: str, rhs_state: Dict[str, torch.Tensor], 
                 label: str, latex_var_mapping: Dict[str, str] = {}):
        super().__init__(lhs, lhs_state, comparator, rhs, rhs_state, label, latex_var_mapping)
        self.endog_name = endog_name