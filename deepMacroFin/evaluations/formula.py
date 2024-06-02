from enum import Enum
from typing import Callable, Dict, List, Union

import torch


class EvaluationMethod(str, Enum):
    Eval = "eval"
    Sympy = "sympy"
    AST = "ast"


def latex_parsing(formula_str: str, latex_var_mapping: Dict[str, str] = {}):
    pass


class Formula:
    '''
        Given a string representation of a formula, and a set of variables 
        (state, value, prices, etc) in a model, parse the formula to a pytorch 
        function that can be evaluated
    '''
    
    def __init__(self, formula_str: str, evaluation_method: Union[EvaluationMethod, str], latex_var_mapping: Dict[str, str] = {}):
        '''
            Inputs:
                formula_str: the string version of the formula
                evaluation_method: Enum, select from `eval`, `sympy`, `ast`, 
                corresponding to the four methods below. For now, only consider eval.
        '''
        self.formula_str = formula_str
        self.evaluation_method = evaluation_method

        if self.evaluation_method == EvaluationMethod.Eval:
            self.eval = self.eval_str
        else:
            raise NotImplementedError(f"{evaluation_method} is not implemented")

    def eval_str(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        '''
            Evaluate the formula with existing functions and provided assignments to variables
            This evaluates the function by simple string parsing
        '''
        # Create a local context with available functions and variables
        local_context = {"__builtins__": None}
        local_context.update(available_functions)
        local_context.update(variables)

        # Directly evaluate the formula string in the context of available functions and variables
        result = eval(self.formula_str, {"__builtins__": None}, local_context)
        return result

    def try_eval(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        '''
            This is for initial checking that the formula is setup correctly
        '''
        try:
            # Create a local context with available functions and variables
            return self.eval(available_functions, variables)
        except Exception as e:
            print(f"Error evaluating formula: {self.formula_str}, error: {e}")
            raise e