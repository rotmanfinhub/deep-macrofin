import torch
from typing import Dict, List, Callable


class Formula:
    """
    Given a string representation of a formula, and a set of variables
    (state, value, prices, etc) in a model, parse the formula to a PyTorch
    function that can be evaluated
    """

    def __init__(self, formula_str: str, variables: List[str], evaluation_method: str):
        """
        Inputs:
        formula_str: the string version of the formula
        variables: a list of variables used (may or may not be useful, decide after implementation)
        evaluation_method: Enum, select from 'eval', 'exec', 'sympy', 'ast',
                           corresponding to the four methods below. For now, only consider eval.
        """
        self.formula_str = formula_str
        self.variables = variables
        self.evaluation_method = evaluation_method

    def eval(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        """
        Evaluate the formula with existing functions and provided assignments to variables
        """
        if self.evaluation_method == 'eval':
            return self.eval_str(available_functions, variables)
        # Placeholder for other methods like 'exec', 'sympy', 'ast'
        # elif self.evaluation_method == 'exec':
        #     return self.eval_exec(available_functions, variables)
        # elif self.evaluation_method == 'sympy':
        #     return self.eval_sympy(available_functions, variables)
        # elif self.evaluation_method == 'ast':
        #     return self.eval_ast(available_functions, variables)

    def eval_str(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        """
        Evaluate the formula using the eval method
        """
        print(f"Evaluating formula: {self.formula_str}")
        print(f"Available functions before eval: {list(available_functions.keys())}")
        try:
            # Ensuring we are in the correct context
            local_context = {"__builtins__": None}
            local_context.update(available_functions)
            # Use a lambda function to correctly access variables from the provided dictionary
            func = eval(f"lambda vars: {self.formula_str}", local_context)

            if not callable(func):
                raise ValueError("The evaluated object is not callable.")
            print(f"Evaluated function: {func}")
            print(f"Variables: {variables}")
            result = func(variables)
            return result
        except Exception as e:
            print(f"Error evaluating formula: {e}")
            raise

    def eval_ast(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        """
        Future work for AST evaluation method
        """
        pass



