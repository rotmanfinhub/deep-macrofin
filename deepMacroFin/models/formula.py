import torch
from typing import Dict, List, Callable


class Formula:
    def __init__(self, formula_str: str, variables: List[str], evaluation_method: str):
        self.formula_str = formula_str
        self.variables = variables
        self.evaluation_method = evaluation_method

    def eval(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        if self.evaluation_method == 'eval':
            return self.eval_str(available_functions, variables)

    def eval_str(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        try:
            # Create a local context with available functions and variables
            local_context = {"__builtins__": None}
            local_context.update(available_functions)
            local_context.update(variables)

            # Directly evaluate the formula string in the context of available functions and variables
            result = eval(self.formula_str, {"__builtins__": None}, local_context)
            return result
        except Exception as e:
            print(f"Error evaluating formula: {self.formula_str}, error: {e}")
            raise
