import torch
from typing import Dict, Callable

def get_derivs_1order(y, x, idx):
    """ Returns the first order derivatives,
        Automatic differentiation used
    """
    dy_dx = torch.autograd.grad(y, x, 
                                create_graph=True, 
                                retain_graph=True, 
                                grad_outputs=torch.ones_like(y))[0][:, idx:idx+1]
    return dy_dx ## Return 'automatic' gradient.

def get_all_derivs(target_var_name="f", all_vars = ["x", "y", "z"]) -> Dict[str, Callable]:
    level_derivatives = {i: {} for i in range(1, len(all_vars) + 1)}
    # first order
    for i, var in enumerate(all_vars):
        # note that we must do idx=i as an additional variable, 
        # otherwise, python always capture the "variable", not its "value" at the time of creation.
        # https://stackoverflow.com/questions/33983980/lambda-in-for-loop-only-takes-last-value
        level_derivatives[1][f"{target_var_name}_{var}"] = lambda output, input, idx=i: get_derivs_1order(output, input, idx)

    # recursively define higher order derivatives
    for derivative_order in range(2, len(all_vars) + 1):
        prev_derivatives = level_derivatives[derivative_order - 1]
        for prev_str, prev_val in prev_derivatives.items():
            for i, var in enumerate(all_vars):
                # same here, we must pass prev_fun as an additional variable
                # otherwise, we will get recursive overflow
                new_val = lambda output, input, idx=i, prev_fun=prev_val: get_derivs_1order(prev_fun(output, input), input, idx)
                level_derivatives[derivative_order][f"{prev_str}{var}"] = new_val

    all_derivatives = {}
    for level_derivative in level_derivatives.values():
        all_derivatives.update(level_derivative)
    
    return all_derivatives


# TODO: Later phase, test the following
# import numpy as np
# import ast
# import operator as op

# # Assuming f1 and f2 are your lambda functions
# f1 = lambda x: x**2
# f2 = lambda x: x+1
# log = np.log  # numpy's log function

# # Create a dictionary mapping function names to functions
# func_dict = {'f1': f1, 'f2': f2, 'log': log}

# # Supported operators
# operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
#              ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}

# def eval_expr(expr):
#     return eval_(ast.parse(expr, mode='eval').body)

# def eval_(node):
#     if isinstance(node, ast.Num):  # <number>
#         return node.n
#     elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
#         return operatorstype(node.op), eval_(node.right))
#     elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
#         return operatorstype(node.op))
#     elif isinstance(node, ast.Name):
#         return func_dict[node.id]
#     else:
#         raise TypeError(node)

# # Your input string
# s = "f1+f2+log"

# # Now you can evaluate the sum of the functions at a specific x
# x = 3  # replace with your actual x
# result = eval_expr(s)

# print(result)  # Output: f1(x) + f2(x) + log(x)