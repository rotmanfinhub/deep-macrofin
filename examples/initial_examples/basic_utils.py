import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Callable

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seeds(0)
torch.set_default_dtype(torch.float32)

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

class Net1(nn.Module):
    def __init__(self, output_size, nn_width, nn_num_layers, input_names=["x"], model_name="f", positive=False):
        super(Net1, self).__init__()

        self.input_names = input_names
        self.input_size = len(input_names)

        layers = [nn.Linear(self.input_size, nn_width), nn.Tanh()]
        for i in range(1, nn_num_layers):
            layers.append(nn.Linear(nn_width, nn_width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(nn_width, output_size))
        self.positive = positive
        self.net = nn.Sequential(*layers)
        nn.init.xavier_normal_(self.net[0].weight)  # Initialize the first linear layer weights

        self.model_name = model_name
        self.derives_template = get_all_derivs(model_name, input_names)
        self.__construct_all_derivatives()

    def forward(self, X: torch.Tensor):
        output = self.net(X)
        if self.positive: output = nn.functional.softplus(output)  # Apply softplus to the output
        return output
    
    def __get_derivative(self, X: torch.Tensor, target_derivative: str):
        X = X.clone().requires_grad_()
        y = self.forward(X)
        return self.derives_template[target_derivative](y, X)

    
    def __construct_all_derivatives(self):
        self.derivatives = {}
        self.derivatives[self.model_name] = self.forward
        for deriv_name in self.derives_template:
            self.derivatives[deriv_name] = lambda x, target_deriv=deriv_name: self.__get_derivative(x, target_deriv) 

class HardCodedFunction(nn.Module):
    def __init__(self, output_size, nn_width, nn_num_layers, input_names=["x"], model_name="f", positive=False):
        super(HardCodedFunction, self).__init__()

        self.model_name = model_name
        self.derives_template = get_all_derivs(model_name, input_names)
        self.__construct_all_derivatives()

    def forward(self, X: torch.Tensor):
        return X[:, 0:1] * X[:, 1:2]
    
    def __get_derivative(self, X: torch.Tensor, target_derivative: str):
        X = X.clone().requires_grad_()
        y = self.forward(X)
        return self.derives_template[target_derivative](y, X)

    
    def __construct_all_derivatives(self):
        self.derivatives = {}
        self.derivatives[self.model_name] = self.forward
        for deriv_name in self.derives_template:
            self.derivatives[deriv_name] = lambda x, target_deriv=deriv_name: self.__get_derivative(x, target_deriv) 

def string_to_function(variables, formula):
    global LOCAL_DICT
    def func(var_dict):
        global LOCAL_DICT
        LOCAL_DICT.update({var: var_dict[var] for var in variables})
        LOCAL_DICT.update(torch.__dict__)  # add torch functions to the local namespace
        return eval(formula, {"__builtins__": None}, LOCAL_DICT)
    return func

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


if __name__ == "__main__":
    global LOCAL_DICT
    x_np = np.random.uniform(low=[-2., -2.], 
                         high=[2., 2.], 
                         size=(2, 2))
    X = torch.Tensor(x_np)
    net = HardCodedFunction(2, 30, 4, ["x", "y"], "f")
    LOCAL_DICT = {}
    LOCAL_DICT.update(torch.__dict__)
    LOCAL_DICT.update(net.derivatives)
    variables = ["X"]
    formula = 'f_x(X)'  # using generic function names
    f = string_to_function(variables, formula)
    print(f({"X": X}))
    print(X[:, 1:2])

    formula = 'f_x(X)+f_y(X)'  # using generic function names
    f = string_to_function(variables, formula)
    print(f({"X": X}))
    print(X[:, 1:2] + X[:, 0:1])

    formula = 'f_x(X)+f_y(X) + f_xy(X)'  # using generic function names
    f = string_to_function(variables, formula)
    print(f({"X": X}))
    print(X[:, 1:2] + X[:, 0:1] + 1)

    net = HardCodedFunction(2, 30, 4, ["x1", "x2"], "f")
    LOCAL_DICT = {}
    LOCAL_DICT.update(torch.__dict__)
    LOCAL_DICT.update(net.derivatives)
    variables = ["X"]
    formula = 'f_x1(X)'  # using generic function names
    f = string_to_function(variables, formula)
    print(f({"X": X}))
    print(X[:, 1:2])

    formula = 'f_x1x2(X)'  # using generic function names
    f = string_to_function(variables, formula)
    print(f({"X": X}))

    formula = 'f(X)'  # using generic function names
    f = string_to_function(variables, formula)
    print(f({"X": X}))
    print(X[:, 0:1] * X[:, 1:2])

    # The following won't work
    # We should evaluate RHS, then assign to LHS in a state dictionary.
    # formula = 'f_x+f_y+f_xy'
    # f = string_to_function(variables, formula)
    # print(f({"X": X})(X))
    # f = string_to_function(variables, "sig=f_x(X)+f_y(X)")
    # f({"X": X})
    # g = string_to_function(variables, "sig2 = sig + 1")
    # g({"X": X})
    # print(sig)
    # print(sig2)

    