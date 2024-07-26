import re
from enum import Enum
from typing import Callable, Dict, List, Union

import torch


class EvaluationMethod(str, Enum):
    Eval = "eval"
    Sympy = "sympy"
    AST = "ast"


LEFT_BRACKETS = r"\\left[\[\{(]"
RIGHT_BRACKETS = r"\\right[\]\})]"
DERIVATIVE_PATTERN = r"\\frac{\s*\\partial\^?(\d*)\s*(.*?)}{(.*?)}"
DERIVATIVE_LOWER_PATTERN = r"\\partial\s*"
DERIVATIVE_PATTERN2 = r"\\frac{\s*d\^?(\d*)\s*(.*?)}{(.*?)}" # not used
DERIVATIVE_LOWER_PATTERN2 = r"d\s*"
FRACTION_PATTERN = r"\\frac{((?!.*\\frac).*?)}{((?!.*\\frac).*?)}" # make sure no additional fraction is within each group, so we parse from inner most fraction to outer fractions.
POWER_PATTERN = r"\^(\w+|\{(.*?)\})"
SQRT_PATTERN = r"\\sqrt\{(.*?)\}" # make sure no additional \{ is within the bracket

def latex_parsing_sqrt(formula_str: str):
    def sqrt_match(match: re.Match):
        # Remove the curly braces if they exist
        group = match.group(1).strip('{}')
        return 'sqrt(' + group + ')'
    formula_str = re.sub(SQRT_PATTERN, sqrt_match, formula_str)
    return formula_str

def latex_parsing_brackets(formula_str: str):
    formula_str = re.sub(LEFT_BRACKETS, "(", formula_str)
    formula_str = re.sub(RIGHT_BRACKETS, ")", formula_str)
    return formula_str

def latex_parsing_powers(formula_str: str):
    def power_match(match: re.Match):
        # Remove the curly braces if they exist
        group = match.group(1).strip('{}')
        return '**(' + group + ')'
    formula_str = re.sub(POWER_PATTERN, power_match, formula_str)
    return formula_str

def latex_parsing_fractions(formula_str: str):
    '''
    This parses both the derivatives and common fractions
    '''
    def derivative_match(match: re.Match):
        function_name = match.group(2)
        variables = re.split(DERIVATIVE_LOWER_PATTERN, match.group(3))
        for i, var in enumerate(variables):
            if "^" in var:
                # higher order of a single variable
                var_name, var_power = var.split("^")
                variables[i] = var_name.strip() * int(var_power.strip())
            else:
                variables[i] = var.strip()
        return function_name + "_" + "".join(variables)
    
    def fraction_match(match: re.Match):
        numer = match.group(1)
        denom = match.group(2)
        return f"({numer})/({denom})"
    
    # firstly parse all the derivatives
    while len(re.findall(DERIVATIVE_PATTERN, formula_str)) > 0:
        formula_str = re.sub(DERIVATIVE_PATTERN, derivative_match, formula_str)
    
    # then parse all other fractions from inside out
    while len(re.findall(FRACTION_PATTERN, formula_str)) > 0:
        formula_str = re.sub(FRACTION_PATTERN, fraction_match, formula_str)
    return formula_str

def latex_parsing_functions(formula_str: str):
    formula_str = formula_str.replace(r"\log", "log")
    formula_str = formula_str.replace(r"\exp", "exp")
    formula_str = formula_str.replace(r"\sin", "sin")
    formula_str = formula_str.replace(r"\cos", "cos")
    formula_str = formula_str.replace(r"\tan", "tan")
    return formula_str

def latex_parsing(formula_str: str, latex_var_mapping: Dict[str, str] = {}):
    '''
    Experimental: Parse a formula in latex form to a string that can be evaluated in python

    Input:
        formula_str: the string to parse
        latex_var_mapping: a dictionary with key being the latex form of a variable, 
        and value being the corresponding representation in python code.
    '''

    # remove the $ symbols from the str
    formula_str = formula_str.replace("$", "").replace("&", "")
    # replace all latex variables with known python variables
    for ltx, var in latex_var_mapping.items():
        formula_str = formula_str.replace(ltx, var)

    formula_str = latex_parsing_brackets(formula_str)
    formula_str = latex_parsing_fractions(formula_str)
    formula_str = latex_parsing_powers(formula_str)
    formula_str = latex_parsing_sqrt(formula_str)
    formula_str = latex_parsing_functions(formula_str)
    return formula_str

torch_func_dict = {
    "pi": torch.pi,
    "sqrt": torch.sqrt,
    "log": torch.log,
    "exp": torch.exp,
    "sinh": torch.sinh,
    "cosh": torch.cosh,
    "tanh": torch.tanh,
    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,
    "arcsin": torch.arcsin,
    "arccos": torch.arccos,
    "arctan": torch.arctan,
}

class Formula:
    '''
    Given a string representation of a formula, and a set of variables 
    (state, value, prices, etc) in a model, parse the formula to a pytorch 
    function that can be evaluated. 

    Latex equation with restricted format is supported for initialization. 
    '''
    
    def __init__(self, formula_str: str, evaluation_method: Union[EvaluationMethod, str], latex_var_mapping: Dict[str, str] = {}):
        '''
        Inputs:
        - formula_str: the string version of the formula. If the provided formula_str is supposed to be a latex string, it must be $ enclosed and in the regular form, e.g. formula_str=r"$x^2*y$", and all multiplication symbols must be explicitly provided as * in the equation.
        - evaluation_method: Enum, select from `eval`, `sympy`, `ast`, 
        corresponding to the four methods below. For now, only consider eval.
        - latex_var_mapping: only used if the formula_str is in latex form, the keys should be the latex expression, and the values should be the corresponding python variable name. 
        All strings with single slash in latex must be defined as a raw string. 
        All spaces in the key must match exactly as in the input formula_str. e.g. latex_var_map = {
                r"\eta_t": "eta",
                r"\rho^i": "rhoi",
                r"\mu^{n h}_t": "munh",
                r"\sigma^{na}_t": "signa",
                r"\sigma^{n ia}_t": "signia",
                r"\sigma_t^{qa}": "sigqa",
                "c_t^i": "ci",
                "c_t^h": "ch",
            }
        '''
        self.formula_str = formula_str.strip()
        if "$" in formula_str:
            self.formula_str = latex_parsing(formula_str, latex_var_mapping)
            self.formula_str = self.formula_str.strip() # to avoid additional spaces
        try:
            self.formula_compiled = compile(self.formula_str, "<string>", "eval")
        except:
            self.formula_compiled = self.formula_str
        self.evaluation_method = evaluation_method

        if self.evaluation_method == EvaluationMethod.Eval:
            self.local_context = {"__builtins__": None}
            self.local_context.update(torch.__dict__)
            self.eval = self.eval_str
        else:
            raise NotImplementedError(f"{evaluation_method} is not implemented")

    def eval_str(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        '''
        Evaluate the formula with existing functions and provided assignments to variables
        This evaluates the function by simple string parsing
        '''
        # Create a local context with available functions and variables
        self.local_context.update(available_functions)
        self.local_context.update(variables)

        # Directly evaluate the formula string in the context of available functions and variables
        result = eval(self.formula_compiled, {"__builtins__": None}, self.local_context)
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
        
if __name__ == "__main__":
    latex_var_map = {
        r"\rho^i": "rhoi",
        r"\zeta^i": "zetai",
        r"\rho^h": "rhoh",
        r"\zeta^h": "zetah",
        r"\xi^i": "xii",
        r"\xi^h": "xih",
        r"\mu^{n i}_t": "muni",
        r"\mu^{n h}_t": "munh",
        r"\sigma^{na}_t": "signa",
        r"\sigma^{n ia}_t": "signia",
        r"\sigma_t^{qa}": "sigqa",
        r"c_t^i": "ci",
        r"c_t^h": "ch",
        
        "q_t^a": "qa",
        r"\xi_t^i": "xii",
        r"\xi_t^h": "xih",
        r"\eta_t": "eta",
        r"\sigma^{\eta a}_t": "sigea",
        r"\sigma_t^{qa}": "sigqa",
        r"\sigma_t^{\xi ia}": "sigxia",
        r"\sigma_t^{\xi ha}": "sigxha",
        r"\mu^{\eta}_t": "mue",
    }
    ltx1 = r"\sigma_t^{qa}"
    print(latex_parsing(ltx1, latex_var_map))
    ltx2 = r"\eta_t^2"
    print(latex_parsing(ltx2, latex_var_map))
    ltx3 = r"\frac{\partial^2 x1}{\partial x2\partial x3} \frac{ \partial qa}{\partial t^2}"
    print(latex_parsing(ltx3, latex_var_map))
    ltx4 = r"\frac{ \partial qa}{\partial t}"
    print(latex_parsing(ltx4, latex_var_map))

    ltx5 = r"\frac{\rho^j}{1-\frac{1}{\zeta^j}} \left( \left(\frac{c_t^j}{\xi^j} \right)^{1-1/\zeta^j}-1 \right)".replace("j", "i")
    print(latex_parsing(ltx5, latex_var_map))
    ltx6 = r"(1-\eta_t)*(\mu^{n i}_t - \mu^{n h}_t) +(\sigma^{na}_t)^2  - \sigma^{n ia}_t*\sigma^{na}_t"
    print(latex_parsing(ltx6, latex_var_map))
    
    ltx7 = r'\frac{\partial q_t^a}{\partial \eta_t} * \sigma^{\eta a}_t * \eta_t'
    print(latex_parsing(ltx7, latex_var_map))

    ltx8 = r'(\frac{\sqrt{3}}{3} * R_b - \frac{\sqrt{3}}{3} * R_c)'
    print(latex_parsing(ltx8, latex_var_map))