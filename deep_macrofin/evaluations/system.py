import re
from collections import OrderedDict
from typing import Callable, Dict, List, Union

import torch

from .comparators import Comparator
from .constraint import Constraint
from .endog_equation import EndogEquation
from .equation import Equation


class System:
    '''
    Represents a system to be evaluated when activation_constraints are all satisfied
    '''
    def __init__(self, 
                 activation_constraints: List[Constraint], 
                 label: str=None, 
                 latex_var_mapping: Dict[str, str] = {}):
        
        self.activation_constraints = activation_constraints
        self.label = label
        self.latex_var_mapping = latex_var_mapping
        assert len(self.activation_constraints) > 0, f"There must be at least one constraint for the system {self.label}."
        
        self.equations: Dict[str, Equation] = OrderedDict()
        self.endog_equations: Dict[str, EndogEquation] = OrderedDict()

        # label to value mapping, used to store all variable values and loss.
        self.variable_val_dict: Dict[str, torch.Tensor] = OrderedDict() # should include all local variables/params + current values, initially, all values in this dictionary can be zero
        self.loss_val_dict: Dict[str, torch.Tensor] = OrderedDict() # should include loss equation (constraints, endogenous equations, HJB equations) labels + corresponding loss values, initially, all values in this dictionary can be zero.
        self.loss_weight_dict: Dict[str, float] = OrderedDict() # should include loss equation labels + corresponding weight
        self.device = "cpu"

    def set_device(self, device):
        self.device = device
    
    def check_name_used(self, name):
        for self_dicts in [self.variable_val_dict,
                           self.loss_val_dict,
                           self.loss_weight_dict]:
            assert name not in self_dicts, f"Name: {name} is used"

    def check_label_used(self, label):
        for self_dicts in [self.variable_val_dict,
                           self.loss_val_dict,
                           self.loss_weight_dict]:
            assert label not in self_dicts, f"Label: {label} is used"

    def add_equation(self, eq: str, label: str=None):
        '''
        Add an equation to define a new variable. 
        '''
        if label is None:
            label = len(self.equations) + 1
        label = f"system_{self.label}_eq_{label}"
        self.check_label_used(label)
        new_eq = Equation(eq, label, self.latex_var_mapping)
        self.equations[label] = new_eq
        self.variable_val_dict[new_eq.lhs.formula_str] = torch.zeros(1, device=self.device)

    def add_endog_equation(self, eq: str, label: str=None, weight=1.0):
        '''
        Add an equation for loss computation based on endogenous variable
        '''
        if label is None:
            label = len(self.endog_equations) + 1
        label = f"system_{self.label}_endogeq_{label}"
        self.check_label_used(label)
        self.endog_equations[label] = EndogEquation(eq, label, self.latex_var_mapping)
        self.loss_val_dict[label] = torch.zeros(1, device=self.device)
        self.loss_weight_dict[label] = weight

    def compute_constraint_mask(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        '''
        Check if the constraint is satisfied. Need to check for each individual batch element.
        Get a mask for loss in each batch element
        '''
        mask = 1
        for constraint in self.activation_constraints:
            # eval no reduce always return lhs - rhs
            constraint_eval = constraint.eval_no_reduce(available_functions, variables)
            if constraint.comparator == Comparator.EQ:
                # satisfied when constraint_eval == 0
                mask *= torch.where(constraint_eval == 0, 1, 0)
            elif constraint.comparator == Comparator.LT:
                # satisfied when constraint_eval < 0
                mask *= torch.where(constraint_eval < 0, 1, 0)
            elif constraint.comparator == Comparator.LEQ:
                # satisfied when constraint_eval <= 0
                mask *= torch.where(constraint_eval <= 0, 1, 0)
            elif constraint.comparator == Comparator.GT:
                # satisfied when constraint_eval > 0
                mask *= torch.where(constraint_eval > 0, 1, 0)
            elif constraint.comparator == Comparator.GEQ:
                # satisfied when constraint_eval >= 0
                mask *= torch.where(constraint_eval >= 0, 1, 0)

        return mask

    def eval(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        '''
        compute the loss based on the system constraint
        '''
        variables_ = variables.copy()
        variables_.update(self.variable_val_dict)
        mask = self.compute_constraint_mask(available_functions, variables_)

        # properly update variables, using equations
        for eq_name in self.equations:
            lhs = self.equations[eq_name].lhs.formula_str
            res = self.equations[eq_name].eval(available_functions, variables_)
            variables_[lhs] = res
            self.variable_val_dict[lhs] = res
            if lhs in variables:
                variables[lhs][mask.squeeze(-1).to(torch.bool)] = res[mask.squeeze(-1).to(torch.bool)]


        # the loss will only be computed for a specific portion for the endogenous equations.
        for label in self.endog_equations:
            self.loss_val_dict[label] = self.endog_equations[label].eval_with_mask(available_functions, variables_, mask)

        total_loss = 0
        for loss_label, loss in self.loss_val_dict.items():
            total_loss += self.loss_weight_dict[loss_label] * loss

        return total_loss


    def __str__(self):
        str_repr = f"{self.label}: \n"
        str_repr += "Activation Constraints:\n"
        for constraint in self.activation_constraints:
            str_repr += str(constraint) + "\n"
        str_repr += "{0:=^40}\n".format("Equations")
        for eq_label, eq in self.equations.items():
            str_repr += str(eq) + "\n"
        str_repr += "\n"

        str_repr += "{0:=^40}\n".format("Endogenous Equations")
        for eq_label, eq in self.endog_equations.items():
            str_repr += str(eq) + "\n"
            str_repr += f"Loss weight: {self.loss_weight_dict[eq_label]}\n"
            str_repr += "-" * 40 + "\n"
        return str_repr