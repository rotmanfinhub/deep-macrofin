import re
from collections import OrderedDict
from typing import Callable, Dict, List, Union

import torch

from .comparators import Comparator
from .constraint import Constraint
from .endog_equation import EndogEquation
from .equation import Equation
from .loss_compute_methods import LOSS_REDUCTION_MAP, LossReductionMethod


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
        self.constraints: Dict[str, Constraint] = OrderedDict()

        self.loss_reduction_dict: Dict[str, LossReductionMethod] = OrderedDict() # used to store all loss function label to reduction method mappings

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

    def add_endog_equation(self, eq: str, label: str=None, weight=1.0, 
                loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
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
        if loss_reduction == LossReductionMethod.NONE:
            raise ValueError(f"{label}: None reduction is not supported in system")
        self.loss_reduction_dict[label] = loss_reduction

    def add_constraint(self, lhs: str, comparator: Comparator, rhs: str, label: str=None, weight=1.0, 
                loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
        '''
        comparator should be one of "=", ">", ">=", "<", "<=", we can use enum for this.

        Use Constraint class to properly convert it to a loss function.
        '''
        if label is None:
            label = len(self.constraints) + 1
        label = f"system_{self.label}_constraint_{label}"
        self.check_label_used(label)
        self.constraints[label] = Constraint(lhs, comparator, rhs, label, self.latex_var_mapping)
        self.loss_val_dict[label] = torch.zeros(1, device=self.device)
        self.loss_weight_dict[label] = weight
        if loss_reduction == LossReductionMethod.NONE:
            raise ValueError(f"{label}: None reduction is not supported in system")
        self.loss_reduction_dict[label] = loss_reduction

    def compute_constraint_mask(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor]):
        '''
        Check if the constraint is satisfied. Need to check for each individual batch element.
        Get a mask for loss in each batch element
        '''
        mask = 1
        for constraint in self.activation_constraints:
            # eval no loss returns non-zero in the locations where the conditions are not satisfied
            constraint_eval = constraint.eval_no_loss(available_functions, variables)
            mask *= torch.where(constraint_eval == 0, 1, 0)
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
            self.loss_val_dict[label] = self.endog_equations[label].eval_with_mask(available_functions, variables_, mask, self.loss_reduction_dict[label])

        for label in self.constraints:
            self.loss_val_dict[label] = self.constraints[label].eval_with_mask(available_functions, variables_, mask, self.loss_reduction_dict[label])

        total_loss = 0
        for loss_label, loss in self.loss_val_dict.items():
            total_loss += self.loss_weight_dict[loss_label] * loss

        return total_loss
    
    def eval_no_loss(self, available_functions: Dict[str, Callable], variables: Dict[str, torch.Tensor], batch_size: int):
        '''
        compute the loss based on the system constraint, but do not reduce the final loss. 
        Used for RAR and active learning
        '''
        variables_ = variables.copy()
        mask = self.compute_constraint_mask(available_functions, variables_)

        # properly update variables, using equations
        for eq_name in self.equations:
            lhs = self.equations[eq_name].lhs.formula_str
            res = self.equations[eq_name].eval(available_functions, variables_)
            variables_[lhs] = res

        total_loss = torch.zeros((batch_size, 1), device=self.device)

        # the loss will only be computed for a specific portion for the endogenous equations.
        for label in self.endog_equations:
            total_loss += torch.abs(self.endog_equations[label].eval_no_loss(available_functions, variables_)).reshape((batch_size, 1))

        for label in self.constraints:
            total_loss += torch.abs(self.constraints[label].eval_no_loss(available_functions, variables_)).reshape((batch_size, 1))
        
        # zero-mask the portion that doesn't satisfy the activation constraints. 
        total_loss = total_loss * mask
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

        str_repr += "{0:=^40}\n".format("Constraints")
        for const_label, const in self.constraints.items():
            str_repr += str(const) + "\n"
            str_repr += f"Loss weight: {self.loss_weight_dict[const_label]}\n"
            str_repr += "-" * 40 + "\n"
        return str_repr