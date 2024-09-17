import atexit
import json
import os
import time
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Dict, List

import pandas as pd
import torch
from tqdm import tqdm

from .evaluations import *
from .event_handler import *
from .models import *
from .utils import *


class PDEModel:
    '''
    PDEModel class to assign variables, equations & constraints, etc.

    Also initialize the neural network architectures for each agent/endogenous variables 
    with some config dictionary.
    '''
    
    '''
    Methods to initialize the model, define variables, and constraints
    '''
    def __init__(self, name: str, 
                 config: Dict[str, Any] = DEFAULT_CONFIG, 
                 latex_var_mapping: Dict[str, str] = {}):
        '''
        Initialize a model with the provided name and config. 
        The config should include the basic training configs, 

        The optimizer is default to AdamW

        DEFAULT_CONFIG={
            "batch_size": 100,
            "num_epochs": 1000,
            "lr": 1e-3,
            "loss_log_interval": 100,
            "optimizer_type": OptimizerType.AdamW,
            "sampling_method": SamplingMethod.UniformRandom,
            "refinement_sample_interval": int(0.2*num_epochs),
            "loss_balancing": False,
            "bernoulli_prob": 0.9999,
            "loss_balancing_temp": 0.1,
            "loss_balancing_alpha": 0.999,
            "soft_adapt_interval": -1,
            "loss_soft_attention": False,
        }

        loss_log_interval: the interval at which loss should be reported/recorded

        latex_var_mapping should include all possible latex to python name conversions. Otherwise latex parsing will fail. Can be omitted if all the input equations/formula are not in latex form. For details, check `Formula` class defined in `evaluations/formula.py`
        '''
        self.name = name
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(config)
        self.batch_size = self.config.get("batch_size", 100)
        self.num_epochs = self.config.get("num_epochs", 1000)
        self.lr = self.config.get("lr", 1e-3)
        self.loss_log_interval = self.config.get("loss_log_interval", 100)
        self.optimizer_type = self.config.get("optimizer_type", OptimizerType.AdamW)
        self.sampling_method = self.config.get("sampling_method", SamplingMethod.UniformRandom)

        self.SAMPLING_METHOD_MAP = {
            SamplingMethod.UniformRandom: self.sample_uniform,
            SamplingMethod.FixedGrid: self.sample_fixed_grid,
            SamplingMethod.ActiveLearning: self.sample_rar_greedy,
            SamplingMethod.RARG: self.sample_rar_greedy,
            SamplingMethod.RARD: self.sample_rar_distribution,
        }
        self.sample = self.SAMPLING_METHOD_MAP[self.sampling_method]

        self.latex_var_mapping = latex_var_mapping
        
        self.state_variables = []
        self.state_variable_constraints = {}

        # label to object mapping, used for actual evaluations
        self.agents: Dict[str, Agent] = OrderedDict()
        self.agent_conditions: Dict[str, AgentConditions] = OrderedDict()
        self.endog_vars: Dict[str, EndogVar] = OrderedDict()
        self.endog_var_conditions: Dict[str, EndogVarConditions] = OrderedDict()
        self.equations: Dict[str, Equation] = OrderedDict()
        self.endog_equations: Dict[str, EndogEquation] = OrderedDict()
        self.constraints: Dict[str, Constraint] = OrderedDict()
        self.hjb_equations : Dict[str, HJBEquation]= OrderedDict()
        self.systems: Dict[str, System] = OrderedDict()
        
        self.local_function_dict: Dict[str, Callable] = OrderedDict() # should include all functions available from agents and endogenous vars (direct evaluation and derivatives)

        self.loss_reduction_dict: Dict[str, LossReductionMethod] = OrderedDict() # used to store all loss function label to reduction method mappings

        # label to value mapping, used to store all variable values and loss.
        self.params: Dict[str, torch.Tensor] = OrderedDict()
        self.variable_val_dict: Dict[str, torch.Tensor] = OrderedDict() # should include all local variables/params + current values, initially, all values in this dictionary can be zero
        self.loss_val_dict: Dict[str, torch.Tensor] = OrderedDict() # should include loss equation (constraints, endogenous equations, HJB equations) labels + corresponding loss values, initially, all values in this dictionary can be zero.
        self.loss_weight_dict: Dict[str, float] = OrderedDict() # should include loss equation labels + corresponding weight
        self.device = "cpu"

        # for residual-based adaptive refinement (RAR) and active learning
        self.anchor_points: torch.Tensor = None
        self.refinement_sample_interval: int = self.config.get("refinement_sample_interval", int(0.2 * self.num_epochs))
        self.refinement_rounds: int = self.num_epochs // self.refinement_sample_interval

    def check_name_used(self, name: str):
        for self_dicts in [self.state_variables,
                           self.agents, 
                           self.endog_vars, 
                           self.local_function_dict, 
                           self.variable_val_dict,
                           self.loss_val_dict,
                           self.loss_weight_dict]:
            assert name not in self_dicts, f"Name: {name} is used"

    def check_label_used(self, label: str):
        for self_dicts in [self.state_variables,
                           self.agents, 
                           self.endog_vars, 
                           self.local_function_dict, 
                           self.variable_val_dict,
                           self.loss_val_dict,
                           self.loss_weight_dict]:
            assert label not in self_dicts, f"Label: {label} is used"

    def set_state(self, names: List[str], constraints: Dict[str, List] = {}):
        '''
        Set the state variables ("grid") of the problem.
        We probably want to add some constraints for each variable (domain). 
        By default, the constraints will be [-1, 1] (for easier sampling). 
        
        Only rectangular regions are supported
        '''
        assert len(self.agents) + len(self.endog_vars) == 0, "Neural networks for agents and endogenous variables have been initialized. State variables cannot be changed."
        for name in names:
            self.check_name_used(name)
        self.state_variables = names
        self.state_variable_constraints = {sv: [-1.0, 1.0] for sv in self.state_variables}
        self.state_variable_constraints.update(constraints)
        constraints_low = []
        constraints_high = []
        
        for svc in self.state_variables:
            constraints_low.append(self.state_variable_constraints[svc][0])
            constraints_high.append(self.state_variable_constraints[svc][1])
        self.state_variable_constraints["sv_low"] = constraints_low
        self.state_variable_constraints["sv_high"] = constraints_high

        for name in self.state_variables:
            self.variable_val_dict[name] = torch.zeros((self.batch_size, 1))

    def set_state_constraints(self, constraints: Dict[str, List] = {}):
        '''
        Overwrite the constraints for state variables, without changing the number of state variables.

        This can be used after loading a pre-trained model.
        '''
        for k in constraints:
            assert k in self.state_variables, f"{k} is not a state variable"
        self.state_variable_constraints.update(constraints)
        constraints_low = []
        constraints_high = []
        
        for svc in self.state_variables:
            constraints_low.append(self.state_variable_constraints[svc][0])
            constraints_high.append(self.state_variable_constraints[svc][1])
        self.state_variable_constraints["sv_low"] = constraints_low
        self.state_variable_constraints["sv_high"] = constraints_high

    def add_param(self, name: str, value: torch.Tensor):
        '''
        Add a single parameter (constant in the PDE system) with name and value.
        '''
        self.check_name_used(name)
        self.params[name] = value
        self.variable_val_dict[name] = value

    def add_params(self, params: Dict[str, Any]):
        '''
        Add a dictionary of parameters (constants in the PDE system) for the system.
        '''
        for name in params:
            self.check_name_used(name)
        self.params.update(params)
        self.variable_val_dict.update(params)

    def add_agent(self, name: str, 
                  config: Dict[str, Any] = DEFAULT_LEARNABLE_VAR_CONFIG,
                  overwrite=False):
        '''
        Add a single agent, with relevant config of neural network representation. 
        If called before states are set, should raise an error.

        Input:
        - overwrite: overwrite the previous agent with the same name, used for loading, default: False

        Config: specifies number of layers/hidden units of the neural network.
            - device: **str**, the device to run the model on (e.g., "cpu", "cuda"), default will be chosen based on whether or not GPU is available
            - hidden_units: **List[int]**, number of units in each layer, default: [30,30,30,30]
            - layer_type: **str**, a selection from the LayerType enum, default: LayerType.MLP
            - activation_type: **str**, a selection from the ActivationType enum, default: ActivationType.Tanh
            - positive: **bool**, apply softplus to the output to be always positive if true, default: false
            - hardcode_function: a lambda function for hardcoded forwarding function.
            - derivative_order: **int**, an additional constraint for the number of derivatives to take, default: 2, so for a function with one state variable, we can still take multiple derivatives
        '''
        assert len(self.state_variables) > 0, "Please set the state variables first"
        if not overwrite:
            self.check_name_used(name)
        agent_config = deepcopy(DEFAULT_LEARNABLE_VAR_CONFIG)
        agent_config.update(config)
        if not torch.cuda.is_available():
            agent_config["device"] = "cpu"

        self.device = agent_config["device"]

        new_agent = Agent(name, self.state_variables, agent_config)
        self.agents[name] = new_agent
        self.local_function_dict.update(new_agent.derivatives)
        for func_name in new_agent.derivatives:
            self.variable_val_dict[func_name] = torch.zeros((self.batch_size, 1), device=self.device)
    
    def add_agents(self, names: List[str], 
                   configs: Dict[str, Dict[str, Any]]={}):
        '''
        Add multiple agents at the same time, each with different configurations.

        Config: specifies number of layers/hidden units of the neural network.
            - device: **str**, the device to run the model on (e.g., "cpu", "cuda"), default will be chosen based on whether or not GPU is available
            - hidden_units: **List[int]**, number of units in each layer, default: [30,30,30,30]
            - layer_type: **str**, a selection from the LayerType enum, default: LayerType.MLP
            - activation_type: **str**, a selection from the ActivationType enum, default: ActivationType.Tanh
            - positive: **bool**, apply softplus to the output to be always positive if true, default: false
            - hardcode_function: a lambda function for hardcoded forwarding function.
            - derivative_order: **int**, an additional constraint for the number of derivatives to take, default: 2, so for a function with one state variable, we can still take multiple derivatives
        '''
        assert len(self.state_variables) > 0, "Please set the state variables first"
        for name in names:
            self.add_agent(name, configs.get(name, DEFAULT_LEARNABLE_VAR_CONFIG))

    def add_agent_condition(self, name: str, 
                            lhs: str, lhs_state: Dict[str, torch.Tensor], 
                            comparator: Comparator, 
                            rhs: str, rhs_state: Dict[str, torch.Tensor], 
                            label: str=None,
                            weight: float=1.0, 
                            loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
        '''
        Add boundary/initial condition for a specific agent

        Input:
        - name: **str**, agent name, 
        - lhs: **str**, the string expression for lhs formula, latex expression not supported, should be functions of specific format agent_name(SV), or simply a constant value
        - lhs_state: **Dict[str, torch.Tensor]**, the specific value of SV to evaluate lhs at for the agent/endogenous variable
        - comparator: **Comparator**
        - rhs: **str**, the string expression for lhs formula, latex expression not supported, should be functions of specific format agent_name(SV), or simply a constant value
        - rhs_state: **Dict[str, torch.Tensor]**, the specific value of SV to evaluate rhs at for the agent/endogenous variable, if rhs is a constant, this can be an empty dictionary
        - label: **str** label for the condition
        - weight: **float**, weight in total loss computation
        - loss_reduction: **LossReductionMethod**, `LossReductionMethod.MSE` for mean squared error, or `LossReductionMethod.MAE` for mean absolute error
        '''
        assert name in self.agents, f"Agent {name} does not exist"
        if label is None:
            label = len(self.agent_conditions) + 1
        label = f"agent_{name}_cond_{label}"
        self.check_label_used(label)
        self.agent_conditions[label] = AgentConditions(name, 
                                                       lhs, lhs_state, 
                                                       comparator, 
                                                       rhs, rhs_state,
                                                       label, self.latex_var_mapping)
        self.loss_val_dict[label] = torch.zeros(1, device=self.device)
        self.loss_weight_dict[label] = weight
        self.loss_reduction_dict[label] = loss_reduction

    def add_endog(self, name: str, 
                  config: Dict[str, Any] = DEFAULT_LEARNABLE_VAR_CONFIG,
                  overwrite=False):
        '''
        Add a single unknown endogenous variable, with relevant config of NN. 
        If called before states are set, should raise an error.

        Input:
        - overwrite: overwrite the previous agent with the same name, used for loading, default: False

        Config: specifies number of layers/hidden units of the neural network.
            - device: **str**, the device to run the model on (e.g., "cpu", "cuda"), default will be chosen based on whether or not GPU is available
            - hidden_units: **List[int]**, number of units in each layer, default: [30,30,30,30]
            - layer_type: **str**, a selection from the LayerType enum, default: LayerType.MLP
            - activation_type: **str**, a selection from the ActivationType enum, default: ActivationType.Tanh
            - positive: **bool**, apply softplus to the output to be always positive if true, default: false
            - hardcode_function: a lambda function for hardcoded forwarding function.
            - derivative_order: **int**, an additional constraint for the number of derivatives to take, default: 2, so for a function with one state variable, we can still take multiple derivatives
        '''
        assert len(self.state_variables) > 0, "Please set the state variables first"
        if not overwrite:
            self.check_name_used(name)
        endog_var_config = deepcopy(DEFAULT_LEARNABLE_VAR_CONFIG)
        endog_var_config.update(config)
        if not torch.cuda.is_available():
            endog_var_config["device"] = "cpu"

        self.device = endog_var_config["device"]

        new_endog_var = EndogVar(name, self.state_variables, endog_var_config)
        self.endog_vars[name] = new_endog_var
        self.local_function_dict.update(new_endog_var.derivatives)
        for func_name in new_endog_var.derivatives:
            self.variable_val_dict[func_name] = torch.zeros((self.batch_size, 1), device=self.device)

    def add_endogs(self, names: List[str], 
                   configs: Dict[str, Dict[str, Any]] = {}):
        '''
        Add multiple endogenous variables at the same time, each with different config.

        Config: specifies number of layers/hidden units of the neural network.
            - device: **str**, the device to run the model on (e.g., "cpu", "cuda"), default will be chosen based on whether or not GPU is available
            - hidden_units: **List[int]**, number of units in each layer, default: [30,30,30,30]
            - layer_type: **str**, a selection from the LayerType enum, default: LayerType.MLP
            - activation_type: **str**, a selection from the ActivationType enum, default: ActivationType.Tanh
            - positive: **bool**, apply softplus to the output to be always positive if true, default: false
            - hardcode_function: a lambda function for hardcoded forwarding function.
            - derivative_order: **int**, an additional constraint for the number of derivatives to take, default: 2, so for a function with one state variable, we can still take multiple derivatives
        '''
        assert len(self.state_variables) > 0, "Please set the state variables first"
        for name in names:
            self.add_endog(name, configs.get(name, DEFAULT_LEARNABLE_VAR_CONFIG))
    
    def add_endog_condition(self, name: str, 
                            lhs: str, lhs_state: Dict[str, torch.Tensor], 
                            comparator: Comparator, 
                            rhs: str, rhs_state: Dict[str, torch.Tensor], 
                            label: str=None,
                            weight=1.0, 
                            loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
        '''
        Add boundary/initial condition for a specific endogenous var

        Input:
        - name: **str**, agent name, 
        - lhs: **str**, the string expression for lhs formula, latex expression not supported, should be functions of specific format agent_name(SV), or simply a constant value
        - lhs_state: **Dict[str, torch.Tensor]**, the specific value of SV to evaluate lhs at for the agent/endogenous variable
        - comparator: **Comparator**
        - rhs: **str**, the string expression for lhs formula, latex expression not supported, should be functions of specific format agent_name(SV), or simply a constant value
        - rhs_state: **Dict[str, torch.Tensor]**, the specific value of SV to evaluate rhs at for the agent/endogenous variable, if rhs is a constant, this can be an empty dictionary
        - label: **str** label for the condition
        - weight: **float**, weight in total loss computation
        - loss_reduction: **LossReductionMethod**, `LossReductionMethod.MSE` for mean squared error, or `LossReductionMethod.MAE` for mean absolute error
        '''
        assert name in self.endog_vars, f"Endogenous variable {name} does not exist"
        if label is None:
            label = len(self.endog_var_conditions) + 1
        label = f"endogvar_{name}_cond_{label}"
        self.check_label_used(label)
        self.endog_var_conditions[label] = EndogVarConditions(name, 
                                                       lhs, lhs_state, 
                                                       comparator, 
                                                       rhs, rhs_state,
                                                       label, self.latex_var_mapping)
        self.loss_val_dict[label] = torch.zeros(1, device=self.device)
        self.loss_weight_dict[label] = weight
        self.loss_reduction_dict[label] = loss_reduction

    def add_equation(self, eq: str, label: str=None):
        '''
        Add an equation to define a new variable. 
        '''
        if label is None:
            label = len(self.equations) + 1
        label = f"eq_{label}"
        self.check_label_used(label)
        new_eq = Equation(eq, label, self.latex_var_mapping)
        self.equations[label] = new_eq
        self.variable_val_dict[new_eq.lhs.formula_str] = torch.zeros((self.batch_size, 1), device=self.device)

    def add_endog_equation(self, eq: str, label: str=None, weight=1.0, 
                loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
        '''
        Add an equation for loss computation based on endogenous variable
        '''
        if label is None:
            label = len(self.endog_equations) + 1
        label = f"endogeq_{label}"
        self.check_label_used(label)
        self.endog_equations[label] = EndogEquation(eq, label, self.latex_var_mapping)
        self.loss_val_dict[label] = torch.zeros(1, device=self.device)
        self.loss_weight_dict[label] = weight
        self.loss_reduction_dict[label] = loss_reduction

    def add_constraint(self, lhs: str, comparator: Comparator, rhs: str, label: str=None, weight=1.0, 
                loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
        '''
        comparator should be one of "=", ">", ">=", "<", "<=", we can use enum for this.

        Use Constraint class to properly convert it to a loss function.
        '''
        if label is None:
            label = len(self.constraints) + 1
        label = f"constraint_{label}"
        self.check_label_used(label)
        self.constraints[label] = Constraint(lhs, comparator, rhs, label, self.latex_var_mapping)
        self.loss_val_dict[label] = torch.zeros(1, device=self.device)
        self.loss_weight_dict[label] = weight
        self.loss_reduction_dict[label] = loss_reduction

    def add_hjb_equation(self, eq: str, label: str=None, weight=1.0, 
                loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
        '''
        Add an equation for loss computation based on an HJB equation (residual form)
        '''
        if label is None:
            label = len(self.hjb_equations) + 1
        label = f"hjbeq_{label}"
        self.check_label_used(label)
        self.hjb_equations[label] = HJBEquation(eq, label, self.latex_var_mapping)
        self.loss_val_dict[label] = torch.zeros(1, device=self.device)
        self.loss_weight_dict[label] = weight
        self.loss_reduction_dict[label] = loss_reduction


    def add_system(self, system: System, weight=1.0):
        '''
        Add a pre-compiled system, which should consist of activation constraint and 
        associated equation(new variable def)/endogenous equation (loss)
        '''
        if system.label is None:
            system.label = len(self.systems) + 1
        label = f"system_{system.label}"
        self.check_label_used(label)
        system.set_device(self.device)
        self.systems[label] = system
        self.loss_val_dict[label] = torch.zeros(1, device=self.device)
        self.loss_weight_dict[label] = weight

    def update_variables(self, SV):
        '''
        update the agent/endogenous variables and variables defined by users using equations
        '''
        # properly update variables, including agent, endogenous variables, their derivatives
        for func_name in self.local_function_dict:
            self.variable_val_dict[func_name] = self.local_function_dict[func_name](SV)

        # properly update variables, using equations
        for eq_name in self.equations:
            lhs = self.equations[eq_name].lhs.formula_str
            self.variable_val_dict[lhs] = self.equations[eq_name].eval({}, self.variable_val_dict)
    
    def loss_fn(self):
        '''
        Compute the loss function, using the endogenous equation/constraints defined.
        The loss is based on 
        self.agent_conditions, 
        self.endog_var_conditions,
        self.endog_equations,
        self.constraints,
        self.hjb_equations,
        self.systems
        '''
        # for agent and endogenous variable conditions, we need to use the exact function to compute the values
        for label in self.agent_conditions:
            self.loss_val_dict[label] = self.agent_conditions[label].eval(self.local_function_dict, self.loss_reduction_dict[label])
        
        for label in self.endog_var_conditions:
            self.loss_val_dict[label] = self.endog_var_conditions[label].eval(self.local_function_dict, self.loss_reduction_dict[label])

        # for all other formula/equations, we can use the pre-computed values of a specific state to compute the loss
        for label in self.endog_equations:
            self.loss_val_dict[label] = self.endog_equations[label].eval({}, self.variable_val_dict, self.loss_reduction_dict[label])

        for label in self.constraints:
            self.loss_val_dict[label] = self.constraints[label].eval({}, self.variable_val_dict, self.loss_reduction_dict[label])

        for label in self.hjb_equations:
            self.loss_val_dict[label] = self.hjb_equations[label].eval({}, self.variable_val_dict, self.loss_reduction_dict[label])

        for label in self.systems:
            self.loss_val_dict[label] = self.systems[label].eval({}, self.variable_val_dict)

    def closure(self, SV):
        self.update_variables(SV)
        self.loss_fn()
        total_loss = 0
        for loss_label, loss in self.loss_val_dict.items():
            total_loss += self.loss_weight_dict[loss_label] * torch.where(loss.isnan(), 0.0, loss)

        self.optimizer.zero_grad()
        total_loss.backward()
        return total_loss        
    
    def closure_soft_attention(self, SV):
        self.update_variables(SV)
        total_loss = 0
        for label in self.agent_conditions:
            self.loss_val_dict[label] = torch.square(self.agent_conditions[label].eval(self.local_function_dict, LossReductionMethod.NONE))
            temp = torch.nanmean(self.loss_weight_dict[label] * self.loss_val_dict[label])
            total_loss += torch.where(temp.isnan(), 0.0, temp)
        
        for label in self.endog_var_conditions:
            self.loss_val_dict[label] = torch.square(self.endog_var_conditions[label].eval(self.local_function_dict, LossReductionMethod.NONE))
            temp = torch.nanmean(self.loss_weight_dict[label] * self.loss_val_dict[label])
            total_loss += torch.where(temp.isnan(), 0.0, temp)

        for label in self.endog_equations:
            self.loss_val_dict[label] = torch.square(self.endog_equations[label].eval_no_loss({}, self.variable_val_dict)).reshape((self.batch_size, 1))
            temp = torch.nanmean(self.loss_weight_dict[label] * self.loss_val_dict[label])
            total_loss += torch.where(temp.isnan(), 0.0, temp)

        for label in self.constraints:
            self.loss_val_dict[label] = torch.square(self.constraints[label].eval_no_loss({}, self.variable_val_dict)).reshape((self.batch_size, 1))
            temp = torch.nanmean(self.loss_weight_dict[label] * self.loss_val_dict[label])
            total_loss += torch.where(temp.isnan(), 0.0, temp)

        for label in self.hjb_equations:
            self.loss_val_dict[label] = torch.square(self.hjb_equations[label].eval_no_loss({}, self.variable_val_dict)).reshape((self.batch_size, 1))
            temp = torch.nanmean(self.loss_weight_dict[label] * self.loss_val_dict[label])
            total_loss += torch.where(temp.isnan(), 0.0, temp)

        for label in self.systems:
            self.loss_val_dict[label] = torch.square(self.systems[label].eval_no_loss({}, self.variable_val_dict, self.batch_size)).reshape((self.batch_size, 1))
            temp = torch.nanmean(self.loss_weight_dict[label] * self.loss_val_dict[label])
            total_loss += torch.where(temp.isnan(), 0.0, temp)

        self.optimizer.zero_grad()
        total_loss.backward()
        return total_loss

        
    def test_step(self, SV):
        '''
        initialize random state variable with proper constraints, compute loss
        '''
        self.update_variables(SV)
        self.loss_fn()
        total_loss = 0
        for loss_label, loss in self.loss_val_dict.items():
            total_loss += torch.nanmean(self.loss_weight_dict[loss_label] * torch.where(loss.isnan(), 0.0, loss))

        loss_dict = self.loss_val_dict.copy()
        if self.config.get("loss_soft_attention", False):
                for k in loss_dict:
                    loss_dict[k] = torch.mean(loss_dict[k])
        loss_dict["total_loss"] = total_loss
        return loss_dict
    
    def set_all_model_training(self):
        for agent_name in self.agents:
            self.agents[agent_name].train()
        for endog_var_name in self.endog_vars:
            self.endog_vars[endog_var_name].train()

    def set_all_model_eval(self):
        for agent_name in self.agents:
            self.agents[agent_name].eval()
        for endog_var_name in self.endog_vars:
            self.endog_vars[endog_var_name].eval()
    
    def set_config(self, config: Dict[str, Any] = DEFAULT_CONFIG):
        '''
        This function overwrites the existing configurations.

        DEFAULT_CONFIG={
            "batch_size": 100,
            "num_epochs": 1000,
            "lr": 1e-3,
            "loss_log_interval": 100,
            "optimizer_type": OptimizerType.AdamW,
            "sampling_method": SamplingMethod.UniformRandom,
            "refinement_sample_interval": int(0.2*num_epochs),
            "loss_balancing": False,
            "bernoulli_prob": 0.9999,
            "loss_balancing_temp": 0.1,
            "loss_balancing_alpha": 0.999,
            "soft_adapt_interval": -1,
            "loss_soft_attention": False,
        }
        '''
        self.config.update(config)
        self.batch_size = self.config.get("batch_size", 100)
        self.num_epochs = self.config.get("num_epochs", 1000)
        self.lr = self.config.get("lr", 1e-3)
        self.loss_log_interval = self.config.get("loss_log_interval", 100)
        self.optimizer_type = self.config.get("optimizer_type", OptimizerType.AdamW)
        self.sampling_method = self.config.get("sampling_method", SamplingMethod.UniformRandom)
        self.sample = self.SAMPLING_METHOD_MAP[self.sampling_method]
        self.refinement_sample_interval = self.config.get("refinement_sample_interval", int(0.2 * self.num_epochs))
        self.refinement_rounds = self.num_epochs // self.refinement_sample_interval

    def set_loss_reduction(self, label: str, loss_reduction: LossReductionMethod):
        if label not in self.loss_reduction_dict:
            raise ValueError(f"{label} is not a valid label for loss function")
        self.loss_reduction_dict[label] = loss_reduction
    
    def sample_uniform(self, epoch):
        SV = np.random.uniform(low=self.state_variable_constraints["sv_low"], 
                         high=self.state_variable_constraints["sv_high"], 
                         size=(self.batch_size, len(self.state_variables)))
        return torch.Tensor(SV).to(self.device)
    
    def sample_fixed_grid(self, epoch):
        if len(self.state_variables) == 1:
            return torch.linspace(self.state_variable_constraints["sv_low"][0], 
                                  self.state_variable_constraints["sv_high"][0], 
                                  steps=self.batch_size, device=self.device).view(-1, 1)
        else:
            sv_ls = [0] * len(self.state_variables)
            for i in range(len(self.state_variables)):
                sv_ls[i] = torch.linspace(self.state_variable_constraints["sv_low"][i], 
                                        self.state_variable_constraints["sv_high"][i], 
                                        steps=self.batch_size, device=self.device)
            return torch.cartesian_prod(*sv_ls)
        
    def get_refinement_loss_dict(self, epoch):
        '''
        Sample a dense subset of the problem domain, compute the loss and return total loss for each point sampled. Used for Residual-based Adaptive Refinement and Active Learning

        Returns:
            {
                "SV": sampled state variables, shape (100000, len(self.state_variables))
                "loss": total loss computed at each sv, shape (100000, 1)
            }
        '''
        # because we need a set of dense points to compute residual for adaptive sampling
        # we set all models to evaluation models so that gradients won't be computed.
        # it speeds up the computation and reduces memory usages
        self.set_all_model_eval()

        # Temporarily set a large batch size for each dimension
        self.batch_size = 100000
        SV = self.sample_uniform(epoch)

        # make a copy of variable value mapping
        # so that we don't break the top level training routine
        variable_val_dict_ = self.variable_val_dict.copy()
        total_loss = torch.zeros((self.batch_size, 1), device=self.device)

        # forward pass
        for i, sv_name in enumerate(self.state_variables):
            variable_val_dict_[sv_name] = SV[:, i:i+1]

        # update variables, including agent, endogenous variables, their derivatives
        for func_name in self.local_function_dict:
            variable_val_dict_[func_name] = self.local_function_dict[func_name](SV)

        # update variables, using equations
        for eq_name in self.equations:
            lhs = self.equations[eq_name].lhs.formula_str
            variable_val_dict_[lhs] = self.equations[eq_name].eval({}, variable_val_dict_)

        # compute total losses, without reducing to a single value, keep the original dimension, but summing up using abs values
        # Note that the conditions (IC/BC, or user pre-defined sampling regions) are not considered
        # Systems are not considered
        for label in self.endog_equations:
            total_loss += torch.abs(self.endog_equations[label].eval_no_loss({}, variable_val_dict_)).reshape((self.batch_size, 1))

        for label in self.constraints:
            total_loss += torch.abs(self.constraints[label].eval_no_loss({}, variable_val_dict_)).reshape((self.batch_size, 1))

        for label in self.hjb_equations:
            total_loss += torch.abs(self.hjb_equations[label].eval_no_loss({}, variable_val_dict_)).reshape((self.batch_size, 1))

        for label in self.systems:
            total_loss += torch.abs(self.systems[label].eval_no_loss({}, variable_val_dict_, self.batch_size)).reshape((self.batch_size, 1))

        self.batch_size = self.config.get("batch_size", 100) # reset the batch size for normal computation
        self.set_all_model_training() # reset the model for training stage

        return {
            "SV": SV,
            "loss": total_loss,
        }
        
    def sample_rar_greedy(self, epoch):
        if epoch % self.refinement_sample_interval == 0 and epoch > 0:
            # check epoch > 0 so we don't need to resample in the first epoch
            refinement_loss_dict = self.get_refinement_loss_dict(epoch)
            SV = refinement_loss_dict["SV"]
            all_losses = refinement_loss_dict["loss"]
            X_ids = torch.topk(all_losses, self.batch_size**len(self.state_variables)//self.refinement_rounds, dim=0)[1].squeeze(-1)
            self.anchor_points = torch.vstack((self.anchor_points, SV[X_ids]))
        fixed_grid_sv = self.sample_fixed_grid(epoch)
        return torch.vstack((fixed_grid_sv, self.anchor_points))
        

    def sample_rar_distribution(self, epoch):
        if epoch % self.refinement_sample_interval == 0 and epoch > 0:
            # check epoch > 0 so we don't need to resample in the first epoch
            refinement_loss_dict = self.get_refinement_loss_dict(epoch)
            SV = refinement_loss_dict["SV"]
            all_losses = refinement_loss_dict["loss"] ** 2
            err_eq = all_losses / all_losses.mean()
            err_eq_normalized = (err_eq / err_eq.sum())[:, 0]
            X_ids = np.random.choice(a=SV.shape[0], size=self.batch_size**len(self.state_variables)//self.refinement_rounds, replace=False, p=err_eq_normalized.detach().cpu().numpy())
            self.anchor_points = torch.vstack((self.anchor_points, SV[X_ids]))
        fixed_grid_sv = self.sample_fixed_grid(epoch)
        return torch.vstack((fixed_grid_sv, self.anchor_points))
    
    def init_loss_balancing(self, *args, **kwargs):
        '''
        Initialize variables for relative loss balancing with random lookback
        https://arxiv.org/pdf/2110.09813
        '''
        self.loss_weight_log_dict = defaultdict(list)
        self.bernoulli_rho = torch.distributions.Bernoulli(self.config.get("bernoulli_prob", 0.9999))
        self.temp = self.config.get("loss_balancing_temp", 0.1)
        self.alpha = self.config.get("loss_balancing_alpha", 0.999)
        self.init_loss_tensor = torch.zeros(len(self.loss_val_dict), device=self.device)
        self.prev_loss_tensor = torch.zeros(len(self.loss_val_dict), device=self.device)

    def loss_balancing_step(self, *args, **kwargs):
        epoch = kwargs.get("epoch", 0)
        if epoch == 0:
            for i, (loss_label, loss) in enumerate(self.loss_val_dict.items()):
                self.init_loss_tensor[i] = torch.where(loss.isnan(), torch.finfo(loss.dtype).eps, loss)
                self.prev_loss_tensor[i] = torch.where(loss.isnan(), torch.finfo(loss.dtype).eps, loss)
        else:
            # relative loss balancing with random lookback
            # https://arxiv.org/pdf/2110.09813
            curr_loss_tensor = torch.zeros_like(self.prev_loss_tensor, device=self.device)
            prev_loss_weight_tensor = torch.zeros_like(self.prev_loss_tensor, device=self.device)
            for i, (loss_label, loss) in enumerate(self.loss_val_dict.items()):
                curr_loss_tensor[i] = torch.where(loss.isnan(), torch.finfo(loss.dtype).eps, loss)
                prev_loss_weight_tensor[i] = self.loss_weight_dict[loss_label]
            
            ratio_prev = curr_loss_tensor / (self.temp * self.prev_loss_tensor)
            ratio_zero = curr_loss_tensor / (self.temp * self.init_loss_tensor)
            bal_prev = len(self.loss_val_dict) * torch.nn.functional.softmax(ratio_prev, dim=-1)
            bal_zero = len(self.loss_val_dict) * torch.nn.functional.softmax(ratio_zero, dim=-1)
            rho = self.bernoulli_rho.sample()
            weight_hist = rho * prev_loss_weight_tensor + (1 - rho) * bal_zero
            new_weight = self.alpha * weight_hist + (1 - self.alpha) * bal_prev
            for i, k in enumerate(self.loss_weight_dict):
                self.loss_weight_dict[k] = new_weight[i].item()
                
            self.loss_weight_log_dict["epoch"].append(epoch)
            for k, v in self.loss_weight_dict.items():
                self.loss_weight_log_dict[k].append(v)
            
            for i, (loss_label, loss) in enumerate(self.loss_val_dict.items()):
                self.prev_loss_tensor[i] = torch.where(loss.isnan(), torch.finfo(loss.dtype).eps, loss).item()

    def init_soft_adapt(self, *args, **kwargs):
        '''
        Initialize variables for soft adapt
        https://arxiv.org/pdf/1912.12355
        '''
        self.loss_adaption_interval = self.config.get("soft_adapt_interval", -1)
        self.beta = self.config.get("loss_balancing_temp", 0.1)
        self.loss_weight_log_dict = defaultdict(list)
        self.loss_hist = {}
        self.loss_hist_count = 0
        for label in self.loss_val_dict:
            self.loss_hist[label] = torch.zeros(self.loss_adaption_interval, device=self.device)

    def soft_adapt_step(self, *args, **kwargs):
        epoch = kwargs.get("epoch", 0)
        for loss_label, loss in self.loss_val_dict.items():
            self.loss_hist[loss_label][self.loss_hist_count] = torch.where(loss.isnan(), torch.finfo(loss.dtype).eps, loss)
        self.loss_hist_count += 1
        if self.loss_hist_count == self.loss_adaption_interval:
            # soft adapt implementation following https://arxiv.org/pdf/1912.12355
            # Loss Weighted + normalized
            # the idea is to assign higher weight to loss functions that decreases more slowly.
            self.loss_hist_count = 0
            rates_of_change = torch.zeros(len(self.loss_hist))
            avg_loss_values = torch.zeros(len(self.loss_hist))
            for i, label in enumerate(self.loss_hist):
                # use the mean rate of change, instead of using finite difference for faster computation
                diff = self.loss_hist[label][1:] - self.loss_hist[label][:-1]
                rates_of_change[i] = torch.mean(diff)
                avg_loss_values[i] = torch.mean(self.loss_hist[label])
            # normalization
            rates_of_change = rates_of_change / torch.sum(torch.abs(rates_of_change))
            # loss weighted softmax
            new_weights = torch.nn.functional.softmax(self.beta * (rates_of_change - rates_of_change.max()), dim=-1)
            new_weights = avg_loss_values * new_weights / torch.sum(avg_loss_values * new_weights)
            
            for i, label in enumerate(self.loss_hist):
                self.loss_weight_dict[label] = new_weights[i].item()
                self.loss_hist[label] = torch.zeros(self.loss_adaption_interval)
            
            self.loss_weight_log_dict["epoch"].append(epoch)
            for k, v in self.loss_weight_dict.items():
                self.loss_weight_log_dict[k].append(v)

    def init_soft_attention(self, *args, **kwargs):
        # a random sample to determine batch size
        assert self.sampling_method == SamplingMethod.FixedGrid, "Soft Attention only works for Fixed Grid sampling."
        SV = self.sample(0)
        B = SV.shape[0]
        self.loss_weight_log_dict = defaultdict(list)
        all_params = []
        # start with uniform weights for each training point
        for k in self.loss_weight_dict:
            if k in self.agent_conditions or k in self.endog_var_conditions:
                # it's either in agent condition or in endog var condition
                cond = self.agent_conditions.get(k, self.endog_var_conditions[k])
                bs = None
                for kk, v in cond.lhs_state.items():
                    if isinstance(v, torch.Tensor):
                        bs = v.shape[0]
                        break
                if bs is None:
                    for kk, v in cond.rhs_state.items():
                        if isinstance(v, torch.Tensor):
                            bs = v.shape[0]
                            break
                curr_params = nn.Parameter(torch.ones((bs, 1), device=self.device))
            else:
                curr_params = nn.Parameter(torch.ones((B, 1), device=self.device))
            self.loss_weight_dict[k] = curr_params
            all_params += [curr_params]
        # perform gradient ascent, the weights should never go negative.
        self.optimizer.add_param_group({"params": all_params, "maximize": True})

    def soft_attention_step(self, *args, **kwargs):
        # The optimization step has been performed in the main loop, so we only need to record the weights
        epoch = kwargs.get("epoch", 0)
        SV = kwargs.get("SV")
        for i in range(SV.shape[0]):
            self.loss_weight_log_dict["epoch"].append(epoch)
            for n, sv_name in enumerate(self.state_variables):
                self.loss_weight_log_dict[sv_name].append(SV[i, n].item())
            for k, v in self.loss_weight_dict.items():
                if k in self.agent_conditions or k in self.endog_var_conditions:
                    continue
                self.loss_weight_log_dict[k].append(v[i, 0].item())


    def train_model(self, model_dir: str="./", filename: str=None, full_log=False):
        '''
        The entire loop of training
        '''

        if self.anchor_points is None:
            self.anchor_points = torch.empty((0, len(self.state_variables)), device=self.device)

        min_loss = torch.inf
        epoch_loss_dict = defaultdict(list)
        min_loss_dict = defaultdict(list)
        all_params = []
        
        model_has_kan = False
        for agent_name, agent in self.agents.items():
            all_params += list(agent.parameters())
            if agent.config["layer_type"] == LayerType.KAN:
                model_has_kan = True
        for endog_var_name, endog_var in self.endog_vars.items():
            all_params += list(endog_var.parameters())
            if endog_var.config["layer_type"] == LayerType.KAN:
                model_has_kan = True
        
        if model_has_kan:
            # KAN can only be trained with LBFGS, 
            # as long as there is one model with KAN, we must route to the default LBFGS
            self.optimizer = LBFGS(all_params, lr=self.lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
        else:
            self.optimizer = OPTIMIZER_MAP[self.optimizer_type](all_params, self.lr)

        os.makedirs(model_dir, exist_ok=True)
        if filename is None:
            filename = filename = self.name
        if "." in filename:
            file_prefix = filename.split(".")[0]
        else:
            file_prefix = filename
        
        log_fn = os.path.join(model_dir, f"{file_prefix}-{self.num_epochs}-log.txt")
        log_file = open(log_fn, "w", encoding="utf-8")

        @atexit.register
        def cleanup_file():
            # make sure the log file is properly closed even after exception
            log_file.close()

        print(str(self), file=log_file)
        self.validate_model_setup(model_dir)
        print("{0:=^80}".format("Training"))
        self.set_all_model_training()
        start_time = time.time()

        OnTrainingStart = EventHandler()
        OnTrainingStep = EventHandler()
        if self.config.get("loss_balancing", False):
            OnTrainingStart += self.init_loss_balancing
            OnTrainingStep += self.loss_balancing_step
        elif self.config.get("loss_soft_attention", False):
            OnTrainingStart += self.init_soft_attention
            OnTrainingStep += self.soft_attention_step
            # override the closure function
            self.closure = self.closure_soft_attention
        elif self.config.get("soft_adapt_interval", -1) > 0:
            OnTrainingStart += self.init_soft_adapt
            OnTrainingStep += self.soft_adapt_step
        
        OnTrainingStart()
        set_seeds(0)
        pbar = tqdm(range(self.num_epochs))
        for epoch in pbar:
            epoch_start_time = time.time()
            
            SV = self.sample(epoch)
            for i, sv_name in enumerate(self.state_variables):
                self.variable_val_dict[sv_name] = SV[:, i:i+1]

            self.optimizer.step(lambda: self.closure(SV))
            total_loss = 0
            for loss_label, loss in self.loss_val_dict.items():
                total_loss += torch.nanmean(self.loss_weight_dict[loss_label] * torch.where(loss.isnan(), 0.0, loss))

            loss_dict = self.loss_val_dict.copy()
            if self.config.get("loss_soft_attention", False):
                for k in loss_dict:
                    loss_dict[k] = torch.mean(loss_dict[k])
            loss_dict["total_loss"] = total_loss

            if full_log:
                formatted_train_loss = ",\n".join([f'{k}: {v:.4f}' for k, v in loss_dict.items()])
            else:
                formatted_train_loss = "%.4f" % loss_dict["total_loss"].item()
            
            if loss_dict["total_loss"].item() < min_loss and all(not v.isnan() for v in loss_dict.values()):
                min_loss = loss_dict["total_loss"].item()
                self.save_model(model_dir, f"{file_prefix}_best.pt")
                min_loss_dict["epoch"].append(len(min_loss_dict["epoch"]))
                for k, v in loss_dict.items():
                    min_loss_dict[k].append(v.item())
            # maybe reload the best model when loss is nan.

            OnTrainingStep(epoch=epoch, SV=SV)

            if epoch % self.loss_log_interval == 0:
                epoch_loss_dict["epoch"].append(epoch)
                for k, v in loss_dict.items():
                    epoch_loss_dict[k].append(v.item())
                pbar.set_description("Total loss: {0:.4f}".format(loss_dict["total_loss"].item()))
            print(f"epoch {epoch}: \ntrain loss :: {formatted_train_loss},\ntime elapsed :: {time.time() - epoch_start_time}", file=log_file)
        print(f"training finished, total time :: {time.time() - start_time}")
        print(f"training finished, total time :: {time.time() - start_time}", file=log_file)
        log_file.close()
        if loss_dict["total_loss"].item() < min_loss and all(not v.isnan() for v in loss_dict.values()):
            self.save_model(model_dir, f"{file_prefix}_best.pt")
        print(f"Best model saved to {model_dir}/{file_prefix}_best.pt if valid")
        self.save_model(model_dir, filename, verbose=True)
        pd.DataFrame(epoch_loss_dict).to_csv(f"{model_dir}/{file_prefix}_loss.csv", index=False)
        pd.DataFrame(min_loss_dict).to_csv(f"{model_dir}/{file_prefix}_min_loss.csv", index=False)

        if self.anchor_points.shape[0] > 0:
            anchor_points_np = self.anchor_points.detach().cpu().numpy()
            np.save(f"{model_dir}/{file_prefix}_anchor_points.npy", anchor_points_np)
            print(f"Anchor points saved to {model_dir}/{file_prefix}_anchor_points.npy")

        if hasattr(self, "loss_weight_log_dict"):
            pd.DataFrame(self.loss_weight_log_dict).to_csv(f"{model_dir}/{file_prefix}_loss_weight.csv", index=False)

        return loss_dict
    
    def train_model_active_learning(self, active_learning_regions: List[List[float]]=[], model_dir: str="./", filename: str=None, full_log=False):
        '''
        The entire loop of training
        active_learning_regions: list of [low, high], describing regions for active learning
        '''
        min_loss = torch.inf
        epoch_loss_dict = defaultdict(list)
        min_loss_dict = defaultdict(list)
        all_params = []
        
        model_has_kan = False
        for agent_name, agent in self.agents.items():
            all_params += list(agent.parameters())
            if agent.config["layer_type"] == LayerType.KAN:
                model_has_kan = True
        for endog_var_name, endog_var in self.endog_vars.items():
            all_params += list(endog_var.parameters())
            if endog_var.config["layer_type"] == LayerType.KAN:
                model_has_kan = True
        
        if model_has_kan:
            # KAN can only be trained with LBFGS, 
            # as long as there is one model with KAN, we must route to the default LBFGS
            self.optimizer = LBFGS(all_params, lr=self.lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
        else:
            self.optimizer = OPTIMIZER_MAP[self.optimizer_type](all_params, self.lr)

        os.makedirs(model_dir, exist_ok=True)
        if filename is None:
            filename = filename = self.name
        if "." in filename:
            file_prefix = filename.split(".")[0]
        else:
            file_prefix = filename
        
        log_fn = os.path.join(model_dir, f"{file_prefix}-{self.num_epochs}-log.txt")
        log_file = open(log_fn, "w", encoding="utf-8")

        @atexit.register
        def cleanup_file():
            # make sure the log file is properly closed even after exception
            log_file.close()

        print(str(self), file=log_file)
        self.validate_model_setup(model_dir)
        print("{0:=^80}".format("Training"))
        self.set_all_model_training()
        start_time = time.time()
        set_seeds(0)
        pbar = tqdm(range(self.num_epochs))

        active_learning_grids = []
        for al_region in active_learning_regions:
            active_learning_grids.append(np.linspace(al_region[0], al_region[1], num=self.batch_size)[:, np.newaxis])
        active_learning_grids = torch.Tensor(np.concatenate(active_learning_grids)).to(self.device)
        for epoch in pbar:
            epoch_start_time = time.time()

            SV = np.random.uniform(low=self.state_variable_constraints["sv_low"], 
                         high=self.state_variable_constraints["sv_high"], 
                         size=(self.batch_size, len(self.state_variables)))
            SV = torch.Tensor(SV).to(self.device)
            SV = torch.cat([SV, active_learning_grids])
            for i, sv_name in enumerate(self.state_variables):
                self.variable_val_dict[sv_name] = SV[:, i:i+1]

            self.optimizer.step(lambda: self.closure(SV))
            total_loss = 0
            for loss_label, loss in self.loss_val_dict.items():
                total_loss += self.loss_weight_dict[loss_label] * torch.where(loss.isnan(), 0.0, loss)

            loss_dict = self.loss_val_dict.copy()
            loss_dict["total_loss"] = total_loss

            if full_log:
                formatted_train_loss = ",\n".join([f'{k}: {v:.4f}' for k, v in loss_dict.items()])
            else:
                formatted_train_loss = "%.4f" % loss_dict["total_loss"]
            
            if loss_dict["total_loss"].item() < min_loss and all(not v.isnan() for v in loss_dict.values()):
                min_loss = loss_dict["total_loss"].item()
                self.save_model(model_dir, f"{file_prefix}_best.pt")
                min_loss_dict["epoch"].append(len(min_loss_dict["epoch"]))
                for k, v in loss_dict.items():
                    min_loss_dict[k].append(v.item())
            # maybe reload the best model when loss is nan.

            if epoch % self.loss_log_interval == 0:
                epoch_loss_dict["epoch"].append(epoch)
                for k, v in loss_dict.items():
                    epoch_loss_dict[k].append(v.item())
                pbar.set_description("Total loss: {0:.4f}".format(loss_dict["total_loss"]))
            print(f"epoch {epoch}: \ntrain loss :: {formatted_train_loss},\ntime elapsed :: {time.time() - epoch_start_time}", file=log_file)
        print(f"training finished, total time :: {time.time() - start_time}")
        print(f"training finished, total time :: {time.time() - start_time}", file=log_file)
        log_file.close()
        if loss_dict["total_loss"].item() < min_loss and all(not v.isnan() for v in loss_dict.values()):
            self.save_model(model_dir, f"{file_prefix}_best.pt")
        print(f"Best model saved to {model_dir}/{file_prefix}_best.pt if valid")
        self.save_model(model_dir, filename, verbose=True)
        pd.DataFrame(epoch_loss_dict).to_csv(f"{model_dir}/{file_prefix}_loss.csv", index=False)
        pd.DataFrame(min_loss_dict).to_csv(f"{model_dir}/{file_prefix}_min_loss.csv", index=False)

        return loss_dict

    
    def eval_model(self, full_log=False):
        '''
        The entire loop of evaluation
        '''
        self.anchor_points = torch.empty((0, len(self.state_variables)), device=self.device)
        
        self.validate_model_setup()
        self.set_all_model_eval()
        print("{0:=^80}".format("Evaluating"))
        SV = self.sample(0)
        for i, sv_name in enumerate(self.state_variables):
            self.variable_val_dict[sv_name] = SV[:, i:i+1]
        loss_dict = self.test_step(SV)

        if full_log:
            formatted_loss = ",\n".join([f'{k}: {v:.4f}' for k, v in loss_dict.items()])
        else:
            formatted_loss = "%.4f" % loss_dict["total_loss"].item()
        print(f"loss :: {formatted_loss}")
        return loss_dict

    def validate_model_setup(self, model_dir="./"):
        '''
        Check that all the equations/constraints given are valid. If not, log the errors in a file, and raise an ultimate error.

        Need to check the following:
        self.agents,
        self.agent_conditions,
        self.endog_vars,
        self.endog_var_conditions,
        self.equations,
        self.endog_equations,
        self.constraints,
        self.hjb_equations,
        self.systems,
        '''
        errors = []
        sv = self.sample(0)
        variable_val_dict_ = self.variable_val_dict.copy()
        for i, sv_name in enumerate(self.state_variables):
            variable_val_dict_[sv_name] = sv[:, i:i+1]

        for agent_name in self.agents:
            try:
                y = self.agents[agent_name].forward(sv)
                assert y.shape[0] == sv.shape[0] and y.shape[1] == 1
            except Exception as e:
                errors.append({
                    "label": agent_name, 
                    "error": str(e)
                })

        for endog_var_name in self.endog_vars:
            try:
                y = self.endog_vars[endog_var_name].forward(sv)
                assert y.shape[0] == sv.shape[0] and y.shape[1] == 1
            except Exception as e:
                errors.append({
                    "label": endog_var_name,
                    "error": str(e),
                })
        
        for func_name in self.local_function_dict:
            variable_val_dict_[func_name] = self.local_function_dict[func_name](sv)

        for label in self.agent_conditions:
            try:
                self.agent_conditions[label].eval(self.local_function_dict)
            except Exception as e:
                if e is not ZeroDivisionError:
                    # it's fine to have zero division. All other errors should be raised
                    errors.append({
                        "label": label,
                        "repr": self.agent_conditions[label].lhs.formula_str + self.agent_conditions[label].comparator + self.agent_conditions[label].rhs.formula_str,
                        "error": str(e),
                        "info": " Please use SV as the hard coded state variable inputs, in lhs or rhs"
                    })
        
        for label in self.endog_var_conditions:
            try:
                self.endog_var_conditions[label].eval(self.local_function_dict)
            except Exception as e:
                if e is not ZeroDivisionError:
                    errors.append({
                        "label": label,
                        "repr": self.endog_var_conditions[label].lhs.formula_str + self.endog_var_conditions[label].comparator + self.endog_var_conditions[label].rhs.formula_str,
                        "error": str(e),
                        "info": " Please use SV as the hard coded state variable inputs, in lhs or rhs"
                    })
        
        for label in self.equations:
            try:
                lhs = self.equations[label].lhs.formula_str
                variable_val_dict_[lhs] = self.equations[label].eval({}, variable_val_dict_)
            except Exception as e:
                if e is not ZeroDivisionError:
                    errors.append({
                        "label": label,
                        "raw": self.equations[label].eq,
                        "parsed": f"{self.equations[label].lhs.formula_str}={self.equations[label].rhs.formula_str}",
                        "error": str(e)
                    })


        for label in self.endog_equations:
            try:
                self.endog_equations[label].eval({}, variable_val_dict_)
            except Exception as e:
                if e is not ZeroDivisionError:
                    errors.append({
                        "label": label,
                        "raw": self.endog_equations[label].eq,
                        "parsed": f"{self.endog_equations[label].lhs.formula_str}={self.endog_equations[label].rhs.formula_str}",
                        "error": str(e)
                    })

        for label in self.constraints:
            try:
                self.constraints[label].eval({}, variable_val_dict_)
            except Exception as e:
                if e is not ZeroDivisionError:
                    errors.append({
                        "label": label,
                        "parsed": self.constraints[label].lhs.formula_str + self.constraints[label].comparator + self.constraints[label].rhs.formula_str,
                        "error": str(e)
                    })

        for label in self.hjb_equations:
            try:
                self.hjb_equations[label].eval({}, variable_val_dict_)
            except Exception as e:
                if e is not ZeroDivisionError:
                    errors.append({
                        "label": label,
                        "raw": self.hjb_equations[label].eq,
                        "parsed": self.hjb_equations[label].parsed_eq.formula_str,
                        "error": str(e)
                    })

        for label in self.systems:
            try:
                self.systems[label].eval({}, variable_val_dict_)
            except Exception as e:
                if e is not ZeroDivisionError:
                    # it's fine to have zero division. All other errors should be raised
                    errors.append({
                        "label": label,
                        "repr": str(self.systems[label]),
                        "error": str(e)
                    })

        if len(errors) > 0:
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, f"{self.name}-errors.txt"), "w", encoding="utf-8") as f:
                f.write("Error Log:\n")
                f.write(json.dumps(errors, indent=True))
            print(json.dumps(errors, indent=True))
            raise Exception(f"Errors when validating model setup, please check {self.name}-errors.txt for details.")

    def save_model(self, model_dir: str = "./", filename: str=None, verbose=False):
        '''
        Save all the agents, endogenous variables (pytorch model and configurations), 
        and all other configurations of the PDE model.

        Inputs:
        - model_dir: the directory to save the model
        - filename: the filename to save the model without suffix, default: self.name 
        '''
        if filename is None:
            filename = self.name
        dict_to_save = {
            "name": self.name,
            "config": self.config,
            "latex_var_mapping": self.latex_var_mapping,
            "state_variables": self.state_variables,
            "state_variable_constraints": self.state_variable_constraints,
            "loss_weight_dict": self.loss_weight_dict,
        }

        for agent in self.agents:
            dict_to_save[f"agent_{agent}_dict"] = self.agents[agent].to_dict()

        for endog_var in self.endog_vars:
            dict_to_save[f"endog_var_{endog_var}_dict"] = self.endog_vars[endog_var].to_dict()

        os.makedirs(model_dir, exist_ok=True)
        torch.save(dict_to_save, f"{model_dir}/{filename}")
        if verbose:
            print(f"Model saved to {model_dir}/{filename}")
    
    def load_model(self, dict_to_load: Dict[str, Any]):
        '''
        Load all the agents, endogenous variables (pytorch model and configurations) from the dictionary
        '''
        self.latex_var_mapping = dict_to_load["latex_var_mapping"]
        self.state_variables = dict_to_load["state_variables"]
        self.state_variable_constraints = dict_to_load["state_variable_constraints"]
        for k, v in dict_to_load.items():
            if k.startswith("agent") and k.endswith("dict"):
                agent_name = v["name"]
                agent_config = v["model_config"]
                self.add_agent(agent_name, agent_config, overwrite=True)
                self.agents[agent_name].from_dict(v)

            if k.startswith("endog_var") and k.endswith("dict"):
                endog_var_name = v["name"]
                endog_var_config = v["model_config"]
                self.add_endog(endog_var_name, endog_var_config, overwrite=True)
                self.endog_vars[endog_var_name].from_dict(v)
        print("Model loaded")

    def __str__(self):
        total_param_count = 0
        str_repr = "{0:=^80}\n".format(f"Summary of Model {self.name}")
        str_repr += "Config: " + json.dumps(self.config, indent=True) + "\n"
        str_repr += "Latex Variable Mapping:\n" + json.dumps(self.latex_var_mapping, indent=True) + "\n"
        try:
            str_repr += "User Defined Parameters:\n" + json.dumps(self.params, indent=True) + "\n\n"
        except:
            param_copy = self.params.copy()
            for k, v in param_copy.items():
                if isinstance(v, torch.Tensor):
                    param_copy[k] = v.item()
            str_repr += "User Defined Parameters:\n" + json.dumps(param_copy, indent=True) + "\n\n"

        str_repr += "{0:=^80}\n".format("State Variables")
        for sv in self.state_variables:
            str_repr += f"{sv}: {self.state_variable_constraints[sv]}\n"
        str_repr += "\n"
        
        str_repr += "{0:=^80}\n".format("Agents")
        for agent_name, agent_model in self.agents.items():
            str_repr += f"Agent Name: {agent_name}\n"
            str_repr += str(agent_model) + "\n"
            num_param = agent_model.get_num_params()
            total_param_count += num_param
            str_repr += f"Num parameters: {num_param}\n"
            str_repr += "-" * 80 + "\n"
        str_repr += "\n"

        str_repr += "{0:=^80}\n".format("Agent Conditions")
        for agent_cond_name, agent_cond in self.agent_conditions.items():
            str_repr += str(agent_cond) + "\n"
            str_repr += f"Loss weight: {self.loss_weight_dict[agent_cond_name]}\n"
            str_repr += "-" * 80 + "\n"
        str_repr += "\n"

        str_repr += "{0:=^80}\n".format("Endogenous Variables")
        for endog_var_name, endog_var in self.endog_vars.items():
            str_repr += f"Endogenous Variable Name: {endog_var_name}\n"
            str_repr += str(endog_var) + "\n"
            num_param = endog_var.get_num_params()
            total_param_count += num_param
            str_repr += f"Num parameters: {num_param}\n"
            str_repr += "-" * 80 + "\n"
        str_repr += "\n"

        str_repr += "{0:=^80}\n".format("Endogenous Variables Conditions")
        for endog_var_cond_name, endog_var_cond in self.endog_var_conditions.items():
            str_repr += str(endog_var_cond) + "\n"
            str_repr += f"Loss weight: {self.loss_weight_dict[endog_var_cond_name]}\n"
            str_repr += "-" * 80 + "\n"
        str_repr += "\n"

        str_repr += "{0:=^80}\n".format("Equations")
        for eq_label, eq in self.equations.items():
            str_repr += str(eq) + "\n"
        str_repr += "\n"

        str_repr += "{0:=^80}\n".format("Endogenous Equations")
        for eq_label, eq in self.endog_equations.items():
            str_repr += str(eq) + "\n"
            str_repr += f"Loss weight: {self.loss_weight_dict[eq_label]}\n"
            str_repr += "-" * 80 + "\n"
        str_repr += "\n"

        str_repr += "{0:=^80}\n".format("Constraints")
        for constraint_label, constraint in self.constraints.items():
            str_repr += str(constraint) + "\n"
            str_repr += f"Loss weight: {self.loss_weight_dict[constraint_label]}\n"
            str_repr += "-" * 80 + "\n"
        str_repr += "\n"

        str_repr += "{0:=^80}\n".format("HJB Equations")
        for hjb_label, hjb_eq in self.hjb_equations.items():
            str_repr += str(hjb_eq) + "\n"
            str_repr += f"Loss weight: {self.loss_weight_dict[hjb_label]}\n"
            str_repr += "-" * 80 + "\n"
        str_repr += "\n"

        str_repr += "{0:=^80}\n".format("Systems")
        for system_label, sys in self.systems.items():
            str_repr += str(sys) + "\n"
            str_repr += f"System loss weight: {self.loss_weight_dict[system_label]}\n"
            str_repr += "-" * 80 + "\n"
        str_repr += "\n"

        return str_repr
    
    def plot_vars(self, vars_to_plot: List[str], ncols: int=4, elev=30, azim=-135, roll=0):
        '''
        Inputs:
        - vars_to_plot: variable names to plot, can be an equation defining a new variable. If Latex, need to be enclosed by $$ symbols
        - ncols: number of columns to plot, default: 4
        - elev, azim, roll: view angles for 3D plots.  https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
        This function is only supported for 1D or 2D state_variables.
        '''
        assert len(self.state_variables) <= 2, "Plot is only supported for problems with no more than 2 state variables"

        variable_var_dict_ = self.variable_val_dict.copy()
        var_to_latex = {}
        for k, v in self.latex_var_mapping.items():
            var_to_latex[v] = k

        sv_ls = [0] * len(self.state_variables)
        for i in range(len(self.state_variables)):
            sv_ls[i] = torch.linspace(self.state_variable_constraints["sv_low"][i], 
                                      self.state_variable_constraints["sv_high"][i], steps=100, device=self.device)
        X = torch.cartesian_prod(*sv_ls)
        
        nrows = len(vars_to_plot) // ncols
        if len(vars_to_plot) % ncols > 0:
            nrows += 1
        if len(self.state_variables) == 1:
            fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
            SV = X.unsqueeze(-1)
            X = X.detach().cpu().numpy().reshape(-1)
            for i, sv_name in enumerate(self.state_variables):
                variable_var_dict_[sv_name] = SV[:, i:i+1]
            # properly update variables, including agent, endogenous variables, their derivatives
            for func_name in self.local_function_dict:
                variable_var_dict_[func_name] = self.local_function_dict[func_name](SV)

            # properly update variables, using equations
            for eq_name in self.equations:
                lhs = self.equations[eq_name].lhs.formula_str
                variable_var_dict_[lhs] = self.equations[eq_name].eval({}, variable_var_dict_)

            sv_text = self.state_variables[0]
            if self.state_variables[0] in var_to_latex:
                sv_text = f"${var_to_latex[self.state_variables[0]]}$"

            for i, curr_var in enumerate(vars_to_plot):
                curr_row = i // ncols
                curr_col = i % ncols
                if nrows == 1:
                    if ncols == 1:
                        curr_ax = ax
                    else:
                        curr_ax = ax[curr_col]
                else:
                    curr_ax = ax[curr_row][curr_col]
                if "$" in curr_var:
                    # parse latex and potentially equation
                    if "=" in curr_var:
                        curr_eq = Equation(curr_var, f"plot_eq{i}", self.latex_var_mapping)
                        lhs = curr_eq.lhs.formula_str
                        variable_var_dict_[lhs] = curr_eq.eval({}, variable_var_dict_)
                        curr_ax.plot(X, variable_var_dict_[lhs].detach().cpu().numpy().reshape(-1))
                        curr_ax.set_xlabel(sv_text)
                        lhs_unparsed = curr_var.split("=")[0].replace("$", "").strip()
                        curr_ax.set_ylabel(f"${lhs_unparsed}$")
                        curr_ax.set_title(f"${lhs_unparsed}$ vs {sv_text}")
                    else:
                        base_var = curr_var.replace("$", "").strip()
                        base_var_non_latex = self.latex_var_mapping.get(base_var, base_var)
                        curr_ax.plot(X, variable_var_dict_[base_var_non_latex].detach().cpu().numpy().reshape(-1))
                        curr_ax.set_xlabel(sv_text)
                        curr_ax.set_ylabel(curr_var)
                        curr_ax.set_title(f"{curr_var} vs {sv_text}")
                else:
                    if "=" in curr_var:
                        curr_eq = Equation(curr_var, f"plot_eq{i}", self.latex_var_mapping)
                        lhs = curr_eq.lhs.formula_str
                        variable_var_dict_[lhs] = curr_eq.eval({}, variable_var_dict_)
                        curr_ax.plot(X, variable_var_dict_[lhs].detach().cpu().numpy().reshape(-1))
                        curr_ax.set_xlabel(sv_text)
                        curr_ax.set_ylabel(lhs)
                        curr_ax.set_title(f"{lhs} vs {sv_text}")
                    else:
                        curr_ax.plot(X, variable_var_dict_[curr_var].detach().cpu().numpy().reshape(-1))
                        curr_ax.set_xlabel(sv_text)
                        curr_ax.set_ylabel(curr_var)
                        curr_ax.set_title(f"{curr_var} vs {sv_text}")
            plt.tight_layout()
            plt.show()
        else:
            fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), subplot_kw={"projection": "3d"})
            SV = torch.clone(X)
            X, Y = torch.meshgrid(sv_ls, indexing="ij")
            X = X.detach().cpu().numpy()
            Y = Y.detach().cpu().numpy()
            for i, sv_name in enumerate(self.state_variables):
                variable_var_dict_[sv_name] = SV[:, i:i+1]
            # properly update variables, including agent, endogenous variables, their derivatives
            for func_name in self.local_function_dict:
                variable_var_dict_[func_name] = self.local_function_dict[func_name](SV)

            # properly update variables, using equations
            for eq_name in self.equations:
                lhs = self.equations[eq_name].lhs.formula_str
                variable_var_dict_[lhs] = self.equations[eq_name].eval({}, variable_var_dict_)

            sv_text0 = self.state_variables[0]
            sv_text1 = self.state_variables[1]
            if self.state_variables[0] in var_to_latex:
                sv_text0 = f"${var_to_latex[self.state_variables[0]]}$"
            if self.state_variables[1] in var_to_latex:
                sv_text1 = f"${var_to_latex[self.state_variables[1]]}$"

            for i, curr_var in enumerate(vars_to_plot):
                curr_row = i // ncols
                curr_col = i % ncols
                if nrows == 1:
                    if ncols == 1:
                        curr_ax = ax
                    else:
                        curr_ax = ax[curr_col]
                else:
                    curr_ax = ax[curr_row][curr_col]
                if "$" in curr_var:
                    # parse latex and potentially equation
                    if "=" in curr_var:
                        curr_eq = Equation(curr_var, f"plot_eq{i}", self.latex_var_mapping)
                        lhs = curr_eq.lhs.formula_str
                        variable_var_dict_[lhs] = curr_eq.eval({}, variable_var_dict_)
                        curr_ax.plot_surface(X, Y, variable_var_dict_[lhs].detach().cpu().numpy().reshape(100, 100))
                        curr_ax.set_xlabel(sv_text0)
                        curr_ax.set_ylabel(sv_text1)
                        lhs_unparsed = curr_var.split("=")[0].replace("$", "").strip()
                        curr_ax.set_zlabel(f"${lhs_unparsed}$")
                        curr_ax.set_title(f"${lhs_unparsed}$ vs ({sv_text0}, {sv_text1})")
                    else:
                        base_var = curr_var.replace("$", "").strip()
                        base_var_non_latex = self.latex_var_mapping.get(base_var, base_var)
                        curr_ax.plot_surface(X, Y, variable_var_dict_[base_var_non_latex].detach().cpu().numpy().reshape(100, 100))
                        curr_ax.set_xlabel(sv_text0)
                        curr_ax.set_ylabel(sv_text1)
                        curr_ax.set_zlabel(curr_var)
                        curr_ax.set_title(f"{curr_var} vs ({sv_text0}, {sv_text1})")
                else:
                    if "=" in curr_var:
                        curr_eq = Equation(curr_var, f"plot_eq{i}", self.latex_var_mapping)
                        lhs = curr_eq.lhs.formula_str
                        variable_var_dict_[lhs] = curr_eq.eval({}, variable_var_dict_)
                        curr_ax.plot_surface(X, Y, variable_var_dict_[lhs].detach().cpu().numpy().reshape(100, 100))
                        curr_ax.set_xlabel(sv_text0)
                        curr_ax.set_ylabel(sv_text1)
                        curr_ax.set_zlabel(curr_var)
                        curr_ax.set_title(f"{lhs} vs ({sv_text0}, {sv_text1})")
                    else:
                        curr_ax.plot_surface(X, Y, variable_var_dict_[curr_var].detach().cpu().numpy().reshape(100, 100))
                        curr_ax.set_xlabel(sv_text0)
                        curr_ax.set_ylabel(sv_text1)
                        curr_ax.set_zlabel(curr_var)
                        curr_ax.set_title(f"{curr_var} vs ({sv_text0}, {sv_text1})")
                curr_ax.view_init(elev, azim, roll)
                curr_ax.set_box_aspect(None, zoom=0.85)
            plt.tight_layout()
            plt.show()

