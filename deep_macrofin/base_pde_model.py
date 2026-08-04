import json
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, List

import torch
from torch import nn

from .evaluations import *
from .event_handler import *
from .models import *
from .utils import *


class BasePDEModel:
    '''
    Abstract base class shared by :class:`PDEModel` (stationary) and
    :class:`PDEModelTimeStep` (time-stepping scheme).

    It holds everything the two solvers have in common: variable/equation/
    constraint definition APIs, the forward evaluation of agents and endogenous
    variables (``update_variables``/``_eval_local_functions``), loss computation,
    model setup validation, saving/loading and the string summary.

    Subclasses must implement ``train_model`` and ``eval_model`` (they differ
    fundamentally), and typically add their own sampling/validation helpers.

    This class is not meant to be instantiated directly.
    '''

    def _init_common(self):
        '''
        Initialize all the shared containers used to hold variables, equations,
        constraints, losses and their values. Subclass ``__init__`` methods are
        expected to have already set ``self.name``, ``self.config``,
        ``self.latex_var_mapping`` and the config-derived scalar attributes
        before calling this.
        '''
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
        self.hjb_equations: Dict[str, HJBEquation] = OrderedDict()
        self.systems: Dict[str, System] = OrderedDict()

        self.local_function_dict: Dict[str, Callable] = OrderedDict() # should include all functions available from agents and endogenous vars (direct evaluation and derivatives)
        self.custom_function_dict: Dict[str, Callable] = OrderedDict() # user-defined functions

        self.loss_reduction_dict: Dict[str, LossReductionMethod] = OrderedDict() # used to store all loss function label to reduction method mappings

        # label to value mapping, used to store all variable values and loss.
        self.params: Dict[str, torch.Tensor] = OrderedDict()
        self.variable_val_dict: Dict[str, torch.Tensor] = OrderedDict() # should include all local variables/params + current values, initially, all values in this dictionary can be zero
        self.loss_val_dict: Dict[str, torch.Tensor] = OrderedDict() # should include loss equation (constraints, endogenous equations, HJB equations) labels + corresponding loss values, initially, all values in this dictionary can be zero.
        self.loss_weight_dict: Dict[str, float] = OrderedDict() # should include loss equation labels + corresponding weight
        self.learnable_params = set() # add a set of strings to keep track of all learnable parameters
        self.device = "cpu"

        # for residual-based adaptive refinement (RAR) and active learning
        self.anchor_points: torch.Tensor = None
        self.refinement_rounds: int = self.config.get("refinement_rounds", 5)

        # experimental: vmap-batched (stacked) evaluation of agents/endog vars
        self.stacked: bool = self.config.get("stacked", False)
        self._stacked_evaluator = None

    '''
    Methods to define variables, equations, and constraints
    '''
    def check_name_used(self, name: str):
        for self_dicts in [self.state_variables,
                           self.agents, 
                           self.endog_vars, 
                           self.local_function_dict, 
                           self.custom_function_dict,
                           self.variable_val_dict,
                           self.loss_val_dict,
                           self.loss_weight_dict]:
            assert name not in self_dicts, f"Name: {name} is used"

    def check_label_used(self, label: str):
        for self_dicts in [self.state_variables,
                           self.agents, 
                           self.endog_vars, 
                           self.local_function_dict, 
                           self.custom_function_dict,
                           self.variable_val_dict,
                           self.loss_val_dict,
                           self.loss_weight_dict]:
            assert label not in self_dicts, f"Label: {label} is used"

    def set_state(self, names: List[str], constraints: Dict[str, List] = {}, reserve_time: bool = False):
        '''
        Set the state variables ("grid") of the problem.
        We probably want to add some constraints for each variable (domain). 
        By default, the constraints will be [-1, 1] (for easier sampling). 
        
        Only rectangular regions are supported

        When ``reserve_time`` is True (used by the time-stepping model), an extra
        ``t`` dimension is appended with domain ``[min_t, max_t]`` from the config.
        '''
        if reserve_time:
            assert "t" not in names, "t is reserved for time stepping, and should not be included in state variables"
        assert len(self.agents) + len(self.endog_vars) == 0, "Neural networks for agents and endogenous variables have been initialized. State variables cannot be changed."
        for name in names:
            self.check_name_used(name)
        self.state_variables = list(names)
        self.state_variable_constraints = {sv: [-1.0, 1.0] for sv in self.state_variables}
        self.state_variable_constraints.update(constraints)

        if reserve_time:
            self.state_variables = self.state_variables + ["t"]
            self.state_variable_constraints["t"] = [self.config["min_t"], self.config["max_t"]]

        constraints_low = []
        constraints_high = []
        
        for svc in self.state_variables:
            constraints_low.append(self.state_variable_constraints[svc][0])
            constraints_high.append(self.state_variable_constraints[svc][1])
        self.state_variable_constraints["sv_low"] = constraints_low
        self.state_variable_constraints["sv_high"] = constraints_high

        for name in self.state_variables:
            self.variable_val_dict[name] = torch.zeros((self.batch_size, 1))
        self.variable_val_dict["SV"] = torch.zeros((self.batch_size, len(self.state_variables)))

        if reserve_time:
            self.boundary_uniform_points = None

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

    def add_learnable_param(self, name: str, init_value: float=1.0):
        '''
        Add a single learnable parameter (constant in the PDE system) with name and initial value.
        '''
        self.check_name_used(name)
        self.variable_val_dict[name] = nn.Parameter(torch.tensor(init_value, dtype=torch.get_default_dtype()), requires_grad=True)
        self.learnable_params.add(name)

    def add_learnable_params(self, params: Dict[str, Any]):
        '''
        Add a dictionary of learnable parameters (constants in the PDE system) for the system, 
        each key value pair represent the name and initial value.
        '''
        for name, init_val in params.items():
            self.add_learnable_param(name, init_val)

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
            - hidden_units: **List[int]**, number of units in each layer, default: [30, 30, 30, 30]
            - output_size: **int**, number of output units, default: 1 for MLP, and last hidden unit size for KAN and MultKAN
            - layer_type: **str**, a selection from the LayerType enum, default: LayerType.MLP
            - activation_type: *str**, a selection from the ActivationType enum, default: ActivationType.Tanh
            - positive: **bool**, apply softplus to the output to be always positive if true, default: false (This has no effect for KAN.)
            - hardcode_function: a lambda function for hardcoded forwarding function, default: None
            - derivative_order: int, an additional constraint for the number of derivatives to take, so for a function with one state variable, we can still take multiple derivatives, default: number of state variables
            - batch_jac_hes: **bool**, whether to use batch jacobian or hessian for computing derivatives, default: False (When True, only name_Jac and name_Hess are included in the derivatives dictionary, and derivative_order is ignored; When False, all derivatives name_x, name_y, etc are included.)
            - input_size_sym: int, number of dimensions that should impose symmetry, for layer_type==LayerType.DeepSet
            - input_size_ext: int, number of extra dimensions, for layer_type==LayerType.DeepSet
            - hidden_units_phi: List[int], number of hidden units in phi, for layer_type==LayerType.DeepSet
            - hidden_units_rho: List[int], number of hidden units in rho, for layer_type==LayerType.DeepSet
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
        self.custom_function_dict.update({f"compute_{k}": v for k, v in new_agent.derivatives.items()})
        for func_name in new_agent.derivatives:
            self.variable_val_dict[func_name] = torch.zeros((self.batch_size, 1), device=self.device)
        self._invalidate_stacked_evaluator()
    
    def add_agents(self, names: List[str], 
                   configs: Dict[str, Dict[str, Any]]={}):
        '''
        Add multiple agents at the same time, each with different configurations.

        Config: specifies number of layers/hidden units of the neural network. See ``add_agent``.
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

        Config: specifies number of layers/hidden units of the neural network. See ``add_agent``.
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
        self.custom_function_dict.update({f"compute_{k}": v for k, v in new_endog_var.derivatives.items()})
        for func_name in new_endog_var.derivatives:
            self.variable_val_dict[func_name] = torch.zeros((self.batch_size, 1), device=self.device)
        self._invalidate_stacked_evaluator()

    def add_endogs(self, names: List[str], 
                   configs: Dict[str, Dict[str, Any]] = {}):
        '''
        Add multiple endogenous variables at the same time, each with different config.

        Config: specifies number of layers/hidden units of the neural network. See ``add_agent``.
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

    def add_equations(self, eqs: List[str]):
        for eq in eqs:
            self.add_equation(eq)

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

    def register_function(self, func: Callable):
        self.check_name_used(func.__name__)
        self.custom_function_dict[func.__name__] = func
    
    def register_functions(self, funcs: List[Callable]):
        for func in funcs:
            self.register_function(func)

    '''
    Forward evaluation / loss computation
    '''
    def _invalidate_stacked_evaluator(self):
        '''
        Drop the cached stacked evaluator so it is rebuilt lazily on the next
        forward. Must be called whenever the underlying agent/endog modules
        change (adding variables, loading a checkpoint).
        '''
        self._stacked_evaluator = None

    def _get_stacked_evaluator(self):
        if getattr(self, "_stacked_evaluator", None) is None:
            from .models import StackedFunctionEvaluator
            self._stacked_evaluator = StackedFunctionEvaluator(self.agents, self.endog_vars)
        return self._stacked_evaluator

    def _eval_local_functions(self, SV, vd):
        '''
        Evaluate all agent/endogenous-variable functions (forward values and
        their derivatives) into ``vd``.

        This is the ONLY place agent/endog networks are forwarded. By default it
        loops over ``self.local_function_dict``, calling each function
        individually. When ``self.stacked`` is enabled it batches all
        same-architecture networks into a single ``vmap`` call (experimental),
        then evaluates any remaining (non-stackable) functions individually.

        Equations, endogenous equations, constraints, HJB equations and systems
        are untouched by this method, so overriding ``update_variables`` (or this
        method) in one place changes the forward everywhere it is used.
        '''
        if not getattr(self, "stacked", False):
            for func_name in self.local_function_dict:
                vd[func_name] = self.local_function_dict[func_name](SV)
            return

        evaluator = self._get_stacked_evaluator()
        covered = evaluator.fill(SV, vd)
        for func_name in self.local_function_dict:
            if func_name in covered:
                continue
            vd[func_name] = self.local_function_dict[func_name](SV)

    def update_variables(self, SV, vd=None):
        '''
        Update the agent/endogenous variables (and their derivatives) and the
        variables defined by users through equations, at the sampled state ``SV``.

        ``vd`` is the value dictionary to populate (defaults to
        ``self.variable_val_dict``). Overriding this single method changes the
        forward computation for training, validation, refinement, plotting, etc.
        '''
        if vd is None:
            vd = self.variable_val_dict
        for i, sv_name in enumerate(self.state_variables):
            vd[sv_name] = SV[:, i:i+1]
        vd["SV"] = SV

        # properly update variables, including agent, endogenous variables, their derivatives
        self._eval_local_functions(SV, vd)

        # properly update variables, using equations
        for eq_name in self.equations:
            lhs = self.equations[eq_name].lhs.formula_str
            vd[lhs] = self.equations[eq_name].eval(self.custom_function_dict, vd)
    
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
            self.loss_val_dict[label] = self.agent_conditions[label].eval(self.local_function_dict | self.custom_function_dict, self.loss_reduction_dict[label])
        
        for label in self.endog_var_conditions:
            self.loss_val_dict[label] = self.endog_var_conditions[label].eval(self.local_function_dict | self.custom_function_dict, self.loss_reduction_dict[label])

        # for all other formula/equations, we can use the pre-computed values of a specific state to compute the loss
        for label in self.endog_equations:
            self.loss_val_dict[label] = self.endog_equations[label].eval(self.custom_function_dict, self.variable_val_dict, self.loss_reduction_dict[label])

        for label in self.constraints:
            self.loss_val_dict[label] = self.constraints[label].eval(self.custom_function_dict, self.variable_val_dict, self.loss_reduction_dict[label])

        for label in self.hjb_equations:
            self.loss_val_dict[label] = self.hjb_equations[label].eval(self.custom_function_dict, self.variable_val_dict, self.loss_reduction_dict[label])

        for label in self.systems:
            self.loss_val_dict[label] = self.systems[label].eval(self.custom_function_dict, self.variable_val_dict)

    def closure(self, SV):
        for i, sv_name in enumerate(self.state_variables):
            self.variable_val_dict[sv_name] = SV[:, i:i+1]
        self.variable_val_dict["SV"] = SV
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
            self.loss_val_dict[label] = torch.square(self.agent_conditions[label].eval(self.local_function_dict | self.custom_function_dict, LossReductionMethod.NONE))
            temp = torch.nanmean(self.loss_weight_dict[label] * self.loss_val_dict[label])
            total_loss += torch.where(temp.isnan(), 0.0, temp)
        
        for label in self.endog_var_conditions:
            self.loss_val_dict[label] = torch.square(self.endog_var_conditions[label].eval(self.local_function_dict | self.custom_function_dict, LossReductionMethod.NONE))
            temp = torch.nanmean(self.loss_weight_dict[label] * self.loss_val_dict[label])
            total_loss += torch.where(temp.isnan(), 0.0, temp)

        for label in self.endog_equations:
            self.loss_val_dict[label] = torch.square(self.endog_equations[label].eval_no_loss(self.custom_function_dict, self.variable_val_dict)).reshape((self.B, 1))
            temp = torch.nanmean(self.loss_weight_dict[label] * self.loss_val_dict[label])
            total_loss += torch.where(temp.isnan(), 0.0, temp)

        for label in self.constraints:
            self.loss_val_dict[label] = torch.square(self.constraints[label].eval_no_loss(self.custom_function_dict, self.variable_val_dict)).reshape((self.B, 1))
            temp = torch.nanmean(self.loss_weight_dict[label] * self.loss_val_dict[label])
            total_loss += torch.where(temp.isnan(), 0.0, temp)

        for label in self.hjb_equations:
            self.loss_val_dict[label] = torch.square(self.hjb_equations[label].eval_no_loss(self.custom_function_dict, self.variable_val_dict)).reshape((self.B, 1))
            temp = torch.nanmean(self.loss_weight_dict[label] * self.loss_val_dict[label])
            total_loss += torch.where(temp.isnan(), 0.0, temp)

        for label in self.systems:
            self.loss_val_dict[label] = torch.square(self.systems[label].eval_no_loss(self.custom_function_dict, self.variable_val_dict, self.B)).reshape((self.B, 1))
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

    def set_loss_reduction(self, label: str, loss_reduction: LossReductionMethod):
        if label not in self.loss_reduction_dict:
            raise ValueError(f"{label} is not a valid label for loss function")
        self.loss_reduction_dict[label] = loss_reduction

    '''
    Sampling helpers shared by both models
    '''
    def sample_uniform(self, epoch=0):
        SV = np.random.uniform(low=self.state_variable_constraints["sv_low"], 
                         high=self.state_variable_constraints["sv_high"], 
                         size=(self.batch_size, len(self.state_variables)))
        return torch.Tensor(SV).to(self.device)
    
    def sample_fixed_grid(self, epoch=0):
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

    '''
    Model setup validation, persistence and summary
    '''
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
        sv.requires_grad_(True)
        variable_val_dict_ = self.variable_val_dict.copy()
        for i, sv_name in enumerate(self.state_variables):
            variable_val_dict_[sv_name] = sv[:, i:i+1]
        variable_val_dict_["SV"] = sv

        for agent_name in self.agents:
            try:
                y = self.agents[agent_name].forward(sv)
                assert y.shape[0] == sv.shape[0] and y.shape[1] == self.agents[agent_name].config["output_size"]
            except Exception as e:
                errors.append({
                    "label": agent_name, 
                    "error": str(e)
                })

        for endog_var_name in self.endog_vars:
            try:
                y = self.endog_vars[endog_var_name].forward(sv)
                assert y.shape[0] == sv.shape[0] and y.shape[1] == self.endog_vars[endog_var_name].config["output_size"]
            except Exception as e:
                errors.append({
                    "label": endog_var_name,
                    "error": str(e),
                })
        
        self._eval_local_functions(sv, variable_val_dict_)

        for label in self.agent_conditions:
            try:
                self.agent_conditions[label].eval(self.local_function_dict | self.custom_function_dict)
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
                self.endog_var_conditions[label].eval(self.local_function_dict | self.custom_function_dict)
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
                variable_val_dict_[lhs] = self.equations[label].eval(self.custom_function_dict, variable_val_dict_)
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
                self.endog_equations[label].eval(self.custom_function_dict, variable_val_dict_)
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
                self.constraints[label].eval(self.custom_function_dict, variable_val_dict_)
            except Exception as e:
                if e is not ZeroDivisionError:
                    errors.append({
                        "label": label,
                        "parsed": self.constraints[label].lhs.formula_str + self.constraints[label].comparator + self.constraints[label].rhs.formula_str,
                        "error": str(e)
                    })

        for label in self.hjb_equations:
            try:
                self.hjb_equations[label].eval(self.custom_function_dict, variable_val_dict_)
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
                self.systems[label].eval(self.custom_function_dict, variable_val_dict_)
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

        for learn_var in self.learnable_params:
            dict_to_save[f"learn_var_{learn_var}"] = {"name": learn_var, "value": self.variable_val_dict[learn_var].tolist()}

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
            
            if k.startswith("learn_var"):
                self.learnable_params.add(v["name"])
                self.variable_val_dict[v["name"]] = nn.Parameter(torch.tensor(v["value"], dtype=torch.get_default_dtype()), requires_grad=True)

        # the underlying modules were rebuilt, so any cached stacked evaluator is stale
        self._invalidate_stacked_evaluator()
        print("Model loaded")

    '''
    Training / evaluation entry points (implemented by subclasses)
    '''
    def train_model(self, model_dir: str="./", filename: str=None, full_log=False, variables_to_track: List[str]=[]):
        raise NotImplementedError("train_model must be implemented by a subclass of BasePDEModel")

    def eval_model(self, full_log=False):
        raise NotImplementedError("eval_model must be implemented by a subclass of BasePDEModel")

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
                    param_copy[k] = v.tolist()
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
