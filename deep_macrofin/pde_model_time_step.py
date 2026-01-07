import atexit
import gc
import json
import os
import time
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Union

import pandas as pd
import torch
from tqdm import tqdm

from .evaluations import *
from .event_handler import *
from .models import *
from .pde_model import PDEModel
from .utils import *


class PDEModelTimeStep(PDEModel):
    '''
    PDEModelTimeStep uses time stepping scheme + neural network to solve for optimality

    PDEModel class to assign variables, equations & constraints, etc.

    Also initialize the neural network architectures for each agent/endogenous variables 
    with some config dictionary.
    '''
    
    '''
    Methods to initialize the model, define variables, and constraints
    '''
    def __init__(self, name: str, 
                 config: Dict[str, Any] = DEFAULT_CONFIG_TIME_STEP, 
                 latex_var_mapping: Dict[str, str] = {}):
        '''
        Initialize a model with the provided name and config. 
        The config should include the basic training configs, 

        The optimizer is default to AdamW

        DEFAULT_CONFIG_TIME_STEP = {
            "batch_size": 100,
            "time_batch_size": None, (if None default to batch_size, if set to a number <= 1, the total batch size will be batch_size, and time steps are randomly sampled iid with the state variables)
            "num_outer_iterations": 100, # number of time stepping iterations
            "num_inner_iterations": 5000, # initial number of training epochs within each time step, it will decay with factor of sqrt{curr_outer_iteration + 1}
            "min_inner_iterations": 1000, # minimum number of training epochs within each time step, number of inner iterations will not decrease below this value
            "loss_log_interval": 50,
            "lr": 1e-3,
            "optimizer_type": OptimizerType.Adam,
            "min_t": 0.0,
            "max_t": 1.0,
            "time_boundary_loss_reduction": LossReductionMethod.MSE,
            "outer_loop_convergence_thres": 1e-4,
            "sampling_method": SamplingMethod.UniformRandom,
            "refinement_rounds": 5,
            "loss_balancing": False,
            "bernoulli_prob": 0.9999,
            "loss_balancing_temp": 0.1,
            "loss_balancing_alpha": 0.999,
        }

        latex_var_mapping should include all possible latex to python name conversions. Otherwise latex parsing will fail. Can be omitted if all the input equations/formula are not in latex form. For details, check `Formula` class defined in `evaluations/formula.py`
        '''
        self.name = name
        self.config = DEFAULT_CONFIG_TIME_STEP.copy()
        self.config.update(config)
        self.batch_size = self.config.get("batch_size", 100)
        self.time_batch_size = self.config.get("time_batch_size", None)
        if self.time_batch_size is None:
            self.time_batch_size = self.batch_size
        self.num_outer_iterations = self.config.get("num_outer_iterations", 100)
        self.num_inner_iterations = self.config.get("num_inner_iterations", 5000)
        self.min_inner_iterations = self.config.get("min_inner_iterations", 1000)
        self.loss_log_interval = self.config.get("loss_log_interval", 50)
        self.time_boundary_loss_reduction = self.config.get("time_boundary_loss_reduction", LossReductionMethod.MSE)
        self.lr = self.config.get("lr", 1e-3)
        self.optimizer_type = self.config.get("optimizer_type", OptimizerType.Adam)

        if self.config["sampling_method"] == SamplingMethod.FixedGrid:
            self.sample = self.sample_fixed_grid
            self.sample_boundary_cond = self.__sample_fixed_grid_boundary_cond
        else:
            if self.time_batch_size <= 1:
                self.sample = self.sample_uniform
            else:
                self.sample = self.sample_uniform_ts
            self.sample_boundary_cond = self.__sample_uniform_boundary_cond
            self.boundary_uniform_points = None

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
        self.custom_function_dict: Dict[str, Callable] = OrderedDict() # user-defined functions

        self.loss_reduction_dict: Dict[str, LossReductionMethod] = OrderedDict() # used to store all loss function label to reduction method mappings

        # label to value mapping, used to store all variable values and loss.
        self.params: Dict[str, torch.Tensor] = OrderedDict()
        self.variable_val_dict: Dict[str, torch.Tensor] = OrderedDict() # should include all local variables/params + current values, initially, all values in this dictionary can be zero
        self.loss_val_dict: Dict[str, torch.Tensor] = OrderedDict() # should include loss equation (constraints, endogenous equations, HJB equations) labels + corresponding loss values, initially, all values in this dictionary can be zero.
        self.loss_weight_dict: Dict[str, float] = OrderedDict() # should include loss equation labels + corresponding weight
        self.initial_guess: Dict[str, Union[float, Callable]] = OrderedDict() # should include the overrides of initial guesses for agents and endog vars
        self.learnable_params = set() # add a set of strings to keep track of all learnable parameters
        self.device = "cpu"

        # for residual-based adaptive refinement (RAR) and active learning
        self.anchor_points: torch.Tensor = None
        self.refinement_rounds: int = self.config.get("refinement_rounds", 5)
        self.OnInnerLoopStart = EventHandler()
        self.OnInnerLoopStep = EventHandler()

    def __set_agent_time_boundary_condition(self, name: str,
                            time_boundary_value: torch.Tensor,
                            weight=1.0, 
                            loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
        '''
        This is equivalent to add_agent_condition, but specifically for the implicit time boundary condition.

        The label will be initialized as f"agent_{name}_cond_time_boundary", and no overwriting can be done 
        using this function
        '''
        assert name in self.agents, f"Agent {name} does not exist"
        output_size = self.agents[name].config["output_size"]
        assert loss_reduction != LossReductionMethod.NONE, "reduction must be applied for time stepping scheme"
        assert time_boundary_value.shape == torch.Size((self.batch_size ** (len(self.state_variables) - 1), output_size)) or time_boundary_value.shape == torch.Size((self.batch_size, output_size)), "shape of boundary value does not match the state variable grid size"
        label = f"agent_{name}_cond_time_boundary"
        self.agent_conditions[label] = AgentConditions(name, 
                                                       f"{name}(SV)", {"SV": self.sample_boundary_cond(self.config["max_t"])}, 
                                                       Comparator.EQ, 
                                                       "bd_val", {"bd_val": time_boundary_value},
                                                       label, self.latex_var_mapping)
        self.loss_val_dict[label] = torch.zeros(1, device=self.device)
        self.loss_weight_dict[label] = weight
        self.loss_reduction_dict[label] = loss_reduction

    def __set_endog_time_boundary_condition(self, name: str,
                            time_boundary_value: torch.Tensor,
                            weight=1.0, 
                            loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
        '''
        This is equivalent to add_endog_condition, but specifically for the implicit time boundary condition.

        The label will be initialized as f"endogvar_{name}_cond_time_boundary", and no overwriting can be done 
        using this function
        '''
        assert name in self.endog_vars, f"Endogenous variable {name} does not exist"
        output_size = self.endog_vars[name].config["output_size"]
        assert loss_reduction != LossReductionMethod.NONE, "reduction must be applied for time stepping scheme"
        assert time_boundary_value.shape == torch.Size((self.batch_size ** (len(self.state_variables) - 1), output_size)) or time_boundary_value.shape == torch.Size((self.batch_size, output_size)), "shape of boundary value does not match the state variable grid size"
        label = f"endogvar_{name}_cond_time_boundary"
        self.endog_var_conditions[label] = EndogVarConditions(name, 
                                                       f"{name}(SV)", {"SV": self.sample_boundary_cond(self.config["max_t"])}, 
                                                       Comparator.EQ, 
                                                       "bd_val", {"bd_val": time_boundary_value},
                                                       label, self.latex_var_mapping)
        self.loss_val_dict[label] = torch.zeros(1, device=self.device)
        self.loss_weight_dict[label] = weight
        self.loss_reduction_dict[label] = loss_reduction

    def set_state(self, names: List[str], constraints: Dict[str, List] = {}):
        '''
        Set the state variables ("grid") of the problem.
        We probably want to add some constraints for each variable (domain). 
        By default, the constraints will be [-1, 1] (for easier sampling). 
        
        Only rectangular regions are supported
        '''
        assert "t" not in names, "t is reserved for time stepping, and should not be included in state variables"
        assert len(self.agents) + len(self.endog_vars) == 0, "Neural networks for agents and endogenous variables have been initialized. State variables cannot be changed."
        for name in names:
            self.check_name_used(name)
        self.state_variables = names
        self.state_variable_constraints = {sv: [-1.0, 1.0] for sv in self.state_variables}
        self.state_variable_constraints.update(constraints)

        self.state_variables += ["t"]
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
        self.boundary_uniform_points = None

    def sample_fixed_grid(self):
        '''
        Sample fixed grid of shape (B^N, 1), where B is batch size, N is number of state variables including time dimension.
        We always have at least 2 variables (one state variable, one hidden time dimension)
        '''
        sv_ls = [0] * len(self.state_variables)
        for i in range(len(self.state_variables)):
            sv_ls[i] = torch.linspace(self.state_variable_constraints["sv_low"][i], 
                                    self.state_variable_constraints["sv_high"][i], 
                                    steps=self.batch_size, device=self.device)
        return torch.cartesian_prod(*sv_ls)
    
    def sample_uniform(self):
        SV = np.random.uniform(low=self.state_variable_constraints["sv_low"], 
                         high=self.state_variable_constraints["sv_high"], 
                         size=(self.batch_size, len(self.state_variables)))
        return torch.Tensor(SV).to(self.device)
    
    def __get_refinement_loss_dict(self):
        '''
        Sample a dense subset of the problem domain, compute the loss and return total loss for each point sampled. Used for Residual-based Adaptive Refinement and Active Learning

        Returns:
            {
                "SV": sampled state variables, shape (1000, len(self.state_variables))
                "loss": total loss computed at each sv, shape (1000, 1)
            }
        '''
        # because we need a set of dense points to compute residual for adaptive sampling
        # we set all models to evaluation models so that gradients won't be computed.
        # it speeds up the computation and reduces memory usages
        self.set_all_model_eval()

        # Temporarily set a large batch size
        self.batch_size = 1000
        SV = self.sample_uniform()
        SV.requires_grad_(True)
        # make a copy of variable value mapping
        # so that we don't break the top level training routine
        variable_val_dict_ = self.variable_val_dict.copy()
        total_loss = torch.zeros((self.batch_size, 1), device=self.device)

        # forward pass
        for i, sv_name in enumerate(self.state_variables):
            variable_val_dict_[sv_name] = SV[:, i:i+1]
        variable_val_dict_["SV"] = SV

        # update variables, including agent, endogenous variables, their derivatives
        for func_name in self.local_function_dict:
            variable_val_dict_[func_name] = self.local_function_dict[func_name](SV)

        # update variables, using equations
        for eq_name in self.equations:
            lhs = self.equations[eq_name].lhs.formula_str
            variable_val_dict_[lhs] = self.equations[eq_name].eval(self.custom_function_dict, variable_val_dict_)

        # compute total losses, without reducing to a single value, keep the original dimension, but summing up using abs values
        # Note that the conditions (IC/BC, or user pre-defined sampling regions) are not considered
        # Systems are not considered
        for label in self.endog_equations:
            total_loss += torch.mean(torch.abs(self.endog_equations[label].eval_no_loss(self.custom_function_dict, variable_val_dict_)), dim=1, keepdim=True)

        for label in self.constraints:
            total_loss += torch.mean(torch.abs(self.constraints[label].eval_no_loss(self.custom_function_dict, variable_val_dict_)), dim=1, keepdim=True)

        for label in self.hjb_equations:
            total_loss += torch.mean(torch.abs(self.hjb_equations[label].eval_no_loss(self.custom_function_dict, variable_val_dict_)), dim=1, keepdim=True)

        for label in self.systems:
            total_loss += torch.mean(torch.abs(self.systems[label].eval_no_loss(self.custom_function_dict, variable_val_dict_, self.batch_size)), dim=1, keepdim=True)

        self.batch_size = self.config.get("batch_size", 100) # reset the batch size for normal computation
        self.set_all_model_training() # reset the model for training stage

        return {
            "SV": SV.detach(),
            "loss": total_loss,
        }
    
    def sample_rar_greedy(self):
        refinement_loss_dict = self.__get_refinement_loss_dict()
        SV = refinement_loss_dict["SV"]
        all_losses = refinement_loss_dict["loss"]
        X_ids = torch.topk(all_losses, self.batch_size//self.refinement_rounds, dim=0)[1].squeeze(-1)
        self.anchor_points = torch.vstack((self.anchor_points, SV[X_ids]))

    def sample_uniform_ts(self):
        SV = np.random.uniform(low=self.state_variable_constraints["sv_low"][:-1], 
                         high=self.state_variable_constraints["sv_high"][:-1], 
                         size=(self.batch_size, len(self.state_variables) - 1))
        SV = torch.Tensor(SV)
        T = torch.linspace(self.state_variable_constraints["sv_low"][-1], 
                           self.state_variable_constraints["sv_high"][-1], 
                           self.time_batch_size)

        SV_repeated = SV.repeat(1, T.shape[0]).view(-1, SV.shape[1])
        T_repeated = T.repeat(1, SV.shape[0]).view(-1, 1)
        return torch.cat((SV_repeated, T_repeated), dim=1).to(self.device)
    
    def __sample_fixed_grid_boundary_cond(self, time_val: float):
        '''
        This is only used for boundary conditions
        '''
        sv_ls = [0] * (len(self.state_variables) - 1)
        for i in range(len(self.state_variables) - 1):
            sv_ls[i] = torch.linspace(self.state_variable_constraints["sv_low"][i], 
                                    self.state_variable_constraints["sv_high"][i], 
                                    steps=self.batch_size, device=self.device)
        sv = torch.cartesian_prod(*sv_ls)
        if len(sv.shape) == 1:
            sv = sv.unsqueeze(-1)
        time_dim = torch.ones((sv.shape[0], 1), device=self.device) * time_val
        return torch.cat([sv, time_dim], dim=-1)
    
    def __sample_uniform_boundary_cond(self, time_val: float):
        if self.boundary_uniform_points is None:
            SV = np.random.uniform(low=self.state_variable_constraints["sv_low"][:-1], 
                         high=self.state_variable_constraints["sv_high"][:-1], 
                         size=(self.batch_size, len(self.state_variables) - 1))
            self.boundary_uniform_points = torch.Tensor(SV).to(self.device)
        time_dim = torch.ones((self.boundary_uniform_points.shape[0], 1), device=self.device) * time_val
        return torch.cat([self.boundary_uniform_points, time_dim], dim=-1)
    
    def __init_optimizer(self):
        all_params = []
        model_has_kan = False
        for agent_name, agent in self.agents.items():
            all_params += list(agent.parameters())
            if agent.config["layer_type"] in [LayerType.KAN, LayerType.MultKAN]:
                model_has_kan = True
        for endog_var_name, endog_var in self.endog_vars.items():
            all_params += list(endog_var.parameters())
            if endog_var.config["layer_type"] in [LayerType.KAN, LayerType.MultKAN]:
                model_has_kan = True
        
        if model_has_kan:
            # KAN can only be trained with LBFGS, 
            # as long as there is one model with KAN, we must route to the default LBFGS
            self.optimizer = LBFGS(all_params, lr=self.lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
        else:
            self.optimizer = OPTIMIZER_MAP[self.optimizer_type](all_params, self.lr)

    def __check_outer_loop_converge(self, SV_T0):
        '''
        This checks that agent(t=0) converges to agent(t=1) and endog(t=0) converges to endog(t=1) 
        '''
        temp_dict = {}
        for i, sv_name in enumerate(self.state_variables):
            temp_dict[sv_name] = SV_T0[:, i:i+1]
        temp_dict["SV"] = SV_T0

        # update variables, including agent, endogenous variables, their derivatives
        for func_name in self.local_function_dict:
            temp_dict[func_name] = self.local_function_dict[func_name](SV_T0)

        # update variables, using equations
        for eq_name in self.equations:
            lhs = self.equations[eq_name].lhs.formula_str
            temp_dict[lhs] = self.equations[eq_name].eval(self.custom_function_dict, temp_dict)

        new_vals = {}
        for k in self.prev_vals:
            new_vals[k] = temp_dict[k].detach()
        
        max_abs_change = 0.
        max_rel_change = 0.
        all_changes = {}
        for k in self.prev_vals:
            mean_new_val = torch.mean(new_vals[k]).item()
            abs_change = torch.mean(torch.abs(new_vals[k] - self.prev_vals[k])).item()
            rel_change = torch.mean(torch.abs((new_vals[k] - self.prev_vals[k]) / self.prev_vals[k])).item()
            print(f"{k}: Mean Value: {mean_new_val:.5f}, Absolute Change: {abs_change:.5f}, Relative Change: {rel_change: .5f}")
            all_changes[f"{k}_mean_val"] = mean_new_val
            all_changes[f"{k}_abs"] = abs_change
            all_changes[f"{k}_rel"] = rel_change
            max_abs_change = max(max_abs_change, abs_change)
            max_rel_change = max(max_rel_change, rel_change)

        # Update for next iteration
        for k in self.prev_vals:
            self.prev_vals[k] = new_vals[k]

        total_rel_change = min(max_abs_change, max_rel_change)
        all_changes["total"] = total_rel_change
        return all_changes
    
    def init_loss_balancing(self, *args, **kwargs):
        '''
        Initialize variables for relative loss balancing with random lookback
        https://arxiv.org/pdf/2110.09813
        '''
        self.loss_weight_log_dict = defaultdict(list)
        self.init_loss_tensor = torch.zeros(len(self.loss_val_dict), device=self.device)
        self.prev_loss_tensor = torch.zeros(len(self.loss_val_dict), device=self.device)

    def loss_balancing_step(self, *args, **kwargs):
        epoch = kwargs.get("epoch", 0)
        outer_loop_iter = kwargs.get("outer_loop_iter", 0)
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
            
            self.loss_weight_log_dict["outer_loop_iter"].append(outer_loop_iter)
            self.loss_weight_log_dict["epoch"].append(epoch)
            for k, v in self.loss_weight_dict.items():
                self.loss_weight_log_dict[k].append(v)
            
            for i, (loss_label, loss) in enumerate(self.loss_val_dict.items()):
                self.prev_loss_tensor[i] = torch.where(loss.isnan(), torch.finfo(loss.dtype).eps, loss).item()
    
    def set_initial_guess(self, initial_guess: Dict[str, Union[float, Callable]]):
        '''
        Set the initial guess (uniform value across the state variable domain) for agents or endogenous variables.

        initial_guess[agent_name] = agent_initial_guess; initial_guess[endog_var_name] = endog_var_initial_guess
        '''
        for k in initial_guess:
            assert k in self.agents or k in self.endog_vars, f"{k} is not a valid agent/endog var name"
        self.initial_guess.update(initial_guess)

    def __validation(self, SV_T0: torch.Tensor):
        self.set_all_model_eval()
        temp = self.loss_val_dict.copy()
        SV_ = SV_T0.detach().clone()
        SV_.requires_grad_(True)
        for i, sv_name in enumerate(self.state_variables):
            self.variable_val_dict[sv_name] = SV_[:, i:i+1]
        self.variable_val_dict["SV"] = SV_

        self.update_variables(SV_)
        self.loss_fn()

        loss_dict = self.loss_val_dict.copy()
        total_loss = 0
        for loss_label, loss in loss_dict.items():
            total_loss += torch.nanmean(self.loss_weight_dict[loss_label] * torch.where(loss.isnan(), 0.0, loss))
        loss_dict["total_loss"] = total_loss
        self.loss_val_dict = temp
        self.set_all_model_training()
        return loss_dict

    def train_model(self, model_dir: str="./", filename: str=None, full_log=False, variables_to_track: List[str]=[]):
        '''
        The entire loop of training

        variables_to_track: additional variables to track changes over each outer loop iteration, any endogenous variable or agent variables are not required. The variables must be included in equation definition.
        '''
        os.makedirs(model_dir, exist_ok=True)
        if self.config["sampling_method"] in [SamplingMethod.RARD, SamplingMethod.RARG]:
            os.makedirs(f"{model_dir}/anchor_points", exist_ok=True)
        if self.config.get("loss_balancing", False):
            os.makedirs(f"{model_dir}/loss_weight_logs", exist_ok=True)
        if filename is None:
            filename = self.name
        if "." in filename:
            file_prefix = filename.split(".")[0]
        else:
            file_prefix = filename
        
        log_fn = os.path.join(model_dir, f"{file_prefix}-{self.num_outer_iterations}-{self.num_inner_iterations}-log.txt")
        log_file = open(log_fn, "w", encoding="utf-8")

        @atexit.register
        def cleanup_file():
            # make sure the log file is properly closed even after exception
            log_file.close()

        print(str(self), file=log_file, flush=True)
        try:
            self.validate_model_setup(model_dir)
        except Exception as e:
            # close the file on exception. This should be the only place for it...
            log_file.close()
            raise e
        gc.collect()
        torch.cuda.empty_cache()
        print("{0:=^80}".format("Training"))
        self.set_all_model_training()
        start_time = time.time()

        SV = self.sample()
        SV.requires_grad_(True)
        for i, sv_name in enumerate(self.state_variables):
            self.variable_val_dict[sv_name] = SV[:, i:i+1]
        self.variable_val_dict["SV"] = SV

        SV_T0 = self.sample_boundary_cond(self.config["min_t"]) # This is used for time step matching
        SV_T0.requires_grad_(True)

        self.prev_vals = {}
        B = SV_T0.shape[0]
        for agent_name in self.agents:
            output_size = self.agents[agent_name].config["output_size"]
            curr_init_guess = self.initial_guess.get(agent_name, 1)
            if isinstance(curr_init_guess, Callable):
                curr_init_guess = curr_init_guess(SV_T0)
            self.prev_vals[agent_name] = torch.ones((SV_T0.shape[0], output_size), device=self.device) * curr_init_guess
        for endog_name in self.endog_vars:
            output_size = self.endog_vars[endog_name].config["output_size"]
            curr_init_guess = self.initial_guess.get(endog_name, 1)
            if isinstance(curr_init_guess, Callable):
                curr_init_guess = curr_init_guess(SV_T0)
            self.prev_vals[endog_name] = torch.ones((SV_T0.shape[0], output_size), device=self.device) * curr_init_guess
        variables_to_check_ = []
        for var in variables_to_track:
            if var in self.variable_val_dict and var not in self.prev_vals:
                self.prev_vals[var] = torch.ones_like(SV_T0[:, 0:1], device=self.device)
                variables_to_check_.append(var)

        change_dict = defaultdict(list)
        min_loss_dict = defaultdict(list)
        global_min_loss_dict = defaultdict(list)
        outer_loop_min_loss = torch.inf
        global_min_loss = torch.inf

        if self.config.get("loss_balancing", False):
            self.OnInnerLoopStart += self.init_loss_balancing
            self.OnInnerLoopStep += self.loss_balancing_step
            self.bernoulli_rho = torch.distributions.Bernoulli(self.config.get("bernoulli_prob", 0.9999))
            self.temp = self.config.get("loss_balancing_temp", 0.1)
            self.alpha = self.config.get("loss_balancing_alpha", 0.999)

        for outer_loop_iter in range(self.config["num_outer_iterations"]):
            self.anchor_points = torch.empty((0, len(self.state_variables)), device=self.device)
            set_seeds(0)
            self.__init_optimizer()
            outer_loop_start_time = time.time()
            for agent_name in self.agents:
                self.__set_agent_time_boundary_condition(agent_name, self.prev_vals[agent_name], 
                                                         loss_reduction=self.time_boundary_loss_reduction)
            for endog_name in self.endog_vars:
                self.__set_endog_time_boundary_condition(endog_name, self.prev_vals[endog_name], 
                                                         loss_reduction=self.time_boundary_loss_reduction)

            # ensure random exploration of the space if uniform random is used
            set_seeds(outer_loop_iter)
            SV = self.sample()
            SV.requires_grad_(True)
            for i, sv_name in enumerate(self.state_variables):
                self.variable_val_dict[sv_name] = SV[:, i:i+1]
            self.variable_val_dict["SV"] = SV
            validation_SV = self.sample()
            validation_SV.requires_grad_(True)

            self.OnInnerLoopStart()
            set_seeds(0)
            min_loss = torch.inf
            num_inner_iters: int = max(int(self.num_inner_iterations / (np.sqrt(outer_loop_iter + 1))), self.min_inner_iterations)
            pbar = tqdm(range(num_inner_iters), dynamic_ncols=True)
            for epoch in pbar:
                self.optimizer.step(lambda: self.closure(SV))
                # within each time step, 
                # evaluate every log interval time and also evaluate at the end of the final epoch

                if epoch % self.loss_log_interval == 0 or epoch == self.num_inner_iterations - 1:
                    loss_dict = self.__validation(validation_SV)
                    curr_loss = loss_dict["total_loss"].item()
                    if curr_loss < min_loss and all(not v.isnan() for v in loss_dict.values()):
                        min_loss = curr_loss
                        self.save_model(model_dir, f"{file_prefix}_temp_best.pt")
                        min_loss_dict["time_loop_iter"].append(outer_loop_iter)
                        min_loss_dict["epoch"].append(len(min_loss_dict["epoch"]))
                        for k, v in loss_dict.items():
                            min_loss_dict[k].append(v.item())
                        pbar.set_description("Min loss: {0:.4f}".format(min_loss))
                    if curr_loss < global_min_loss and all(not v.isnan() for v in loss_dict.values()):
                        global_min_loss = curr_loss
                        global_min_loss_dict["time_loop_iter"].append(outer_loop_iter)
                        global_min_loss_dict["epoch"].append(len(global_min_loss_dict["epoch"]))
                        for k, v in loss_dict.items():
                            global_min_loss_dict[k].append(v.item())

                if self.config["sampling_method"] == SamplingMethod.RARG and epoch % (num_inner_iters // self.refinement_rounds) == 0 and epoch > 0:
                    self.sample_rar_greedy()
                    SV = torch.vstack([SV, self.anchor_points])
                    SV.requires_grad_(True)
                    for i, sv_name in enumerate(self.state_variables):
                        self.variable_val_dict[sv_name] = SV[:, i:i+1]
                    self.variable_val_dict["SV"] = SV
                self.OnInnerLoopStep(epoch=epoch, outer_loop_iter=outer_loop_iter)
                torch.cuda.empty_cache()

            outer_loop_finish_time = time.time()
            dict_to_load = torch.load(f=f"{model_dir}/{file_prefix}_temp_best.pt", map_location=self.device, weights_only=False)
            self.load_model(dict_to_load)
            all_changes = self.__check_outer_loop_converge(SV_T0)
            torch.cuda.empty_cache()
            if self.anchor_points is not None and len(self.anchor_points) > 0:
                anchor_points_np = self.anchor_points.detach().cpu().numpy()
                np.save(os.path.join(model_dir, "anchor_points", f"{file_prefix}_anchor_points_{outer_loop_iter}.npy"), anchor_points_np)
            if hasattr(self, "loss_weight_log_dict"):
                pd.DataFrame(self.loss_weight_log_dict).to_csv(f"{model_dir}/loss_weight_logs/{file_prefix}_loss_weight_{outer_loop_iter}.csv", index=False)
                for k in self.loss_weight_log_dict:
                    self.loss_weight_log_dict[k].clear()
                del self.loss_weight_log_dict
                self.loss_weight_log_dict = None
                self.init_loss_tensor = torch.fill(self.init_loss_tensor, 0.0) 
                self.prev_loss_tensor = torch.fill(self.prev_loss_tensor, 0.0) # if not fill 0, seems to crash the program
            loss_dict = self.__validation(SV_T0) # keep track of the loss at minimum time step only
            total_loss = loss_dict["total_loss"]
            if total_loss < outer_loop_min_loss:
                print(f"Updating min loss from {outer_loop_min_loss:.4f} to {total_loss:.4f}")
                outer_loop_min_loss = total_loss
                loss_dict = self.loss_val_dict.copy()
                loss_dict["total_loss"] = total_loss
                self.save_model(model_dir, f"{file_prefix}_best.pt")

            print(f"Outer Loop {outer_loop_iter} Finished in {outer_loop_finish_time - outer_loop_start_time:.4f}s. Loading best model...")

            print(f"Outer Loop {outer_loop_iter} Finished in {outer_loop_finish_time - outer_loop_start_time:.4f}s. Final Result:", file=log_file)
            for k in self.agents:
                mean_new_val = all_changes[f"{k}_mean_val"]
                abs_change = all_changes[f"{k}_abs"]
                rel_change = all_changes[f"{k}_rel"]
                print(f"{k}: Mean Value: {mean_new_val:.5f}, Absolute Change: {abs_change:.5f}, Relative Change: {rel_change: .5f}", file=log_file)
            for k in self.endog_vars:
                mean_new_val = all_changes[f"{k}_mean_val"]
                abs_change = all_changes[f"{k}_abs"]
                rel_change = all_changes[f"{k}_rel"]
                print(f"{k}: Mean Value: {mean_new_val:.5f}, Absolute Change: {abs_change:.5f}, Relative Change: {rel_change: .5f}", file=log_file)
            for k in variables_to_check_:
                mean_new_val = all_changes[f"{k}_mean_val"]
                abs_change = all_changes[f"{k}_abs"]
                rel_change = all_changes[f"{k}_rel"]
                print(f"{k}: Mean Value: {mean_new_val:.5f}, Absolute Change: {abs_change:.5f}, Relative Change: {rel_change: .5f}", file=log_file)
            log_file.flush()
            change_dict["outer_loop_iter"].append(outer_loop_iter)
            for k, v in all_changes.items():
                change_dict[k].append(v)
            
            if all_changes["total"] < self.config["outer_loop_convergence_thres"]:
                break
        print(f"training finished, total time :: {time.time() - start_time}")
        print(f"training finished, total time :: {time.time() - start_time}", file=log_file)
        log_file.close()
        if loss_dict["total_loss"].item() < min_loss and all(not v.isnan() for v in loss_dict.values()):
            self.save_model(model_dir, f"{file_prefix}_best.pt")
        print(f"Best model saved to {model_dir}/{file_prefix}_best.pt if valid")
        self.save_model(model_dir, filename, verbose=True)
        pd.DataFrame(min_loss_dict).to_csv(f"{model_dir}/{file_prefix}_min_loss.csv", index=False)
        pd.DataFrame(global_min_loss_dict).to_csv(f"{model_dir}/{file_prefix}_global_min_loss.csv", index=False)
        pd.DataFrame(change_dict).to_csv(f"{model_dir}/{file_prefix}_change_dict.csv", index=False)
        return loss_dict
    
    def eval_model(self, full_log=False):
        '''
        The entire loop of evaluation
        '''
        self.validate_model_setup()
        self.set_all_model_eval()
        print("{0:=^80}".format("Evaluating"))
        SV = self.sample()
        SV.requires_grad_(True)
        for i, sv_name in enumerate(self.state_variables):
            self.variable_val_dict[sv_name] = SV[:, i:i+1]
        self.variable_val_dict["SV"] = SV
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
        sv = self.sample()
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
        
        for func_name in self.local_function_dict:
            variable_val_dict_[func_name] = self.local_function_dict[func_name](sv)

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
    
    def plot_vars(self, vars_to_plot: List[str], ncols: int=4, elev=30, azim=-135, roll=0):
        '''
        Inputs:
            vars_to_plot: variable names to plot, can be an equation defining a new variable. If Latex, need to be enclosed by $$ symbols
            ncols: number of columns to plot, default: 4
        This function is only supported for 1D or 2D state_variables.
        '''
        assert len(self.state_variables) <= 3, "Plot is only supported for problems with no more than 2 state variables"

        variable_var_dict_ = self.variable_val_dict.copy()
        var_to_latex = {}
        for k, v in self.latex_var_mapping.items():
            var_to_latex[v] = k

        sv_ls = [0] * (len(self.state_variables) - 1)
        for i in range(len(self.state_variables) - 1):
            sv_ls[i] = torch.linspace(self.state_variable_constraints["sv_low"][i], 
                                    self.state_variable_constraints["sv_high"][i], 
                                    steps=100, device=self.device)
        sv = torch.cartesian_prod(*sv_ls)
        if len(sv.shape) == 1:
            sv = sv.unsqueeze(-1)
        time_dim = torch.zeros((sv.shape[0], 1), device=self.device)
        X = torch.cat([sv, time_dim], dim=-1)
        
        nrows = len(vars_to_plot) // ncols
        if len(vars_to_plot) % ncols > 0:
            nrows += 1
        if len(self.state_variables) - 1 == 1:
            fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
            SV = torch.clone(X)
            SV.requires_grad_(True)
            X = X.detach().cpu().numpy()[:, :1].reshape(-1)
            for i, sv_name in enumerate(self.state_variables):
                variable_var_dict_[sv_name] = SV[:, i:i+1]
            variable_var_dict_["SV"] = SV
            # properly update variables, including agent, endogenous variables, their derivatives
            for func_name in self.local_function_dict:
                variable_var_dict_[func_name] = self.local_function_dict[func_name](SV)

            # properly update variables, using equations
            for eq_name in self.equations:
                lhs = self.equations[eq_name].lhs.formula_str
                variable_var_dict_[lhs] = self.equations[eq_name].eval(self.custom_function_dict, variable_var_dict_)

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
                        variable_var_dict_[lhs] = curr_eq.eval(self.custom_function_dict, variable_var_dict_)
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
                        variable_var_dict_[lhs] = curr_eq.eval(self.custom_function_dict, variable_var_dict_)
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
            X, Y = torch.meshgrid(sv_ls[:2], indexing="ij")
            X = X.detach().cpu().numpy()
            Y = Y.detach().cpu().numpy()
            SV.requires_grad_(True)
            for i, sv_name in enumerate(self.state_variables):
                variable_var_dict_[sv_name] = SV[:, i:i+1]
            variable_var_dict_["SV"] = SV
            # properly update variables, including agent, endogenous variables, their derivatives
            for func_name in self.local_function_dict:
                variable_var_dict_[func_name] = self.local_function_dict[func_name](SV)

            # properly update variables, using equations
            for eq_name in self.equations:
                lhs = self.equations[eq_name].lhs.formula_str
                variable_var_dict_[lhs] = self.equations[eq_name].eval(self.custom_function_dict, variable_var_dict_)

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
                        variable_var_dict_[lhs] = curr_eq.eval(self.custom_function_dict, variable_var_dict_)
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
                        variable_var_dict_[lhs] = curr_eq.eval(self.custom_function_dict, variable_var_dict_)
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