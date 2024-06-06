import json
import os
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from .evaluations import *
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

        The optimizer is default to Adam

        DEFAULT_CONFIG={
            "batch_size": 100,
            "num_epochs": 1000,
            "lr": 1e-3
        }

        latex_var_mapping should include all possible latex to python name conversions. Otherwise latex parsing will fail. Can be omitted if all the input equations/formula are not in latex form. For details, check `Formula` class defined in `evaluations/formula.py`
        '''
        self.name = name
        self.config = config
        self.batch_size = config.get("batch_size", 100)
        self.num_epochs = config.get("num_epochs", 500)
        self.lr = config.get("lr", 1e-3)
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

        # label to value mapping, used to store all variable values and loss.
        self.variable_val_dict: Dict[str, torch.Tensor] = OrderedDict() # should include all local variables/params + current values, initially, all values in this dictionary can be zero
        self.loss_val_dict: Dict[str, torch.Tensor] = OrderedDict() # should include loss equation (constraints, endogenous equations, HJB equations) labels + corresponding loss values, initially, all values in this dictionary can be zero.
        self.loss_weight_dict: Dict[str, float] = OrderedDict() # should include loss equation labels + corresponding weight
        self.device = "cpu"

    def check_name_used(self, name):
        for self_dicts in [self.state_variables,
                           self.agents, 
                           self.endog_vars, 
                           self.local_function_dict, 
                           self.variable_val_dict,
                           self.loss_val_dict,
                           self.loss_weight_dict]:
            assert name not in self_dicts, f"Name: {name} is used"

    def check_label_used(self, label):
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
        
        for svc in self.state_variable_constraints:
            constraints_low.append(self.state_variable_constraints[svc][0])
            constraints_high.append(self.state_variable_constraints[svc][1])
        self.state_variable_constraints["sv_low"] = constraints_low
        self.state_variable_constraints["sv_high"] = constraints_high

        for name in self.state_variables:
            self.variable_val_dict[name] = torch.zeros((self.batch_size, 1))

    def add_param(self, name, value):
        '''
        Add a single parameter (constant in the PDE system) with name and value.
        '''
        self.check_name_used(name)
        self.variable_val_dict[name] = value

    def add_params(self, params: Dict[str, Any]):
        '''
        Add a dictionary of parameters (constants in the PDE system) for the system.
        '''
        for name in params:
            self.check_name_used(name)
        self.variable_val_dict.update(params)

    def add_agent(self, name: str, 
                  config: Dict[str, Any] = DEFAULT_LEARNABLE_VAR_CONFIG,
                  overwrite=False):
        '''
        Add a single agent, with relevant config of neural network representation. 
        If called before states are set, should raise an error.

        Input:
        - overwrite: overwrite the previous agent with the same name, used for loading, default: False
        '''
        assert len(self.state_variables) > 0, "Please set the state variables first"
        if not overwrite:
            self.check_name_used(name)
        agent_config = deepcopy(DEFAULT_LEARNABLE_VAR_CONFIG)
        agent_config.update(config)

        self.device = config["device"]

        new_agent = Agent(name, self.state_variables, agent_config)
        self.agents[name] = new_agent
        self.local_function_dict.update(new_agent.derivatives)
        for func_name in new_agent.derivatives:
            self.variable_val_dict[func_name] = torch.zeros((self.batch_size, 1), device=self.device)
    
    def add_agents(self, names: List[str], 
                   configs: Dict[str, Dict[str, Any]]):
        '''
        Add multiple agents at the same time, each with different configurations.
        '''
        assert len(self.state_variables) > 0, "Please set the state variables first"
        for name in names:
            self.add_agent(name, configs.get(name, DEFAULT_LEARNABLE_VAR_CONFIG))

    def add_agent_condition(self, name: str, 
                            lhs: str, lhs_state: Dict[str, torch.Tensor], 
                            comparator: Comparator, 
                            rhs: str, rhs_state: Dict[str, torch.Tensor], 
                            label: str,
                            weight: float=1.0):
        '''
        Add boundary/initial condition for a specific agent

        Input:
        - name: agent name, 
        - lhs: the string expression for lhs formula, latex expression not supported, should be functions of specific format agent_name(SV), or simply a constant value
        - lhs_state: the specific value of SV to evaluate lhs at for the agent/endogenous variable
        - comparator: 
        - rhs: the string expression for lhs formula, latex expression not supported, should be functions of specific format agent_name(SV), or simply a constant value
        - rhs_state: the specific value of SV to evaluate rhs at for the agent/endogenous variable, if rhs is a constant, this can be an empty dictionary
        - label: label for the condition
        '''
        assert name in self.agents, f"Agent {name} does not exist"
        label = f"agent_{name}_cond_{label}"
        self.check_label_used(label)
        self.agent_conditions[label] = AgentConditions(name, 
                                                       lhs, lhs_state, 
                                                       comparator, 
                                                       rhs, rhs_state,
                                                       label, self.latex_var_mapping)
        self.loss_val_dict[label] = torch.zeros(1, device=self.device)
        self.loss_weight_dict[label] = weight

    def add_endog(self, name: str, 
                  config: Dict[str, Any] = DEFAULT_LEARNABLE_VAR_CONFIG,
                  overwrite=False):
        '''
        Add a single unknown endogenous variable, with relevant config of NN. 
        If called before states are set, should raise an error.

        Input:
        - overwrite: overwrite the previous agent with the same name, used for loading, default: False
        '''
        assert len(self.state_variables) > 0, "Please set the state variables first"
        if not overwrite:
            self.check_name_used(name)
        endog_var_config = deepcopy(DEFAULT_LEARNABLE_VAR_CONFIG)
        endog_var_config.update(config)

        self.device = config["device"]

        new_endog_var = EndogVar(name, self.state_variables, endog_var_config)
        self.endog_vars[name] = new_endog_var
        self.local_function_dict.update(new_endog_var.derivatives)
        for func_name in new_endog_var.derivatives:
            self.variable_val_dict[func_name] = torch.zeros((self.batch_size, 1), device=self.device)

    def add_endogs(self, names: List[str], 
                   configs: Dict[str, Dict[str, Any]]):
        '''
        Add multiple endogenous variables at the same time, each with different config.
        '''
        assert len(self.state_variables) > 0, "Please set the state variables first"
        for name in names:
            self.add_endog(name, configs.get(name, DEFAULT_LEARNABLE_VAR_CONFIG))
    
    def add_endog_condition(self, name: str, 
                            lhs: str, lhs_state: Dict[str, torch.Tensor], 
                            comparator: Comparator, 
                            rhs: str, rhs_state: Dict[str, torch.Tensor], 
                            label: str,
                            weight=1.0):
        '''
        Add boundary/initial condition for a specific endogenous var

        Input:
        - name: endogenous variable name
        - lhs: the string expression for lhs formula, latex expression not supported, should be functions of specific format endog_name(SV), or simply a constant value
        - lhs_state: the specific value of SV to evaluate lhs at for the agent/endogenous variable
        - comparator: 
        - rhs: the string expression for lhs formula, latex expression not supported, should be functions of specific format endog_name(SV), or simply a constant value
        - rhs_state: the specific value of SV to evaluate rhs at for the agent/endogenous variable, if rhs is a constant, this can be an empty dictionary
        - label: label for the condition
        '''
        assert name in self.endog_vars, f"Endogenous variable {name} does not exist"
        label = f"endogvar_{name}_cond_{label}"
        self.check_label_used(label)
        self.endog_var_conditions[label] = EndogVarConditions(name, 
                                                       lhs, lhs_state, 
                                                       comparator, 
                                                       rhs, rhs_state,
                                                       label, self.latex_var_mapping)
        self.loss_val_dict[label] = torch.zeros(1, device=self.device)
        self.loss_weight_dict[label] = weight

    def add_equation(self, eq: str, label: str=None):
        '''
        Add an equation to define a new variable. 
        '''
        pass

    def add_endog_equation(self, eq: str, label: str=None, weight=1.0):
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

    def add_constraint(self, lhs, comparator: Comparator, rhs, label=None, weight=1.0):
        '''
        comparator should be one of "=", ">", ">=", "<", "<=", we can use enum for this.

        Use Constraint class to properly convert it to a loss function.
        '''
        pass

    def add_system(self, system: System, label=None, weight=1.0):
        '''
        Decide in a later stage. 
        It should be some multiplication of loss functions 
        e.g. \prod ReLU(constraints to trigger the system) * loss induced by the system.
        '''
        pass
    
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
            self.loss_val_dict[label] = self.agent_conditions[label].eval(self.local_function_dict)
        
        for label in self.endog_var_conditions:
            self.loss_val_dict[label] = self.endog_var_conditions[label].eval(self.local_function_dict)

        # for all other formula/equations, we can use the pre-computed values of a specific state to compute the loss
        for label in self.endog_equations:
            self.loss_val_dict[label] = self.endog_equations[label].eval({}, self.variable_val_dict)

        for label in self.constraints:
            self.loss_val_dict[label] = self.constraints[label].eval({}, self.variable_val_dict)

        for label in self.hjb_equations:
            self.loss_val_dict[label] = self.hjb_equations[label].eval({}, self.variable_val_dict)

        for label in self.systems:
            self.loss_val_dict[label] = self.systems[label].eval({}, self.variable_val_dict)
        

    def train_step(self):
        '''
        initialize random state variable with proper constraints, compute loss and update parameters
        '''
        self.optimizer.zero_grad()
        SV = np.random.uniform(low=self.state_variable_constraints["sv_low"], 
                         high=self.state_variable_constraints["sv_high"], 
                         size=(self.batch_size, len(self.state_variables)))
        SV = torch.Tensor(SV)
        for i, sv_name in enumerate(self.state_variables):
            self.variable_val_dict[sv_name] = SV[:, i:i+1]

        # properly update variables, including agent, endogenous variables, their derivatives
        for func_name in self.local_function_dict:
            self.variable_val_dict[func_name] = self.local_function_dict[func_name](SV)

        # properly update variables, using equations
        for eq_name in self.equations:
            lhs = self.equations[eq_name].lhs
            self.variable_val_dict[lhs] = self.equations[eq_name].eval({}, self.variable_val_dict)

        self.loss_fn()
        total_loss = 0
        for loss_label, loss in self.loss_val_dict.items():
            total_loss += self.loss_weight_dict[loss_label] * loss
        
        total_loss.backward()
        self.optimizer.step()

        loss_dict = self.loss_val_dict.copy()
        loss_dict["total_loss"] = total_loss
        return loss_dict
        

    def test_step(self):
        '''
        initialize random state variable with proper constraints, compute loss
        '''
        SV = np.random.uniform(low=self.state_variable_constraints["sv_low"], 
                         high=self.state_variable_constraints["sv_high"], 
                         size=(self.batch_size, len(self.state_variables)))
        SV = torch.Tensor(SV)
        for i, sv_name in enumerate(self.state_variables):
            self.variable_val_dict[sv_name] = SV[:, i:i+1]

        # properly update variables, including agent, endogenous variables, their derivatives
        for func_name in self.local_function_dict:
            self.variable_val_dict[func_name] = self.local_function_dict[func_name](SV)

        # properly update variables, using equations
        for eq_name in self.equations:
            lhs = self.equations[eq_name].lhs
            self.variable_val_dict[lhs] = self.equations[eq_name].eval({}, self.variable_val_dict)

        self.loss_fn()
        total_loss = 0
        for loss_label, loss in self.loss_val_dict.items():
            total_loss += self.loss_weight_dict[loss_label] * loss

        loss_dict = self.loss_val_dict.copy()
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
    
    def train_model(self, model_dir="./", filename=None, full_log=False):
        '''
        The entire loop of training
        '''

        all_params = []
        for agent_name, agent in self.agents.items():
            all_params += list(agent.parameters())
        for endog_var_name, endog_var in self.endog_vars.items():
            all_params += list(endog_var.parameters())
        
        self.optimizer = torch.optim.AdamW(all_params, self.lr)

        os.makedirs(model_dir, exist_ok=True)
        if filename is None:
            filename = filename = self.name
        if "." in filename:
            file_prefix = filename.split(".")[0]
        else:
            file_prefix = filename
        
        log_fn = os.path.join(model_dir, f"{file_prefix}-{self.num_epochs}-log.txt")
        log_file = open(log_fn, "w", encoding="utf-8")
        print(str(self), file=log_file)
        self.validate_model_setup(model_dir)
        print("{0:=^80}".format("Training"))
        self.set_all_model_training()
        start_time = time.time()
        set_seeds(0)
        pbar = tqdm(range(self.num_epochs))
        for epoch in pbar:
            epoch_start_time = time.time()
            
            loss_dict = self.train_step()

            if full_log:
                formatted_train_loss = ",\n".join([f'{k}: {v:.4f}' for k, v in loss_dict.items()])
            else:
                formatted_train_loss = "%.4f" % loss_dict["total_loss"]
            # print(f"epoch {epoch}: \ntrain loss :: {formatted_train_loss},\ntime elapsed :: {time.time() - epoch_start_time}")
            if epoch % 100 == 0:
                pbar.set_description("Total loss: {0:.4f}".format(loss_dict["total_loss"]))
            print(f"epoch {epoch}: \ntrain loss :: {formatted_train_loss},\ntime elapsed :: {time.time() - epoch_start_time}", file=log_file)
        print(f"training finished, total time :: {time.time() - start_time}")
        print(f"training finished, total time :: {time.time() - start_time}", file=log_file)
        log_file.close()
        self.save_model(model_dir, filename)

        return loss_dict
    
    def eval_model(self, full_log=False):
        '''
        The entire loop of evaluation
        '''
        self.validate_model_setup()
        self.set_all_model_eval()
        print("{0:=^80}".format("Evaluating"))
        loss_dict = self.test_step()

        if full_log:
            formatted_loss = ",\n".join([f'{k}: {v:.4f}' for k, v in loss_dict.items()])
        else:
            formatted_loss = "%.4f" % loss_dict["total_loss"]
        print(f"loss :: {formatted_loss}")
        return loss_dict

    def train_model_kan(self, model_dir="./", filename=None, full_log=False):
        '''
        Currently, I don't want to give too many configurations for KAN as an initial testing step
        '''
        all_params = []
        for agent_name, agent in self.agents.items():
            all_params += list(agent.parameters())
        for endog_var_name, endog_var in self.endog_vars.items():
            all_params += list(endog_var.parameters())

        self.optimizer = LBFGS(all_params, lr=self.lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

        os.makedirs(model_dir, exist_ok=True)
        if filename is None:
            filename = filename = self.name
        if "." in filename:
            file_prefix = filename.split(".")[0]
        else:
            file_prefix = filename
        
        log_fn = os.path.join(model_dir, f"{file_prefix}-{self.num_epochs}-log.txt")
        log_file = open(log_fn, "w", encoding="utf-8")
        print(str(self), file=log_file)
        self.validate_model_setup(model_dir)
        print("{0:=^80}".format("Training"))
        self.set_all_model_training()
        start_time = time.time()
        set_seeds(0)
        for epoch in tqdm(range(self.num_epochs)):
            epoch_start_time = time.time()

            def closure(model: PDEModel):
                model.optimizer.zero_grad()
                SV = np.random.uniform(low=model.state_variable_constraints["sv_low"], 
                         high=model.state_variable_constraints["sv_high"], 
                         size=(model.batch_size, len(model.state_variables)))
                SV = torch.Tensor(SV)
                for i, sv_name in enumerate(model.state_variables):
                    model.variable_val_dict[sv_name] = SV[:, i:i+1]

                # properly update variables, including agent, endogenous variables, their derivatives
                for func_name in model.local_function_dict:
                    model.variable_val_dict[func_name] = model.local_function_dict[func_name](SV)

                # properly update variables, using equations
                for eq_name in model.equations:
                    lhs = model.equations[eq_name].lhs
                    model.variable_val_dict[lhs] = model.equations[eq_name].eval({}, model.variable_val_dict)

                model.loss_fn()
                total_loss = 0
                for loss_label, loss in model.loss_val_dict.items():
                    total_loss += model.loss_weight_dict[loss_label] * loss
                
                total_loss.backward()
                return total_loss

            self.optimizer.step(lambda : closure(self))

            loss_dict = self.loss_val_dict.copy()
            total_loss = 0
            for loss_label, loss in loss_dict.items():
                total_loss += self.loss_weight_dict[loss_label] * loss
            loss_dict["total_loss"] = total_loss

            if full_log:
                formatted_train_loss = ",\n".join([f'{k}: {v:.4f}' for k, v in loss_dict.items()])
            else:
                formatted_train_loss = "%.4f" % loss_dict["total_loss"]
            print(f"epoch {epoch}: \ntrain loss :: {formatted_train_loss},\ntime elapsed :: {time.time() - epoch_start_time}", file=log_file)
        print(f"training finished, total time :: {time.time() - start_time}")
        print(f"training finished, total time :: {time.time() - start_time}", file=log_file)
        log_file.close()
        self.save_model(model_dir, filename)
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
        sv = torch.rand((self.batch_size, len(self.state_variables)), device=self.device)
        for i, sv_name in enumerate(self.state_variables):
            self.variable_val_dict[sv_name] = sv[:, i:i+1]

        for agent_name in self.agents:
            try:
                y = self.agents[agent_name].forward(sv)
                assert y.shape[0] == self.batch_size and y.shape[1] == 1
            except Exception as e:
                errors.append([agent_name, str(e)])

        for endog_var_name in self.endog_vars:
            try:
                y = self.endog_vars[endog_var_name].forward(sv)
                assert y.shape[0] == self.batch_size and y.shape[1] == 1
            except Exception as e:
                errors.append([endog_var_name, str(e)])

        for label in self.agent_conditions:
            try:
                self.agent_conditions[label].eval(self.local_function_dict)
            except Exception as e:
                if e is not ZeroDivisionError:
                    # it's fine to have zero division. All other errors should be raised
                    errors.append([label, str(e) + " Please use SV as the hard coded state variable inputs, in lhs or rhs"])
        
        for label in self.endog_var_conditions:
            try:
                self.endog_var_conditions[label].eval(self.local_function_dict)
            except Exception as e:
                if e is not ZeroDivisionError:
                    errors.append([label, str(e) + " Please use SV as the hard coded state variable inputs, in lhs or rhs"])
        
        for label in self.equations:
            try:
                self.endog_equations[label].eval({}, self.variable_val_dict)
            except Exception as e:
                if e is not ZeroDivisionError:
                    errors.append([label, str(e)])


        for label in self.endog_equations:
            try:
                self.endog_equations[label].eval({}, self.variable_val_dict)
            except Exception as e:
                if e is not ZeroDivisionError:
                    errors.append([label, str(e)])

        for label in self.constraints:
            try:
                self.constraints[label].eval({}, self.variable_val_dict)
            except Exception as e:
                if e is not ZeroDivisionError:
                    errors.append([label, str(e)])

        for label in self.hjb_equations:
            try:
                self.hjb_equations[label].eval({}, self.variable_val_dict)
            except Exception as e:
                if e is not ZeroDivisionError:
                    errors.append([label, str(e)])

        for label in self.systems:
            try:
                self.systems[label].eval({}, self.variable_val_dict)
            except Exception as e:
                if e is not ZeroDivisionError:
                    # it's fine to have zero division. All other errors should be raised
                    errors.append([label, str(e)])

        if len(errors) > 0:
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, f"{self.name}-errors.txt"), "w", encoding="utf-8") as f:
                f.write("Error Log:\n")
                f.writelines([f"{e[0]} : {e[1]}" for e in errors])
            print(json.dumps(errors, indent=True))
            raise Exception(f"Errors when validating model setup, please check {self.name}-errors.txt for details.")

    def save_model(self, model_dir: str = "./", filename: str=None):
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
        str_repr += "Latex Variable Mapping: " + json.dumps(self.latex_var_mapping, indent=True) + "\n\n"

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

