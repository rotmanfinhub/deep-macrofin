from typing import Any, Dict, List 

import torch

from evaluations import *

class PDEModel:
    '''
    PDEModel class to assign variables, equations & constraints, etc.

    Also initialize the neural network architectures for each agent/endogenous variables 
    with some config dictionary.
    '''
    
    '''
    Methods to initialize the model, define variables, and constraints
    '''
    def __init__(self, name: str, config: Dict[str, Any], latex_var_mapping: Dict[str, str] = {}):
        '''
        Initialize a model with the provided name and config. 
        The config should include the basic training configs, 
        e.g. optimizer, step size, number of epochs.
        '''
        pass

    def set_state(self, names: List[str], constraints: Dict[str, List]):
        '''
        Set the state variables ("grid") of the problem.
        We probably want to add some constraints for each variable (domain). 
        By default, the constraints will be [-inf, inf] (no restriction). 
        '''
        pass

    def add_param(self, name, value):
        '''
        Add a single parameter with name and value.
        '''
        pass

    def add_params(self, params: Dict[str, Any]):
        '''
        Add a dictionary of parameters for the system.
        '''
        pass

    def add_agent(self, name: str, config: Dict[str, Any]):
        '''
        Add a single agent, with relevant config of neural network representation. 
        If called before states are set, should raise an error.
        '''
        pass
    
    def add_agents(self, names: List[str], configs: Dict[str, Dict[str, Any]]):
        '''
        Add multiple agents at the same time, each with different configurations.
        '''
        pass

    def add_agent_condition(self, name: str, lhs, lhs_state: Dict[str, torch.Tensor], comparator, rhs, rhs_state: Dict[str, torch.Tensor], label):
        '''
        Add boundary/initial condition for a specific agent
        '''
        pass

    def add_endog(self, name: str, config: Dict[str, Any]):
        '''
        Add a single unknown endogenous variable, with relevant config of NN. 
        If called before states are set, should raise an error.
        '''
        pass

    def add_endogs(self, names: List[str], configs: Dict[str, Dict[str, Any]]):
        '''
        Add multiple endogenous variables at the same time, each with different config.
        '''
        pass
    
    def add_endog_condition(self, name: str, lhs, lhs_state: Dict[str, torch.Tensor], comparator, rhs, rhs_state: Dict[str, torch.Tensor], label):
        '''
        Add boundary/initial condition for a specific endogenous var
        '''
        pass

    def add_equation(self, eq: str, label: str):
        '''
        Add an equation to define a new variable. 
        When label is none, the eq itself will be used as a label.
        '''
        pass

    def add_constraint(self, lhs, comparator, rhs, label):
        '''
        comparator should be one of "==", ">", ">=", "<", "<=", we can use enum for this.

        Use Constraint class to properly convert it to a loss function.
        '''
        pass

    def add_system(self, system: System):
        '''
        Decide in a later stage. 
        It should be some multiplication of loss functions 
        e.g. \prod ReLU(constraints to trigger the system) * loss induced by the system.
        '''
        pass
    
    def loss_fn(self):
        '''
        Compute the loss function, using the endogenous equation/constraints defined.
        '''
        pass

    def train_step(self, x):
        '''
        forward input, compute loss and update parameters
        '''
        pass

    def test_step(self, x):
        '''
        forward input, compute loss
        '''
        pass
    
    def train_model(self):
        '''
        The entire loop of training
        '''
        pass
    
    def eval_model(self):
        '''
        The entire loop of evaluation
        '''
        pass

    def save_model(self):
        '''
        Save all the agents, endogenous variables (pytorch model and configurations), 
        and all other configurations of the PDE model.
        '''
        pass
    
    def load_model(self, f=None, dict_to_load: Dict[str, Any]={}):
        '''
        Load all the agents, endogenous variables (pytorch model and configurations), 
        and all other configurations of the PDE model.
        '''
        pass