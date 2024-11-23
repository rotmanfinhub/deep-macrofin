# deep_macrofin.pde_model

This is the main interface to construct the PDE system to solve.

## PDEModel
```py
class PDEModel(name: str, 
                config: Dict[str, Any] = DEFAULT_CONFIG, 
                latex_var_mapping: Dict[str, str] = {})
```

Initialize a PDEModel with the provided name and config. 

**Parameters**:

- name: **str**, name of the PDE model.
- config: **Dict[str, Any]**, defines the training configuration. Check [Constants](utils.md#constants) for default values.
    - batch_size: **int**
    - num_epochs: **int**
    - lr: **float**, learning rate for optimizer
    - loss_log_interval: **int**, the interval at which loss should be reported/recorded
    - optimizer_type: OptimizerType.Adam, OptimizerType.AdamW or OptimizerType.LBFGS
    - sampling_method: SamplingMethod.UniformRandom, SamplingMethod.FixedGrid, SamplingMethod.ActiveLearning
    - refinement_sample_interval: **int**,
    - loss_balancing: **bool**, use Relative Loss Balancing with Random Lookback (ReLoBRaLo) for loss weight update
    - bernoulli_prob: **float**, parameter for loss balancing 
    - loss_balancing_temp: **float**, parameter for loss balancing
    - loss_balancing_alpha: **float**, parameter for loss balancing
    - soft_adapt_interval: **int**, if larger than 0, use soft adapt for loss weight update, and the value is set to be the look-back interval.
    - loss_soft_attention: **bool**, use soft attention for grid-wise loss weight updates.
- latex_var_mapping: **Dict[str, str]**, it should include all possible latex to python name conversions. Otherwise latex parsing will fail. Can be omitted if all the input equations/formula are not in latex form. For details, check [`Formula`](evaluations.md#formula).


### set_state
```py
def set_state(self, names: List[str], constraints: Dict[str, List] = {})
```
Set the state variables ("grid") of the problem. By default, the constraints will be [-1, 1] (for easier sampling). Only rectangular regions are supported. Once an agent or endogenous variable has been added, calling set_state will raise an error.

### set_state_constraints
```py
def set_state_constraints(self, constraints: Dict[str, List] = {})
```
Overwrite the constraints for state variables, without changing the number of state variables. This can be used after adding an agent or endogenous variable and after loading a pre-trained model.

### add_param
```py
def add_param(self, name: str, value: torch.Tensor)
```
Add a single parameter (constant in the PDE system) with name and value.

### add_params
```py
def add_params(self, params: Dict[str, Any])
```
Add a dictionary of parameters (constants in the PDE system) for the system.

### add_agent
```py
def add_agent(self, name: str, 
            config: Dict[str, Any] = DEFAULT_LEARNABLE_VAR_CONFIG,
            overwrite=False)
```

Add a single [Agent](models.md#agent), with relevant config of neural network representation. If called before states are set, should raise an error.

**Parameters**:

- name: unique identifier of agent.
- Config: specifies number of layers/hidden units of the neural network. Check [Constants](utils.md#constants) for default values.
- overwrite: overwrite the previous agent with the same name, used for loading, default: False

### add_agents
```py
def add_agents(self, names: List[str], 
               configs: Dict[str, Dict[str, Any]]={})
```
Add multiple [Agents](models.md#agent) at the same time, each with different configurations.

### add_agent_condition
```py
def add_agent_condition(self, name: str, 
                    lhs: str, lhs_state: Dict[str, torch.Tensor], 
                    comparator: Comparator, 
                    rhs: str, rhs_state: Dict[str, torch.Tensor], 
                    label: str=None,
                    weight: float=1.0, 
                    loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
```
Add boundary/initial condition for a specific agent with associated weight

**Parameters**:

- name: **str**, agent name, 
- lhs: **str**, the string expression for lhs formula, latex expression not supported, should be functions of specific format agent_name(SV), or simply a constant value
- lhs_state: **Dict[str, torch.Tensor]**, the specific value of SV to evaluate lhs at for the agent/endogenous variable
- comparator: **Comparator**
- rhs: **str**, the string expression for lhs formula, latex expression not supported, should be functions of specific format agent_name(SV), or simply a constant value
- rhs_state: **Dict[str, torch.Tensor]**, the specific value of SV to evaluate rhs at for the agent/endogenous variable, if rhs is a constant, this can be an empty dictionary
- label: **str** label for the condition, by default, it will self-increment `agent_cond_1`, `agent_cond_2`,...
- weight: **float**, weight in total loss computation
- loss_reduction: **LossReductionMethod**, `LossReductionMethod.MSE` for mean squared error, or `LossReductionMethod.MAE` for mean absolute error

### add_endog
```py
def add_endog(self, name: str, 
            config: Dict[str, Any] = DEFAULT_LEARNABLE_VAR_CONFIG,
            overwrite=False)
```

Add a single [EndogVar](models.md#endogvar), with relevant config of neural network representation. If called before states are set, should raise an error.

**Parameters**:

- name: unique identifier of the endogenous variable.
- Config: specifies number of layers/hidden units of the neural network. Check [Constants](utils.md#constants) for default values.
- overwrite: overwrite the previous endogenous variable with the same name, used for loading, default: False

### add_endogs
```py
def add_endogs(self, names: List[str], 
               configs: Dict[str, Dict[str, Any]]={})
```
Add multiple [EndogVars](models.md#agent) at the same time, each with different configurations.

### add_endog_condition
```py
def add_endog_condition(self, name: str, 
                    lhs: str, lhs_state: Dict[str, torch.Tensor], 
                    comparator: Comparator, 
                    rhs: str, rhs_state: Dict[str, torch.Tensor], 
                    label: str=None,
                    weight: float=1.0, 
                    loss_reduction: LossReductionMethod=LossReductionMethod.MSE):
```
Add boundary/initial condition for a specific endogenous variable with associated weight

**Parameters**:

- name: **str**, agent name, 
- lhs: **str**, the string expression for lhs formula, latex expression not supported, should be functions of specific format agent_name(SV), or simply a constant value
- lhs_state: **Dict[str, torch.Tensor]**, the specific value of SV to evaluate lhs at for the agent/endogenous variable
- comparator: **Comparator**
- rhs: **str**, the string expression for lhs formula, latex expression not supported, should be functions of specific format agent_name(SV), or simply a constant value
- rhs_state: **Dict[str, torch.Tensor]**, the specific value of SV to evaluate rhs at for the agent/endogenous variable, if rhs is a constant, this can be an empty dictionary
- label: **str** label for the condition, by default, it will self-increment `endog_cond_1`, `endog_cond_2`,...
- weight: **float**, weight in total loss computation
- loss_reduction: **LossReductionMethod**, `LossReductionMethod.MSE` for mean squared error, or `LossReductionMethod.MAE` for mean absolute error

### add_equation
```py
def add_equation(self, eq: str, label: str=None)
```

Add an [equation](evaluations.md#equation) to define a new variable. 


### add_endog_equation
```py
def add_endog_equation(self, eq: str, label: str=None, weight=1.0, loss_reduction: LossReductionMethod=LossReductionMethod.MSE)
```

Add an [endogenous equation](evaluations.md#endogequation) for loss computation.

### add_constraint
```py
def add_constraint(self, lhs: str, comparator: Comparator, rhs: str, label: str=None, weight=1.0, loss_reduction: LossReductionMethod=LossReductionMethod.MSE)
```

Add a [constraint](evaluations.md#constraint) for loss computation.

### add_hjb_equation
```py
def add_hjb_equation(self, eq: str, label: str=None, weight=1.0, loss_reduction: LossReductionMethod=LossReductionMethod.MSE)
```

Add an [HJB Equation](evaluations.md#hjbequation) for loss computation.

### add_system
```py
def add_system(self, system: System, weight=1.0)
```

Add a [System](evaluations.md#system) for loss computation.

### set_config
```py
def set_config(self, config: Dict[str, Any] = DEFAULT_CONFIG)
```

This function overwrites the existing configurations. Can be used for L-BFGS finetuning

### train_model
```py
def train_model(self, model_dir: str="./", filename: str=None, full_log=False, variables_to_track: List[str]=[]):
```

The entire loop of training

**Parameters**:

- model_dir: **str**, the directory to save the model. If the directory doesn't exist, it will be created automatically.
- filename: **str**, the filename of the model, it will be the prefix for loss table and log file.
- full_log: **bool**, whether or not log all individual losses in the log file.
- variables_to_track: **List[str]**, variables to keep track of.

### eval_model
```py
def eval_model(self, full_log=False)
```

The entire loop of evaluation

### validate_model_setup
```py
def validate_model_setup(self, model_dir="./")
```

Check that all the equations/constraints given are valid. If not, log the errors in a file, and raise an ultimate error.

### save_model
```py
def save_model(self, model_dir: str = "./", filename: str=None, verbose=False)
```

Save all the agents, endogenous variables (pytorch model and configurations), and all other configurations of the PDE model.

**Parameters**:

- model_dir: **str**, the directory to save the model
- filename: **str**, the filename to save the model without suffix, default: self.name 


### load_model
```py
def load_model(self, dict_to_load: Dict[str, Any])
```

Load all the agents, endogenous variables (pytorch model and configurations) from the dictionary.


### plot_vars
```py
def plot_vars(self, vars_to_plot: List[str], ncols: int=4)
```

**Parameters**:

- vars_to_plot: **List[str]**, variable names to plot, can be an equation defining a new variable. If Latex, need to be enclosed by $$ symbols
- ncols: **int**, number of columns to plot, default: 4
- elev, azim, roll: view angles for 3D plots. See <a href="https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html">Matplotlib Document</a> for details.

