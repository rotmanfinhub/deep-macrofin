# deep_macrofin.models

## LearnableVar


```py
class LearnableVar(name: str, state_variables: List[str], config: Dict[str, Any])
```

Base class for agents and endogenous variables. This is a subclass of `torch.nn` module.

**Parameters**:  

- name: **str**, The name of the model.  
- state_variables: **List[str]**, List of state variables.  
- config: **Dict[str, Any]**, specifies number of layers/hidden units of the neural network and highest order of derivatives to take. 
    - device: **str**, the device to run the model on (e.g., "cpu", "cuda"), default will be chosen based on whether or not GPU is available
    - hidden_units: **List[int]**, number of units in each layer, default: [30, 30, 30, 30]
    - output_size: **int**, number of output units, default: 1 for MLP, and last hidden unit size for KAN and MultKAN
    - layer_type: **str**, a selection from the LayerType enum, default: LayerType.MLP
    - activation_type: *str**, a selection from the ActivationType enum, default: ActivationType.Tanh
    - positive: **bool**, apply softplus to the output to be always positive if true, default: false (This has no effect for KAN.)
    - hardcode_function: a lambda function for hardcoded forwarding function, default: None
    - derivative_order: **int**, an additional constraint for the number of derivatives to take, so for a function with one state variable, we can still take multiple derivatives, default: number of state variables
    - batch_jac_hes: **bool**, whether to use batch jacobian or hessian for computing derivatives, default: False (When True, only name_Jac and name_Hess are included in the derivatives dictionary, and derivative_order is ignored; When False, all derivatives name_x, name_y, etc are included.)

### get_all_derivatives
Get all derivatives of the current variable, upto a specific order defined by the user with `derivative_order`. The construction of derivatives can be found in [derivative_utils](#derivative_utils).

```py
def get_all_derivatives(self):
    '''
    Returns a dictionary of derivative functional mapping 
    e.g. if name="qa", state_variables=["e", "t"], derivative_order=2, it will return 
    {
        "qa": self.forward
        "qa_e": lambda x:self.compute_derivative(x, "e")
        "qa_t": lambda x:self.compute_derivative(x, "t"),
        "qa_ee": lambda x:self.compute_derivative(x, "ee"),
        "qa_tt": lambda x:self.compute_derivative(x, "tt"),
        "qa_et": lambda x:self.compute_derivative(x, "et"),
        "qa_te": lambda x:self.compute_derivative(x, "te"),
    }

    If "batch_jac_hes" is True, it will only include the name_Jac and name_Hess in the dictionary. 
    {
        "qa_Jac": lambda x:vmap(jacrev(self.forward))(x),
        "qa_Hess": lambda x:vmap(hessian(self.forward))(x),
    }

    Note that the last two will be the same for C^2 functions, 
    but we keep them for completeness. 
    '''
```

### plot
Plot a specific function attached to the learnable variable over the domain.

```py
def plot(self, target: str, domain: Dict[str, List[np.float32]]={}, ax=None):
    '''
    Inputs:
        target: name for the original function, or the associated derivatives to plot
        domain: the range of state variables to plot. 
        If state_variables=["x", "y"] domain = {"x": [0,1], "y":[-1,1]}, it will be plotted on the region [0,1]x[-1,1].
        If one of the variable is not provided in the domain, [0,1] will be taken as the default
        ax: a matplotlib.Axes object to plot on, if not provided, it will be plotted on a new figure

    This function is only supported for 1D or 2D state_variables.
    '''
```

### to_dict
```py
def to_dict(self)
```

Save all the configurations and weights to a dictionary.

```py
dict_to_save = {
    "name": self.name,
    "model": self.state_dict(),
    "model_config": self.config,
    "system_rng": random.getstate(),
    "numpy_rng": np.random.get_state(),
    "torch_rng": torch.random.get_rng_state(),
}
```

### from_dict
```py
def from_dict(self, dict_to_load: Dict[str, Any])
```

Load all the configurations and weights from a dictionary.

## Agent

```py
class Agent(name: str, state_variables: List[str], config: Dict[str, Any])
```

Subclass of `LearnableVar`. Defines agent wealth multipliers. 

## EndogVar

```py
class EndogVar(name: str, state_variables: List[str], config: Dict[str, Any])
```

Subclass of `LearnableVar`. Defines endogenous variables. 

## derivative_utils

### get_derivs_1order
```py
def get_derivs_1order(y, x, idx):
```
Returns the first order derivatives of $y$ w.r.t. $x$. Automatic differentiation used.

**Example**:
```py
x = torch.tensor([[1.0, 2.0]]) # assuming two variables x1, x2
x.requires_grad_(True)
y1 = x**2
print(get_derivs_1order(y1, x, 0)) 
'''
Output: [[2.]] dy1/dx1 = 2
'''
print(get_derivs_1order(y1, x, 1))
'''
Output: [[4.]] dy1/dx2 = 4
'''

x = torch.tensor([[1.0, 2.0]]) # assuming two variables x1, x2
x.requires_grad_(True)
y2 = x[:,0:1] * x[:,1:2] 
print(get_derivs_1order(y2, x, 0)) 
'''
Output: [[2.]] dy2/dx1 = 2
'''
print(get_derivs_1order(y2, x, 1))
'''
Output: [[1.]] dy2/dx2 = 1
'''
```

### get_all_derivs
```py
def get_all_derivs(target_var_name="f", all_vars: List[str] = ["x", "y", "z"], derivative_order = 2) -> Dict[str, Callable]:
```

Implements Algorithm 1 in the paper. Higher order derivatives are computed iteratively using dynamic programming and first order derivative.
<!-- TODO: Add reference to our paper -->
**Example**:
```py
derivs = get_all_derivs(target_var_name="qa", all_vars= ["e", "t"], derivative_order = 2)
'''
derivs = {
    "qa_e": lambda y, x, idx=0: get_derivs_1order(y, x, idx)
    "qa_t": lambda y, x, idx=1: get_derivs_1order(y, x, idx),
    "qa_ee": lambda y, x, idx=0: get_derivs_1order(y, x, idx), # here y=qa_e
    "qa_et": lambda y, x, idx=1: get_derivs_1order(y, x, idx), # here y=qa_e
    "qa_te": lambda y, x, idx=0: get_derivs_1order(y, x, idx), # here y=qa_t
    "qa_tt": lambda y, x, idx=1: get_derivs_1order(y, x, idx), # here y=qa_t
}

Afterwards, [e,t] should be passed as x into the lambda function
'''
```

## Activation Functions

The supported activation functions are listed in `ActivationTypes` below. ReLU, SiLU, Sigmoid and Tanh are default PyTorch activation functions. Wavelet is from <a href="https://openreview.net/forum?id=DO2WFXU1Be" target="_blank">Zhao, Ding, and Prakash 2024</a>[^1]. It is a learnable activation function of the form:

$$\text{Wavelet}(x) = w_1 \sin(x) + w_2 \cos(x),$$

where $w_1$ and $w_2$ are learnable parameters.

[^1]: Zhiyuan Zhao, Xueying Ding, and B. Aditya Prakash, *"PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks"*, The Twelfth International Conference on Learning Representations, 2024

## Constants

These constants are used to identify model/layer/activation types for initialization.

```py
class LearnableModelType(str, Enum):
    Agent="Agent"
    EndogVar="EndogVar"

class LayerType(str, Enum):
    MLP="MLP"
    KAN="KAN"
    MultKAN="MultKAN"

class ActivationType(str, Enum):
    ReLU="relu"
    SiLU="silu"
    Sigmoid="sigmoid"
    Tanh="tanh"
    Wavelet="wavelet"
```
