# deep_macrofin.models

## LearnableVar


```py
class LearnableVar(name: str, state_variables: List[str], config: Dict[str, Any])
```
Inputs:  
- name (str): The name of the model.  
- state_variables (List[str]): List of state variables.  

Config: specifies number of layers/hidden units of the neural network and highest order of derivatives to take.  
- device: **str**, the device to run the model on (e.g., "cpu", "cuda"), default will be chosen based on whether or not GPU is available  
- hidden_units: **List[int]**, number of units in each layer, default: [30, 30, 30, 30]  
- layer_type: **str**, a selection from the LayerType enum, default: LayerType.MLP  
- activation_type: **str**, a selection from the ActivationType enum, default: ActivationType.Tanh  
- positive: **bool**, apply softplus to the output to be always positive if true, default: false  
- hardcode_function: a lambda function for hardcoded forwarding function, default: None  
- derivative_order: **int**, an additional constraint for the number of derivatives to take, so for a function with one state variable, we can still take multiple derivatives, default: number of state variables  

Base class for 

### Agent

### EndogVar


## derivative_utils

```py
def get_derivs_1order(y, x, idx):
```

```py
def get_all_derivs(target_var_name="f", all_vars: List[str] = ["x", "y", "z"], derivative_order = 2) -> Dict[str, Callable]:
```

## Constants

```py
class LearnableModelType(str, Enum):
    Agent="Agent"
    EndogVar="EndogVar"

class LayerType(str, Enum):
    MLP="MLP"
    KAN="KAN"

class ActivationType(str, Enum):
    ReLU="relu"
    SiLU="silu"
    Sigmoid="sigmoid"
    Tanh="tanh"
```
