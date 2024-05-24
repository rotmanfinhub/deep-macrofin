from typing import Any, Dict, List

from .learnable_var import LearnableVar
from .model_utils import LearnableModelType


class Agent(LearnableVar):
    def __init__(self, name: str, state_variables: List[str], config: Dict[str, Any]):
        '''
        Initialize the Agent model.

        Inputs:
            - name (str): The name of the model.
            - state_variables (List[str]): List of state variables.

        Config: specifies number of layers/hidden units of the neural network.
            - device: **str**, the device to run the model on (e.g., "cpu", "cuda"), default will be chosen based on whether or not GPU is available
            - hidden_units: **List[int]**, number of units in each layer
            - layer_type: **str**, a selection from the LayerType enum, default: LayerType.MLP
            - activation_type: *str**, a selection from the ActivationType enum, default: ActivationType.Tanh
            - positive: **bool**, apply softplus to the output to be always positive if true, default: false
            - hardcode_function: a lambda function for hardcoded forwarding function.
            - derivative_order: int, an additional constraint for the number of derivatives to take, default: 2, so for a function with one state variable, we can still take multiple derivatives 
        '''
        super(Agent, self).__init__(name, state_variables, config)
        self.model_type = LearnableModelType.Agent

