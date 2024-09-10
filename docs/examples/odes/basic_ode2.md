# Base ODE 2

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/basic_examples/basic_odes.ipynb" target="_blank">basic_odes.ipynb</a>.

## Problem Setup
$$\frac{d x}{d t} = x, x(0)=1$$

The solution is $x(t)=e^t$

## Implementation

1. Import necessary packages
```py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from deep_macrofin import PDEModel
from deep_macrofin import ActivationType, Comparator, EndogVar, EndogVarConditions, EndogEquation
```

2. Define problem  
Here, we use the default training configuration, and default setup for learnable endogenous variable.
```py
ode2 = PDEModel("ode2") # define PDE model to solve
ode2.set_state(["t"], {"t": [-2., 2.]}) # set the state variable, which defines the dimensionality of the problem
ode2.add_endog("x") # we use endogenous variable to represent the function we want to approximate
ode2.add_endog_equation("x_t=x", label="base_ode") 
ode2.add_endog_condition("x", 
                              "x(SV)", {"SV": torch.zeros((1, 1))},
                              Comparator.EQ,
                              "1", {},
                              label="initial_condition") 
```

3. Train and evaluate
```py
ode2.train_model("./models/ode2", "ode2.pt", True)
ode2.eval_model(True)
```

4. To load a trained model
```py
ode2.load_model(torch.load("./models/ode2/ode2.pt"))
```

5. Plot the solutions
```py
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
t = np.linspace(-2, 2)
ax[0].plot(t, np.exp(t), label="e^t")
ax[1].plot(t, np.exp(t), label="e^t")
ax[2].plot(t, np.exp(t), label="e^t")
ode2.endog_vars["x"].plot("x", {"t": [-2, 2]}, ax=ax[0])
ode2.endog_vars["x"].plot("x_t", {"t": [-2, 2]}, ax=ax[1])
ode2.endog_vars["x"].plot("x_tt", {"t": [-2, 2]}, ax=ax[2])
plt.subplots_adjust()
plt.show()
```
