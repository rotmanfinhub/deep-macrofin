# Base ODE 1

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/develop/examples/basic_examples/basic_odes.ipynb" target="_blank">basic_odes.ipynb</a>.

## Problem Setup
$$\frac{dx}{dt} = 2t, x(0)=1, t\in[-2,2]$$

The solution is $x(t)=t^2+1$

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

2. Define problem.  
Here, we use the default training configuration, and default setup for learnable endogenous variable.

```py
ode1 = PDEModel("ode1") # define PDE model to solve
ode1.set_state(["t"], {"t": [-2., 2.]}) # set the state variable, which defines the dimensionality of the problem
ode1.add_endog("x") # we use endogenous variable to represent the function we want to approximate
ode1.add_endog_equation(r"$\frac{\partial x}{\partial t} = 2 * t$", label="base_ode") # endogenous equations are used to represent the ODE
ode1.add_endog_condition("x", 
                              "x(SV)", {"SV": torch.zeros((1, 1))},
                              Comparator.EQ,
                              "1", {},
                              label="initial_condition") # define initial condition
```

3. Train and evaluate
```py
ode1.train_model("./models/ode1", "ode1.pt", True)
ode1.eval_model(True)
```

4. To load a trained model
```py
ode1.load_model(torch.load("./models/ode1/ode1.pt"))
```

5. Plot the solutions
```py
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
t = np.linspace(-2, 2)
ax[0].plot(t, t**2+1, label="t^2+1")
ax[1].plot(t, 2*t, label="2*t")
ax[2].plot(t, np.ones_like(t) * 2, label="2")
ode1.endog_vars["x"].plot("x", {"t": [-2, 2]}, ax=ax[0])
ode1.endog_vars["x"].plot("x_t", {"t": [-2, 2]}, ax=ax[1])
ode1.endog_vars["x"].plot("x_tt", {"t": [-2, 2]}, ax=ax[2])
plt.subplots_adjust()
plt.show()
```
