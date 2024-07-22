# Second-order ODE

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/basic_examples/basic_odes2.ipynb" target="_blank">basic_odes2.ipynb</a>.

## Problem Setup
$$y''-10y'+9y=5t, y(0)=-1, y'(0)=2$$

The solution is $y=\frac{50}{81} + \frac{5}{9}t + \frac{31}{81}e^{9t} - 2e^t$

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
ode = PDEModel("second_order_linear", config={"num_epochs": 10000}) # define PDE model to solve
ode.set_state(["t"], {"t": [0., 0.25]}) # set the state variable, which defines the dimensionality of the problem
ode.add_endog("y", config={
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "hidden_units": [50, 50, 50],
    "activation_type": ActivationType.Tanh,
    "positive": False,
    "derivative_order": 2,
}) # we use endogenous variable to represent the function we want to approximate
ode.add_endog_equation("y_tt-10*y_t+9*y=5*t", label="base_ode", weight=0.01) # endogenous equations are used to represent the ODE
ode.add_endog_condition("y", 
                            "y(SV)", {"SV": torch.zeros((1, 1))},
                            Comparator.EQ,
                            "-1", {},
                            label="ic1") # define initial condition
ode.add_endog_condition("y", 
                            "y_t(SV)", {"SV": torch.zeros((1, 1))},
                            Comparator.EQ,
                            "2", {},
                            label="ic2") # add a second initial condition
```

3. Train and evaluate
```py
ode.train_model("./models/second_order_ode3", "second_order_ode3.pt", True)
ode.eval_model(True)
```

4. To load a trained model
```py
ode1.load_model(torch.load("./models/ode1/ode1.pt"))
```

5. Plot the solutions
```py
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
x = np.linspace(0, 0.25)
ax[0].plot(x, 50/81+5/9*x+31/81*np.exp(9*x)-2*np.exp(x), label="y_true")
ax[1].plot(x, 5/9+31/9*np.exp(9*x)-2*np.exp(x), label="y_t_true")
ax[2].plot(x, 31*np.exp(9*x)-2*np.exp(x), label="y_tt_true")
ode.endog_vars["y"].plot("y", {"t": [0, 0.25]}, ax=ax[0])
ode.endog_vars["y"].plot("y_t", {"t": [0, 0.25]}, ax=ax[1])
ode.endog_vars["y"].plot("y_tt", {"t": [0, 0.25]}, ax=ax[2])
plt.subplots_adjust()
plt.show()
```
