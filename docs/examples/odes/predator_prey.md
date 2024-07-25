# Predator-Prey Model

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/develop/examples/basic_examples/system_ode.ipynb" target="_blank">system_ode.ipynb</a>.

## Problem Setup
$$\begin{bmatrix}\dot{x} \\ \dot{y}\end{bmatrix} = \begin{bmatrix} \alpha x - \beta xy \\ \delta xy - \gamma y\end{bmatrix}$$

In the example, $\alpha=1.1, \beta=0.4, \delta=0.4, \gamma=0.1$

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
Here, we define predator-prey dynamics with specific initial conditions and training epochs.
```py
lv = PDEModel("lotka_volterra", {"num_epochs": 2000}) # define PDE model to solve
lv.set_state(["t"], {"t": [0., 5.]}) # set the state variable, which defines the dimensionality of the problem
lv.add_endog("x") 
lv.add_endog("y") # we use endogenous variable to represent the function we want to approximate
lv.add_endog_equation("x_t = 1.1 * x - 0.4*x*y", label="base_ode1")
lv.add_endog_equation("y_t = 0.4 * x * y - 0.1 * y") # endogenous equations are used to represent the ODE
lv.add_endog_condition("x", 
                              "x(SV)", {"SV": torch.zeros((1, 1))},
                              Comparator.EQ,
                              "1", {},
                              label="initial_condition") # define initial condition
lv.add_endog_condition("y", 
                              "y(SV)", {"SV": torch.zeros((1, 1))},
                              Comparator.EQ,
                              "1", {},
                              label="initial_condition") # define second initial condition
```

3. Train and evaluate
```py
lv.train_model("./models/lotka_volterra", "lotka_volterra.pt", True)
lv.eval_model(True)
```

4. To load a trained model
```py
lv.load_model(torch.load("./models/lotka_volterra/lotka_volterra.pt"))
```

5. Plot the solutions
```py
fig, ax = plt.subplots(2, 2, figsize=(11, 11))
t = np.linspace(0., 5.)
x = lv.endog_vars["x"].derivatives["x"](torch.Tensor(t).unsqueeze(-1).to(lv.device)).detach().cpu().numpy()
y = lv.endog_vars["y"].derivatives["y"](torch.Tensor(t).unsqueeze(-1).to(lv.device)).detach().cpu().numpy()
lv.endog_vars["x"].plot("x", {"t": [0., 5.]}, ax=ax[0][0])
lv.endog_vars["y"].plot("y", {"t": [0., 5.]}, ax=ax[0][1])
ax[1][0].plot(t, 1.1*x-0.4*x*y, label="1.1x-0.4xy")
ax[1][1].plot(t, 0.4*x*y-0.1*y, label="0.4xy-0.1y")
lv.endog_vars["x"].plot("x_t", {"t": [0., 5.]}, ax=ax[1][0])
lv.endog_vars["y"].plot("y_t", {"t": [0., 5.]}, ax=ax[1][1])
plt.subplots_adjust()
plt.show()
```
