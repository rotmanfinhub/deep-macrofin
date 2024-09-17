# Diffusion Equation

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/basic_examples/diffusion_equations.ipynb" target="_blank">diffusion_equations.ipynb</a>.

## Problem Setup
$$\frac{\partial y}{\partial t} = \frac{\partial^2 y}{\partial x^2} - e^{-t} (\sin(\pi x) - \pi^2 \sin (\pi x))$$

with $x\in [-1,1], t\in[0,1]$, $y(x,0)=\sin(\pi x)$, $y(-1,t)=y(1,t)=0$.

Solution: $y=e^{-t}\sin(\pi x)$

## Implementation

1. Import necessary packages
```py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from deep_macrofin import PDEModel
from deep_macrofin import Comparator, EndogVar, EndogVarConditions, EndogEquation
```

2. Define problem
Here, we set up the endogenous variable, endogenous equation, boundary and initial conditions.
```py
pde1 = PDEModel("diffusion_1d") # define pde model to solve
pde1.set_state(["x", "t"], {"x": [-1., 1.], "t": [0, 1.]}) # set the state variable, which defines the dimensionality of the problem
pde1.add_endog("y") # we use endogenous variable to represent the function we want to approximate
pde1.add_endog_equation("y_t = y_xx - exp(-t) * (sin(pi * x) - pi**2 * sin(pi*x))", label="base_pde") # endogenous equations are used to represent the PDE

mone_xs = -1 * torch.ones((100, 2)) # Create a tensor for boundary condition at x = -1, with t values from 0 to 1
mone_xs[:, 1] = torch.Tensor(np.linspace(0, 1, 100))

one_xs = torch.ones((100, 2)) # Create a tensor for boundary condition at x = 1, with t values from 0 to 1
one_xs[:, 1] = torch.Tensor(np.linspace(0, 1, 100))

zero_ts = torch.zeros((100, 2)) # Create a tensor for initial condition at t = 0, with x values from -1 to 1
zero_ts[:, 0] = torch.Tensor(np.linspace(-1, 1, 100))
y_zero_ts = torch.sin(torch.pi * zero_ts[:, 0:1])

pde1.add_endog_condition("y", 
                              "y(SV)", {"SV": mone_xs},
                              Comparator.EQ,
                              "0", {},
                              label="bc_zerox") # Add boundary condition 
pde1.add_endog_condition("y", 
                              "y(SV)", {"SV": one_xs},
                              Comparator.EQ,
                              "0", {},
                              label="bc_onex") # Add boundary condition
pde1.add_endog_condition("y", 
                              "y(SV)", {"SV": zero_ts},
                              Comparator.EQ,
                              "y_zero_ts", {"y_zero_ts": y_zero_ts},
                              label="ic") # Add initial condition
```

3. Train and evaluate
```py
pde1.train_model("./models/diffusion_1d", "diffusion_1d.pt", True)
pde1.eval_model(True)
```

4. To load a trained model
```py
pde1.load_model(torch.load("./models/diffusion_1d/diffusion_1d.pt"))
```

5. Plot the solutions
```py
fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={"projection": "3d"})
x_np = np.linspace(-1, 1, 100)
t_np = np.linspace(0, 1, 100)
X, T = np.meshgrid(x_np, t_np)
exact_Y = np.exp(-T) * np.sin(np.pi*X)
ax.plot_surface(X, T, exact_Y, label="exact", alpha=0.6)
pde1.endog_vars["y"].plot("y", {"x": [-1., 1.], "t": [0, 1.]}, ax=ax)
plt.subplots_adjust()
plt.show()
```
