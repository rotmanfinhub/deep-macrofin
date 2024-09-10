# Time-dependent Heat Equation

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/basic_examples/basic_pdes.ipynb" target="_blank">basic_pdes.ipynb</a>.

## Problem Setup
$$\frac{\partial u}{\partial t} = 0.4 \frac{\partial^2 u}{\partial x^2}, x\in [0,1], t\in[0,1],$$ 

$$u(0,t)=u(1,t)=0, u(x,0)=\sin(\pi x)$$

Solution is $u(x,t)=e^{-0.4\pi^2t} \sin(\pi x)$

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
Here, we first set up the endogenous variables, equations, and initialize boundary conditions.
```py
pde2 = PDEModel("time_heat") # define pde model to solve
pde2.set_state(["x", "t"], {"x": [0, 1.], "t": [0, 1.]}) # set the state variable, which defines the dimensionality of the problem
pde2.add_endog("u") # we use endogenous variable to represent the function we want to approximate
pde2.add_endog_equation(r"$\frac{\partial u}{\partial t} = 0.4 * \frac{\partial^2 u}{\partial x^2}$", label="base_pde") # endogenous equations are used to represent the PDE

zero_xs = torch.zeros((100, 2)) # Create a tensor for boundary condition at x = 0, with t values from 0 to 1
zero_xs[:, 1] = torch.Tensor(np.linspace(0, 1, 100))

one_xs = torch.ones((100, 2)) # Create a tensor for boundary condition at x = 1, with t values from 0 to 1
one_xs[:, 1] = torch.Tensor(np.linspace(0, 1, 100))

zero_ts = torch.zeros((100, 2)) # Create a tensor for initial condition at t = 0, with x values from 0 to 1
zero_ts[:, 0] = torch.Tensor(np.linspace(0, 1, 100))
u_zero_ts = torch.sin(torch.pi * zero_ts[:, 0:1])

pde2.add_endog_condition("u", 
                              "u(SV)", {"SV": zero_xs},
                              Comparator.EQ,
                              "0", {},
                              label="bc_zerox") # Add boundary condition
pde2.add_endog_condition("u", 
                              "u(SV)", {"SV": one_xs},
                              Comparator.EQ,
                              "0", {},
                              label="bc_onex") # Add boundary condition
pde2.add_endog_condition("u", 
                              "u(SV)", {"SV": zero_ts},
                              Comparator.EQ,
                              "u_zero_ts", {"u_zero_ts": u_zero_ts},
                              label="ic") # Add initial condition
```

3. Train and evaluate
```py
pde2.train_model("./models/time_heat", "time_heat.pt", True)
pde2.eval_model(True)
```

4. To load a trained model
```py
pde2.load_model(torch.load("./models/time_heat/time_heat.pt"))
```

5. Plot the solutions
```py
fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={"projection": "3d"})
x_np = np.linspace(0, 1, 100)
t_np = np.linspace(0, 1, 100)
X, T = np.meshgrid(x_np, t_np)
exact_Z = np.exp(-0.4*np.pi**2 * T) * np.sin(np.pi*X)
ax.plot_surface(X, T, exact_Z, label="exact", alpha=0.6)
pde2.endog_vars["u"].plot("u", {"x": [0, 1.], "y": [0, 1.]}, ax=ax)
plt.subplots_adjust()
plt.show()
```
