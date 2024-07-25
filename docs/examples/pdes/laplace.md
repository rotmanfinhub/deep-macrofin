# Laplace Equation Dirichlet Problem

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/develop/examples/basic_examples/basic_pdes.ipynb" target="_blank">basic_pdes.ipynb</a>.

## Problem Setup
$$\nabla^2 T = \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0, T(x,0)=T(x,\pi)=0, T(0,y)=1$$

The solution is $T(x,y) = \frac{2}{\pi} \arctan\frac{\sin y}{\sinh x}$

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
pde1 = PDEModel("laplace_dirichlet") # define PDE model for Laplace's equation with Dirichlet boundary conditions
pde1.set_state(["x", "y"], {"x": [0, 3.], "y": [0, np.pi]}) # set state variables "x" and "y" with their respective ranges
pde1.add_endog("T") # add endogenous variable "T" to represent the temperature

pde1.add_endog_equation(r"$\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0$", label="base_pde") 
# add Laplace's equation as the endogenous equation

zero_ys = torch.zeros((100, 2)) # create tensor for boundary condition at y=0
zero_ys[:, 0] = torch.Tensor(np.linspace(0, 3, 100)) # set x-values from 0 to 3

pi_ys = torch.zeros((100, 2)) # create tensor for boundary condition at y=pi
pi_ys[:, 0] = torch.Tensor(np.linspace(0, 3, 100)) # set x-values from 0 to 3
pi_ys[:, 1] = torch.pi # set y-value to pi

zero_xs = torch.zeros((100, 2)) # create tensor for boundary condition at x=0
zero_xs[:, 1] = torch.Tensor(np.linspace(0, np.pi, 100)) # set y-values from 0 to pi

pde1.add_endog_condition("T", 
                              "T(SV)", {"SV": zero_ys},
                              Comparator.EQ,
                              "0", {},
                              label="bc_zeroy") # add boundary condition for T=0 at y=0

pde1.add_endog_condition("T", 
                              "T(SV)", {"SV": pi_ys},
                              Comparator.EQ,
                              "0", {},
                              label="bc_piy") # add boundary condition for T=0 at y=pi

pde1.add_endog_condition("T", 
                              "T(SV)", {"SV": zero_xs},
                              Comparator.EQ,
                              "1", {},
                              label="bc_zerox") # add boundary condition for T=1 at x=0
```

3. Train and evaluate
```py
pde1.train_model("./models/laplace_dirichlet", "laplace_dirichlet.pt", True)
pde1.eval_model(True)
```

4. To load a trained model
```py
pde1.load_model(torch.load("./models/laplace_dirichlet/laplace_dirichlet.pt"))
```

5. Plot the solutions
```py
fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={"projection": "3d"})
x_np = np.linspace(0, 3, 100)
y_np = np.linspace(0, np.pi, 100)
X, Y = np.meshgrid(x_np, y_np)
exact_Z = 2. / np.pi * np.arctan(np.sin(Y) / np.sinh(X))
ax.plot_surface(X, Y, exact_Z, label="exact", alpha=0.6)
pde1.endog_vars["T"].plot("T", {"x": [0, 3.], "y": [0, np.pi]}, ax=ax)
plt.subplots_adjust()
plt.show()
```
