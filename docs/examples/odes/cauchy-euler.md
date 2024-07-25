# Cauchy-Euler Equation

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/develop/examples/basic_examples/basic_odes.ipynb" target="_blank">basic_odes.ipynb</a>.

## Problem Setup
$$ x^2 y'' + 6xy' + 4y =0, y(1)=6, y(2)=\frac{5}{4}$$

Solution: $y=4x^{-4} + 2 x^{-1}$

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
ode5 = PDEModel("cauchy_euler") # define PDE model to solve
ode5.set_state(["x"], {"x": [1., 2.]}) # set the state variable, which defines the dimensionality of the problem
ode5.add_endog("y") # we use endogenous variable to represent the function we want to approximate
ode5.add_endog_equation("x**2 * y_xx + 6*x*y_x + 4*y=0", label="base_ode") # endogenous equations are used to represent the ODE
ode5.add_endog_condition("y", 
                              "y(SV)", {"SV": torch.ones((1, 1))},
                              Comparator.EQ,
                              "6", {},
                              label="ic1") # define initial condition
ode5.add_endog_condition("y", 
                              "y(SV)", {"SV": 2 * torch.ones((1, 1))},
                              Comparator.EQ,
                              "5/4", {},
                              label="ic2") # define second initial condition
```

3. Train and evaluate
```py
ode5.train_model("./models/cauchy_euler", "cauchy_euler.pt", True)
ode5.eval_model(True)
```

4. To load a trained model
```py
ode5.load_model(torch.load("./models/cauchy_euler/cauchy_euler.pt"))
```

5. Plot the solutions
```py
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
x = np.linspace(1, 2)
ax[0].plot(x, 4*x**(-4) + 2 * x**(-1), label="4x^{-4}+2x^{-1}")
ax[1].plot(x, -16*x**(-5)-2*x**(-2), label="-16x^{-5}-2x^{-2}")
ax[2].plot(x, 80*x**(-6) + 4*x**(-3), label="80x^{-6}+4x^{-3}")
ode5.endog_vars["y"].plot("y", {"x": [1, 2]}, ax=ax[0])
ode5.endog_vars["y"].plot("y_x", {"x": [1, 2]}, ax=ax[1])
ode5.endog_vars["y"].plot("y_xx", {"x": [1, 2]}, ax=ax[2])
plt.subplots_adjust()
plt.show()
```
