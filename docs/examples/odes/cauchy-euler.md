# Cauchy-Euler Equation

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/paper_example/base_ode.ipynb" target="_blank">base_ode.ipynb</a>.

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

2. Define problem  
Here, we set up the endogenous variable, endogenous equation, and initial conditions.
```py
ode = PDEModel("cauchy_euler") # define PDE model to solve
ode.set_state(["x"], {"x": [1., 2.]}) # set the state variable, which defines the dimensionality of the problem
ode.add_endog("y") # we use endogenous variable to represent the function we want to approximate
ode.add_endog_equation("x**2 * y_xx + 6*x*y_x + 4*y=0", label="base_ode") # endogenous equations are used to represent the ODE
ode.add_endog_condition("y", 
                              "y(SV)", {"SV": torch.ones((1, 1))},
                              Comparator.EQ,
                              "6", {},
                              label="ic1") # define initial condition
ode.add_endog_condition("y", 
                              "y(SV)", {"SV": 2 * torch.ones((1, 1))},
                              Comparator.EQ,
                              "5/4", {},
                              label="ic2") # define second initial condition
```

3. Train and evaluate
```py
ode.train_model("./models/cauchy_euler", "cauchy_euler.pt", True)
ode.eval_model(True)
```

4. To load a trained model
```py
ode.load_model(torch.load("./models/cauchy_euler/cauchy_euler.pt"))
```

5. Plot the solutions
```py
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
x = np.linspace(1, 2)
ax[0].plot(x, 4*x**(-4) + 2 * x**(-1), label="4x^{-4}+2x^{-1}")
ax[1].plot(x, -16*x**(-5)-2*x**(-2), label="-16x^{-5}-2x^{-2}")
ax[2].plot(x, 80*x**(-6) + 4*x**(-3), label="80x^{-6}+4x^{-3}")
ode.endog_vars["y"].plot("y", {"x": [1, 2]}, ax=ax[0])
ode.endog_vars["y"].plot("y_x", {"x": [1, 2]}, ax=ax[1])
ode.endog_vars["y"].plot("y_xx", {"x": [1, 2]}, ax=ax[2])
plt.subplots_adjust()
plt.show()
```

## KAN Approach

Here, we try using the KAN layer for the endogenous variable. The rest of the steps are the same.
```py
from deep_macrofin import OptimizerType, set_seeds, LayerType
set_seeds(0) # set random seed for reproducibility
ode = PDEModel("cauchy_euler", config={"num_epochs": 100, "lr": 1, "loss_log_interval": 10}) # define PDE model to solve
ode.set_state(["x"], {"x": [1., 2.]}) # set the state variable, which defines the dimensionality of the problem
ode.add_endog("y", config={
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "hidden_units": [1, 5, 5, 1],
    "layer_type": LayerType.KAN,
    "activation_type": ActivationType.SiLU,
    "positive": False,
    "derivative_order": 2,
}) # we use endogenous variable to represent the function we want to approximate
ode.add_endog_equation("x**2 * y_xx + 6*x*y_x + 4*y=0", label="base_ode") # endogenous equations are used to represent the ODE
ode.add_endog_condition("y", 
                              "y(SV)", {"SV": torch.ones((1, 1))},
                              Comparator.EQ,
                              "6", {},
                              label="ic1") # define initial condition
ode.add_endog_condition("y", 
                              "y(SV)", {"SV": 2 * torch.ones((1, 1))},
                              Comparator.EQ,
                              "5/4", {},
                              label="ic2") # define second initial condition
```