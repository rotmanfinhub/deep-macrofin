# Discontinuous and Oscillating Function

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/basic_examples/function_approximation.ipynb" target="_blank">function_approximation.ipynb</a>.

## Problem Setup
$$
y=
\begin{cases} 
    5 + \sum_{k=1}^4 \sin(kx), x<0\\
    \cos(10x), x\geq 0 
\end{cases}
$$

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
discont_approx2 = PDEModel("discontinuous_approximator", {"num_epochs": 50000}) # define PDE model with 50,000 epochs
discont_approx2.set_state(["x"], {"x": [-3., 3.]}) # set the state variable "x" with its range
discont_approx2.add_endog("y", { 
    "hidden_units": [40, 40], # specify the architecture with two hidden layers of 40 units each
    "activation_type": ActivationType.SiLU, # set the activation function to SiLU
})
discont_approx2.add_endog_equation("y=5+sin(x)+sin(2*x)+sin(3*x)+sin(4*x)") 
discont_approx2.add_endog_equation("y=cos(10*x)") 
```

3. Train and evaluate
```py
discont_approx2.train_model("./models/discont_approx2", "discont_approx2.pt", True)
discont_approx2.eval_model(True)
```

4. To load a trained model
```py
discont_approx2.load_model(torch.load("./models/discont_approx2/discont_approx2_best.pt"))
```

5. Plot the solutions
```py
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
x = np.linspace(-3, 3)
x_neg = np.linspace(-3, 0)
x_pos = np.linspace(0, 3)
ax[0].plot(x_neg, 5+np.sin(x_neg)+np.sin(2*x_neg)+np.sin(3*x_neg)+np.sin(4*x_neg), label="5+sin(x)+sin(2x)+sin(3x)+sin(4x)", color="red")
ax[0].plot(x_pos, np.cos(10*x_pos), label="cos(10x)", color="red")
ax[1].plot(x_neg, np.cos(x_neg)+2*np.cos(2*x_neg)+3*np.cos(3*x_neg)+4*np.cos(4*x_neg), label="cos(x)+2cos(2x)+3cos(3x)+4cos(4x)", color="red")
ax[1].plot(x_pos, -10*np.sin(10*x_pos), label="-10sin(10x)", color="red")
ax[2].plot(x_neg, -np.sin(x_neg)-4*np.sin(2*x_neg)-9*np.sin(3*x_neg)-16*np.sin(4*x_neg), label="-sin(x)-4sin(2x)-9sin(3x)-16sin(4x)", color="red")
ax[2].plot(x_pos, -100*np.cos(10*x_pos), label="-100cos(10x)", color="red")
discont_approx2.endog_vars["y"].plot("y", {"x": [-3, 3]}, ax=ax[0])
discont_approx2.endog_vars["y"].plot("y_x", {"x": [-3, 3]}, ax=ax[1])
discont_approx2.endog_vars["y"].plot("y_xx", {"x": [-3, 3]}, ax=ax[2])
plt.subplots_adjust()
plt.show()
```
