# Log Utility Problem

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/develop/examples/pymacrofin_eg/1d_problem.ipynb" target="_blank">1d_problem.ipynb</a>.

## Problem Setup
This is Proposition 4 from <a href="https://www.aeaweb.org/articles?id=10.1257/aer.104.2.379" target="_blank">Brunnermeier and Sannikov 2014</a>[^1]


[^1]: Brunnermeier, Markus K. and Sannikov, Yuliy, *"A Macroeconomic Model with a Financial Sector"*, SIAM Review, 104(2): 379–421, 2014

## Implementation

1. Import necessary packages
```py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from deep_macrofin import PDEModel
from deep_macrofin import Comparator, Constraint, System, OptimizerType, plot_loss_df, set_seeds
```

2. Define problem  
Here, we first set a random seed for reproducibility, and we set up endogenous variable and define parameters.
```py
set_seeds(0) # set random seed for reproducibility
pde_model = PDEModel("BruSan14_log_utility", 
                     {"num_epochs": 6000, "loss_log_interval": 100, 
                      "optimizer_type": OptimizerType.Adam}, latex_var_mapping=latex_var_mapping) # define PDE model to solve
pde_model.set_state(["e"], {"e": [0., 1.]}) # set the state variable, which defines the dimensionality of the problem
pde_model.add_endogs(["q", "psi"], configs={ # we use endogenous variable to represent the function we want to approximate
    "q": {
        "positive": True
    },
    "psi": {
        "positive": True
    }
})
pde_model.add_params({ # define parameters
    "sig": .1,
    "deltae": .05,
    "deltah": .05,
    "rho": .06,
    "r": .05,
    "ae": .11,
    "ah": .07,
    "kappa": 2,
})
pde_model.add_endog_equation("q=1.05+.06/.3*e") # endogenous equations are used to represent the ODE
pde_model.add_endog_equation("psi = 1/.3*e")
pde_model.add_endog_equation("q=1.1 - .03/.7*e")
pde_model.add_endog_equation("psi = 1")
```

3. Train and evaluate
```py
pde_model.train_model("./models/BruSan14_log_utility", "model_init.pt", True)
pde_model.eval_model(True)
```

4. To load a trained model
```py
pde_model.load_model(torch.load("./models/BruSan14_log_utility/model_init_best.pt"))
```

5. Plot the solutions
```py
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
e = np.linspace(0, 1)
e1 = np.linspace(0, 0.3)
e2 = np.linspace(0.3, 1)
ax[0].plot(e1, 1.05+0.06/0.3*e1, label="1.05+.06/.3*e", color="red")
ax[0].plot(e2, 1.1 - .03/.7*e2, label="1.1 - .03/.7*e", color="red")
ax[1].plot(e1, 1/.3*e1, label="1/.3e", color="red")
ax[1].plot(e2, np.ones_like(e2), label="1", color="red")
pde_model.endog_vars["q"].plot("q", {"e": [0, 1]}, ax=ax[0])
pde_model.endog_vars["psi"].plot("psi", {"e": [0, 1]}, ax=ax[1])
plt.subplots_adjust()
plt.show()
```