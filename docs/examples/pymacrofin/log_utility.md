# Log Utility Problem

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/develop/examples/pymacrofin_eg/1d_problem.ipynb" target="_blank">1d_problem.ipynb</a>.

## Problem Setup
This is Proposition 4 from <a href="https://www.aeaweb.org/articles?id=10.1257/aer.104.2.379" target="_blank">Brunnermeier and Sannikov 2014</a>[^1]

[^1]: Brunnermeier, Markus K. and Sannikov, Yuliy, *"A Macroeconomic Model with a Financial Sector"*, SIAM Review, 104(2): 379–421, 2014

In the implementation, we firstly fit the initial functions defined by PyMacroFin:

$$
q = 
\begin{cases}
        1.05 + 0.06/0.3 \eta, \eta < 0.3\\
        1.1 - 0.03/0.7 \eta, \eta \geq 0.3
\end{cases}
$$

$$
\psi = 
\begin{cases}
    1/0.3 \eta, \eta < 0.3\\
    1, \eta \geq 0.3
\end{cases}
$$

A single initial condition $q(0)=\sqrt{2\underline{a}\kappa + (\kappa r)^2 + 1} - \kappa r$ is used.

We rewrite the equations defining $\sigma_t^q$, and use the following equations and endogenous equations for training the model, with an additional constraint $\psi \leq 1$.

Equations:

$$\iota = \frac{q^2-1}{2\kappa}$$

$$\sigma_t^q = \frac{\sigma}{1 - \frac{1}{q}\frac{\partial q}{\partial \eta}(\psi - \eta)} - \sigma$$

$$\sigma_t^\eta = \frac{\psi - \eta}{\eta}(\sigma + \sigma_t^q)$$

$$\mu_t^\eta = (\sigma_t^\eta)^2 + \frac{a - \iota}{q} + (1-\psi)(\underline{\delta} - \delta) - \rho$$

Endogenous equations:

$$(r(1-\eta) + \rho\eta)q = \psi a + (1-\psi)\underline{a} - \iota$$

$$(\sigma + \sigma_t^q) ^2(\psi / \eta - (1-\psi) / (1-\eta)) = \frac{a - \underline{a}}{q} + \underline{\delta} - \delta$$

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

2. Fit initial function
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
sys1 = System([Constraint("e", Comparator.LT, "0.3", label="smaller")], "sys1")
sys1.add_endog_equation("q=1.05+.06/.3*e")
sys1.add_endog_equation("psi = 1/.3*e")
sys2 = System([Constraint("e", Comparator.GEQ, "0.3", label="smaller")], "sys2")
sys2.add_endog_equation("q=1.1 - .03/.7*e")
sys2.add_endog_equation("psi = 1")
pde_model.add_system(sys1)
pde_model.add_system(sys2)

print(pde_model)
if not os.path.exists("./models/BruSan14_log_utility/model_init.pt"):
    pde_model.train_model("./models/BruSan14_log_utility", "model_init.pt", True)
    pde_model.load_model(torch.load("./models/BruSan14_log_utility/model_init_best.pt"))
    pde_model.eval_model(True)
else:
    pde_model.load_model(torch.load("./models/BruSan14_log_utility/model_init_best.pt"))
    pde_model.eval_model(True)
```

3. Define the problem
```py
set_seeds(0)
pde_model = PDEModel("BruSan14_log_utility", {"num_epochs": 100, "loss_log_interval": 10, "optimizer_type": OptimizerType.Adam}, latex_var_mapping=latex_var_mapping)
pde_model.set_state(["e"], {"e": [0., 1.0]})
pde_model.add_endogs(["q", "psi"], configs={
    "q": {
        "positive": True
    },
    "psi": {
        "positive": True
    }
})
pde_model.add_params({
    "sig": .1,
    "deltae": .05,
    "deltah": .05,
    "rho": .06,
    "r": .05,
    "a": .11,
    "ah": .07,
    "kappa": 2,
})
pde_model.add_endog_condition("q", 
                              "q(SV)", {"SV": torch.zeros((1, 1))},
                              Comparator.EQ,
                              "(2*ah*kappa + (kappa*r)**2 + 1)**0.5 - kappa*r", pde_model.variable_val_dict,
                              label="q_min")
pde_model.add_equation(r"$\iota = \frac{q^2-1}{ 2 * \kappa}$")
pde_model.add_equation(r"$\sigma_t^q = \frac{\sigma}{1 - \frac{1}{q} * \frac{\partial q}{\partial \eta} * (\psi - \eta)} - \sigma$")
pde_model.add_equation(r"$\sigma_t^\eta = \frac{\psi - \eta}{\eta} * (\sigma + \sigma_t^q)$")
pde_model.add_equation(r"$\mu_t^\eta = (\sigma_t^\eta)^2 + \frac{a - \iota}{q} + (1-\psi) * (\underline{\delta} - \delta) - \rho$")

pde_model.add_constraint("psi", Comparator.LEQ, "1")
pde_model.add_endog_equation(r"$(\sigma + \sigma_t^q) ^2 * (\psi / \eta - (1-\psi) / (1-\eta)) = \frac{a - \underline{a}}{q} + \underline{\delta} - \delta$")

sys1 = System([Constraint("psi", Comparator.LT, "1", "non-opt")], label="non-opt", latex_var_mapping=latex_var_mapping)
sys1.add_endog_equation(r"$(r*(1-\eta) + \rho * \eta) * q = \psi * a + (1-\psi) * \underline{a} - \iota$")
sys2 = System([Constraint("psi", Comparator.GEQ, "1", "opt")], label="opt", latex_var_mapping=latex_var_mapping)
sys2.add_endog_equation(r"$(r*(1-\eta) + \rho * \eta) * q = a - \iota$")

pde_model.add_system(sys1)
pde_model.add_system(sys2)
```

4. Train and evaluate
```py
pde_model.load_model(torch.load("./models/BruSan14_log_utility/model_init_best.pt"))
pde_model.train_model("./models/BruSan14_log_utility", "model.pt", True)
```

5. To load a trained model
```py
pde_model.load_model(torch.load("./models/BruSan14_log_utility/model_init_best.pt"))
```

6. Plot the solutions
```py
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
e = np.linspace(0, 1)
pde_model.endog_vars["q"].plot("q", {"e": [0, 1]}, ax=ax[0])
pde_model.endog_vars["psi"].plot("psi", {"e": [0, 1]}, ax=ax[1])
plt.subplots_adjust()
plt.show()
```