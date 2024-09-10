# Log Utility Problem

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/pymacrofin_eg/1d_problem_relu.ipynb" target="_blank">1d_problem.ipynb</a>.

## Problem Setup
This is Proposition 4 from <a href="https://www.aeaweb.org/articles?id=10.1257/aer.104.2.379" target="_blank">Brunnermeier and Sannikov 2014</a>[^1]

[^1]: Brunnermeier, Markus K. and Sannikov, Yuliy, *"A Macroeconomic Model with a Financial Sector"*, SIAM Review, 104(2): 379–421, 2014

In the deep neural network, we don't have to fit the initial guess function as in PyMacroFin any more.

$q$ should satisfy $q(0)=\frac{\underline{a} - \iota(q(0))}{r}$. We rewrite with $\iota(q) = \frac{q^2-1}{2\kappa}$, and simplify: 

$$q(0) r + \frac{q(0)^2 - 1}{2\kappa} = \underline{a}.$$

We rewrite the equations defining $\sigma_t^q$, and use the following equations and endogenous equations for training the model, with an additional constraint $\psi \leq 1$.

Equations:

$$\iota = \frac{q^2-1}{ 2 \kappa}$$

$$\sigma_t^q = \frac{\sigma}{1 - \frac{1}{q} \frac{\partial q}{\partial \eta} (\psi - \eta)} - \sigma$$

$$\sigma_t^\eta = \frac{\psi - \eta}{\eta} (\sigma + \sigma_t^q)$$

$$\mu_t^\eta = (\sigma_t^\eta)^2 + \frac{a - \iota}{q} + (1-\psi) (\underline{\delta} - \delta) - \rho$$

We constrain $\psi$ by $\psi \leq 1$

Endogenous equations:

$$(\sigma + \sigma_t^q)^2 (\psi / \eta - (1-\psi) / (1-\eta)) = \frac{a - \underline{a}}{q} + \underline{\delta} - \delta$$

$$(r(1-\eta) + \rho \eta) q = \psi a + (1-\psi) \underline{a} - \iota$$

## Implementation

1. Import necessary packages
```py
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from deep_macrofin import PDEModel
from deep_macrofin import ActivationType, Comparator, Constraint, SamplingMethod, System, OptimizerType, plot_loss_df, set_seeds
```

2. Define latex variable mapping
```py
latex_var_mapping = {
    r"\sigma_t^q": "sigq",
    r"\sigma_t^\theta": "sigtheta",
    r"\sigma_t^\eta": "sige",
    r"\mu_t^\eta": "mue",
    r"\mu_t^q": "muq",
    r"\mu_t^\theta": "mutheta",


    r"\rho": "rho",
    r"\underline{a}": "ah",
    r"\underline{\delta}": "deltah",
    r"\delta": "deltae",
    r"\sigma": "sig",
    r"\kappa": "kappa",

    r"\eta": "e",

    r"\theta": "theta",
    r"\psi": "psi",
    r"\iota": "iota",
    r"\Phi": "phi",

}
```

3. Define the problem. We use ReLU as activation function, due to the discontinuity in the first order derivative. Also, we apply a fixed grid sampling method.
```py
set_seeds(0)
pde_model = PDEModel("BruSan14_log_utility", {"sampling_method": SamplingMethod.FixedGrid, "batch_size": 200,
    "num_epochs": 10000, "loss_log_interval": 100, "optimizer_type": OptimizerType.Adam}, latex_var_mapping=latex_var_mapping)
pde_model.set_state(["e"], {"e": [0.01, 0.99]})
pde_model.add_endogs(["q", "psi"], configs={
    "q": {
        "positive": True,
        "hidden_units": [50, 50], 
        "activation_type": ActivationType.ReLU,
    },
    "psi": {
        "positive": True, 
        "hidden_units": [50, 50], 
        "activation_type": ActivationType.ReLU,
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
                              "q(SV)*r + (q(SV) * q(SV) - 1) / (2*kappa) - ah", {"SV": torch.zeros((1, 1)), 
                                                                                 "r": 0.05, "ah": .07, "kappa": 2,},
                              Comparator.EQ,
                              "0", {},
                              label="q_min")
pde_model.add_equation(r"$\iota = \frac{q^2-1}{ 2 * \kappa}$")
pde_model.add_equation(r"$\sigma_t^q = \frac{\sigma}{1 - \frac{1}{q} * \frac{\partial q}{\partial \eta} * (\psi - \eta)} - \sigma$")
pde_model.add_equation(r"$\sigma_t^\eta = \frac{\psi - \eta}{\eta} * (\sigma + \sigma_t^q)$")
pde_model.add_equation(r"$\mu_t^\eta = (\sigma_t^\eta)^2 + \frac{a - \iota}{q} + (1-\psi) * (\underline{\delta} - \delta) - \rho$")

pde_model.add_constraint("psi", Comparator.LEQ, "1")
pde_model.add_endog_equation(r"$(\sigma + \sigma_t^q) ^2 * (\psi / \eta - (1-\psi) / (1-\eta)) = \frac{a - \underline{a}}{q} + \underline{\delta} - \delta$")
pde_model.add_endog_equation(r"$(r*(1-\eta) + \rho * \eta) * q = \psi * a + (1-\psi) * \underline{a} - \iota$")
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
pde_model.plot_vars(["q", "psi",
                    r"$\sigma_t^q$",
                     r"$\eta\sigma^\eta = \eta*\sigma_t^\eta$",
                     r"$\eta\mu^\eta = \eta*\mu_t^\eta$",
                     "er = psi/e*(sig+sigq)**2"], ncols=3)
```