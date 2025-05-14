# Bewley-Huggett-Aiyagari Problem

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/macro_problems/bewley-huggett-aiyagari/bha_timestep.ipynb" target="_blank">bha_timestep.ipynb</a>. 

Finite difference method is adapted from [https://benjaminmoll.com/codes/](https://benjaminmoll.com/codes/) HJB_simple.m

The timestepping version will be more stable than the basic training.

## Problem Setup

Let $a$ be the state variable and $V$ the value function. The HJB equation is 

$$
\rho V = \sup \left\{u(c) + \partial_a V (ra+y-c)\right\},
$$

where $u(c) = \begin{cases}
\log(c), \gamma=1\\
\frac{c^{1-\gamma}}{1-\gamma}, \text{ else}
\end{cases}$. The optimality condition is

$$
c^{-\gamma} = \partial_a V
$$

We apply timestepping sheme:

$$
\rho V = \partial_t V + u(c) + \partial_a V (ra+y-c)
$$

The full set of equations and constraints are:

$$
\begin{align*}
u(c) &= \begin{cases}
\log(c), \gamma=1\\
\frac{c^{1-\gamma}}{1-\gamma}, \text{ else}
\end{cases}\\
c^{-\gamma} &= \frac{\partial V}{\partial a}\\
\frac{\partial c}{\partial a} &\geq 0\\
\rho V &= \partial_t V + u(c) + \partial_a V (ra+y-c)
\end{align*}
$$



### Parameter and Variable Definitions
| Parameter | Definition | Value |
|:---:|:---:|:---:|
|$\gamma$ | relative risk aversion| $\gamma=2$ |
|$r$ | interest rate | $r=0.045$ |
|$y$ | income | $y=0.1$ |
|$\rho$ | discount rate | $\rho=0.05$ |

| Type | Definition |
|:---:|:---:|
|State Variables | $a\in[0,1]$|
|Endogenous Variables | $V$, $c$ |

## Implementation

1. Import necessary packages
```py
import os
import time
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from deep_macrofin import (ActivationType, Comparator, LossReductionMethod, OptimizerType, PDEModelTimeStep, 
                           SamplingMethod, set_seeds)
```

2. Define utility function
```py
def utility(c, gamma):
    if gamma == 1:
        return np.log(c)
    else:
        return c**(1-gamma)/(1-gamma)

def utility_deriv(c, gamma):
    if gamma == 1:
        return 1 / c
    else:
        return c**(-gamma)

def inverse_marginal_deriv(dV, gamma):
    if gamma == 1:
        return 1 / dV
    else:
        return dV**(-1/gamma)
```

3. Construct the model with functional initial guess
```py
set_seeds(seed)
model = PDEModelTimeStep("bha", config=training_config)
model.set_state(["a"], {"a": [0.01, 1.0]}) #  
model.add_params(params)
model.add_endog("V", config=model_configs["V"])
model.add_endog("c", config=model_configs["c"])
if params["gamma"] == 1:
    endog_cond = torch.log(torch.tensor(params["y"], dtype=torch.float32, device=model.device))
    utility_eq = "u=log(c)"
else:
    endog_cond = params["y"]**(1-params["gamma"]) / ((1-params["gamma"]) * params["rho"])
    utility_eq = "u=c**(1-gamma)/(1-gamma)"
zero_a_bd = torch.zeros((100, 2), device=model.device)
zero_a_bd[:, 1] = torch.linspace(0, 1, 100)
model.add_endog_condition("V", 
                            "V(SV)", 
                            {"SV": zero_a_bd},
                            Comparator.EQ,
                            "ec", {"ec": endog_cond},
                            label="v1")
model.add_endog_condition("c", 
                            "c(SV)", 
                            {"SV": zero_a_bd},
                            Comparator.EQ,
                            "y", params,
                            label="c1")
model.add_equation("s=r*a+y-c")
model.add_equation(utility_eq)
model.add_endog_equation("c**(-gamma)=V_a")
model.add_constraint("c_a", Comparator.GEQ, "0")
model.add_hjb_equation("V_t + u+ V_a * s-rho*V")

def init_guess_c(SV, r, y):
    a = SV[:, :1]
    return (r*a + y).detach()
def init_guess_V(SV, r, y, gamma, rho):
    a = SV[:, :1]
    c = r*a + y
    return (c**(1-gamma)/((1-gamma)*rho)).detach()

model.set_initial_guess({
    "V": lambda SV: init_guess_V(SV, params["r"], params["y"], params["gamma"], params["rho"]),
    "c": lambda SV: init_guess_c(SV, params["r"], params["y"])
})
```

4. Train the model with specific parameters
```py
PARAMS = {
    "gamma": 2, # Risk aversion
    "r": 0.045, # interest rate
    "y": 0.1, # income
    "rho": 0.05, # Discount rate
}
TRAINING_CONFIGS = {
    "num_outer_iterations": 10, 
    "num_inner_iterations": 10000, 
    "time_batch_size": 4, 
    "optimizer_type": OptimizerType.Adam,
    "sampling_method": SamplingMethod.UniformRandom,
    "time_boundary_loss_reduction": LossReductionMethod.MSE,
}
MODEL_CONFIGS = {
    "V": {"hidden_units": [64] * 3},
    "c": {"hidden_units": [32] * 3, "positive": True, "activation_type": ActivationType.SiLU},
}
model.train_model(BASE_DIR, f"model.pt", True)
model.eval_model(True)
```

5. Plot the solutions, comparing with finite difference solution.
```py
a_fd = fd_res["a"]
v_fd = fd_res["v"]
c_fd = fd_res["c"]

SV = torch.zeros((a_fd.shape[0], 2), device=model.device)
SV[:, 0] = torch.tensor(a_fd, device=model.device, dtype=torch.float32)
for i, sv_name in enumerate(model.state_variables):
    model.variable_val_dict[sv_name] = SV[:, i:i+1]
model.variable_val_dict["SV"] = SV
model.update_variables(SV)
V_model = model.variable_val_dict["V"].detach().cpu().numpy().reshape(-1)
c_model = model.variable_val_dict["c"].detach().cpu().numpy().reshape(-1)

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].plot(a_fd, v_fd, linestyle="-.", color="red", label="Finite Difference")
ax[0].plot(a_fd, V_model, color="blue", label=f"Deep-MacroFin")
ax[0].legend()
ax[0].set_xlabel("$a$")
ax[0].set_ylabel("$V(a)$")

ax[1].plot(a_fd, c_fd, linestyle="-.", color="red", label="Finite Difference")
ax[1].plot(a_fd, c_model, color="blue", label=f"Deep-MacroFin")
ax[1].legend()
ax[1].set_xlabel("$a$")
ax[1].set_ylabel("$c(a)$")
plt.tight_layout()
plt.show()
```