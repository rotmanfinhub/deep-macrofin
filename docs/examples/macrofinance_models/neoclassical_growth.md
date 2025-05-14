# Neoclassical Growth Problem

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/macro_problems/neoclassical-growth/ncg_timestep.ipynb" target="_blank">ncg_timestep.ipynb</a>. 

Finite difference method is adapted from [https://benjaminmoll.com/codes/](https://benjaminmoll.com/codes/) HJB_NGM.m

## Problem Setup

Let $k$ be the state variable (capital price) and $V$ the value function. The HJB equation is 

$$
\rho V = \sup \left\{u(c) + \partial_k V (A k^\alpha - \delta k - c)\right\},
$$

where $u(c) = \begin{cases}
\log(c), \gamma=1\\
\frac{c^{1-\gamma}}{1-\gamma}, \text{ else}
\end{cases}$. The optimality condition is

$$
c^{-\gamma} = \partial_k V
$$

The steady state capital price is $k_{ss} = \left(\frac{\alpha}{\rho + \delta}\right)^{\frac{1}{1-\alpha}}$.

We apply timestepping sheme:

$$
\rho V = \partial_t V + u(c) + \partial_k V (A k^\alpha - \delta k - c)
$$

The full set of equations and constraints are:

$$
\begin{align*}
u(c) &= \begin{cases}
\log(c), \gamma=1\\
\frac{c^{1-\gamma}}{1-\gamma}, \text{ else}
\end{cases}\\
c^{-\gamma} &= \frac{\partial V}{\partial a}\\
\partial_k c &\geq 0\\
\rho V &= \partial_t V + u(c) + \partial_k V (A k^\alpha - \delta k - c)
\end{align*}
$$



### Parameter and Variable Definitions
| Parameter | Definition | Value |
|:---:|:---:|:---:|
|$\gamma$ | relative risk aversion| $\gamma=2$ |
|$\alpha$ | return to scale | $\alpha=0.3$ |
|$\delta$ | capital depreciation | $\delta=0.05$ |
|$A$ | productivity | $A=1$ |

| Type | Definition |
|:---:|:---:|
|State Variables | $k\in [0.01 k_{ss}, 2 k_{ss}]$|
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
from deep_macrofin import (Comparator, OptimizerType, PDEModel,
                           PDEModelTimeStep, set_seeds)
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

3. Construct the model with constant initial guess
```py
kss = (params["alpha"] / (params["rho"] + params["delta"])) ** (1 / (1 - params["alpha"]))
ckss = kss**params["alpha"] - params["delta"] * kss
set_seeds(seed)
model = PDEModelTimeStep("ncg", config=training_config)
model.set_state(["k"], {"k": [0.01 * kss, 2 * kss]}) #  
model.add_params(params)
model.add_endog("V", config=model_configs["V"])
model.add_endog("c", config=model_configs["c"])
if params["gamma"] == 1:
    endog_cond = torch.log(torch.tensor(ckss, dtype=torch.float32, device=model.device))/params["rho"]
    utility_eq = "u=log(c)"
else:
    endog_cond = ckss**(1-params["gamma"]) / ((1-params["gamma"]) * params["rho"])
    utility_eq = "u=c**(1-gamma)/(1-gamma)"

ss_bd = torch.zeros((100, 2), device=model.device)
ss_bd[:, 0] = kss
ss_bd[:, 1] = torch.linspace(0, 1, 100, device=model.device)
model.add_endog_condition("V", 
                        "V(SV)", 
                        {"SV": ss_bd},
                        Comparator.EQ,
                        "ec", {"ec": endog_cond},
                        label="v1")
model.add_endog_condition("c", 
                        "c(SV)", 
                        {"SV": ss_bd},
                        Comparator.EQ,
                        "kss**alpha - delta * kss", params | {"kss": kss},
                        label="c1")
model.add_equation("s=k**alpha - delta * k - c")
model.add_equation(utility_eq)
model.add_endog_equation("c**(-gamma)=V_k")
model.add_constraint("c_k", Comparator.GEQ, "0")
model.add_hjb_equation("V_t + u+ V_k * s-rho*V")
model.set_initial_guess(init_guess)
```

4. Train the model with specific parameters
```py
PARAMS = {
        "gamma": 2, # Risk aversion
        "alpha": 0.3, # Returns to scale
        "delta": 0.05, # Capital depreciation
        "rho": 0.05, # Discount rate
        "A": 1, # Productivity
    }
kss = (PARAMS["alpha"] / (PARAMS["rho"] + PARAMS["delta"])) ** (1 / (1 - PARAMS["alpha"]))

TRAINING_CONFIGS = {"num_outer_iterations": 20, "num_inner_iterations": 3000,  
        "time_batch_size": 4, "optimizer_type": OptimizerType.Adam}

MODEL_CONFIGS = {
    "V": {"hidden_units": [64] * 4},
    "c": {"hidden_units": [32] * 4, "positive": True},
}
model.train_model(BASE_DIR, f"model.pt", True)
model.eval_model(True)
```

5. Plot the solutions, comparing with finite difference solution.
```py
k_fd = fd_res["k"]
v_fd = fd_res["v"]
c_fd = fd_res["c"]
idx = k_fd > 0.01 * kss
k_fd = k_fd[idx]
v_fd = v_fd[idx]
c_fd = c_fd[idx]

SV = torch.zeros((k_fd.shape[0], 2), device=model.device)
SV[:, 0] = torch.tensor(k_fd, device=model.device, dtype=torch.float32)
for i, sv_name in enumerate(model.state_variables):
    model.variable_val_dict[sv_name] = SV[:, i:i+1]
model.variable_val_dict["SV"] = SV
model.update_variables(SV)
V_model = model.variable_val_dict["V"].detach().cpu().numpy().reshape(-1)
c_model = model.variable_val_dict["c"].detach().cpu().numpy().reshape(-1)

fig, ax = plt.subplots(1, 2, figsize=(20, 10))
ax[0].plot(k_fd, v_fd, linestyle="-.", color="red", label="Finite Difference")
ax[0].plot(k_fd, V_model, color="blue", label=f"Deep-MacroFin")
ax[0].legend()
ax[0].set_xlabel("$k$")
ax[0].set_ylabel("$V(k)$")

ax[1].plot(k_fd, c_fd, linestyle="-.", color="red", label="Finite Difference")
ax[1].plot(k_fd, c_model, color="blue", label=f"Deep-MacroFin")
ax[1].legend()
ax[1].set_xlabel("$k$")
ax[1].set_ylabel("$c(k)$")
plt.tight_layout()
plt.show()
```