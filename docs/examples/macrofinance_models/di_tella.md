# Di Tella (2017)

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/pymacrofin_eg/ditella_with_investment.ipynb" target="_blank">ditella_with_investment.ipynb</a>.

## Problem Setup
This is from <a href="https://www.journals.uchicago.edu/doi/10.1086/694290" target="_blank">Di Tella 2017</a>[^1]

[^1]: Sebastian Di Tella, *"Uncertainty Shocks and Balance Sheet Recessions"*, Journal of Political Economy, 125(6): 2038-2081, 2017

### Parameter and Variable Definitions
| Parameter | Definition | Value |
|:---:|:---:|:---:|
|$a$ | relative risk aversion| $a=1$ |
|$\sigma$ | volatility of TFP shocks | $\sigma=0.0125$ |
|$\lambda$ | mean reversion coefficient for idiosyncratic risk | $\lambda=1.38$ |
|$\bar{v}$ | long-run mean of idiosyncratic risk | $\bar{v}=0.25$ |
|$\bar{\sigma_v}$ | idiosyncratic volatility of capital on aggregate risk | $\bar{\sigma_v}=-0.17$ |
|$\rho$ | discount rate |$\rho=0.0665$ |
|$\gamma$ | risk aversion rate | $\gamma=5$ |
|$\psi$ | inverse of elasticity of intertemporal substitution | $\psi=0.5$ s.t. EIS=2 |
|$\tau$ | Poisson retirement rate for experts | $\tau=1.15$ |
|$\phi$ | moral hazzard | $\phi=0.2$ |
|$A$ | second order coefficient for investment function | $A=53$ |
|$B$ | first order coefficient for investment function | $B=-0.8668571428571438$ |
|$\delta$ | shift for investment function | $\delta=0.05$ |

| Type | Definition |
|:---:|:---:|
|State Variables | $(x, v)\in [0.05, 0.95]\times [0.05,0.95]$ |
|Agents | $\xi$ (experts), $\zeta$ (households) |
|Endogenous Variables | $p$ (price), $r$ (risk-free rate) |

> Note: For an ideal model, $(x,v)\in (0,1)\times (0,\infty)$, we use $[0.05, 0.95]\times [0.05,0.95]$ to be consistent with the original paper.

### Equations

$$g = \frac{1}{2A}(p - B) - \delta$$

$$\iota = A(g+\delta)^2 + B(g+\delta)$$

$$\mu_v = \lambda(\bar{v} - v)$$

$$\sigma_v = \bar{\sigma_v}\sqrt{v} $$

$$\hat{e} = \rho^{1/\psi}\xi^{(\psi-1)/\psi}$$

$$\hat{c} = \rho^{1/\psi}\zeta^{(\psi-1)/\psi}$$

$$\sigma_{x,1} = (1-x)x\frac{1-\gamma}{\gamma}\left( \frac{1}{\xi}\frac{\partial \xi}{\partial v} - \frac{1}{\zeta}\frac{\partial \zeta}{\partial v} \right)$$

$$\sigma_{x,2} = 1 - (1-x)x\frac{1-\gamma}{\gamma}\left( \frac{1}{\xi}\frac{\partial \xi}{\partial x} - \frac{1}{\zeta}\frac{\partial \zeta}{\partial x} \right)$$

$$\sigma_x = \frac{\sigma_{x,1}}{\sigma_{x,2}}\sigma_v$$

$$\sigma_p = \frac{1}{p}\left( \frac{\partial p}{\partial v}\sigma_v + \frac{\partial p}{\partial x}\sigma_x \right)$$

$$\sigma_\xi = \frac{1}{\xi}\left( \frac{\partial \xi}{\partial v}\sigma_v + \frac{\partial \xi}{\partial x}\sigma_x \right)$$

$$\sigma_\zeta = \frac{1}{\zeta}\left( \frac{\partial \zeta}{\partial v}\sigma_v + \frac{\partial \zeta}{\partial x}\sigma_x \right)$$

$$\sigma_n = \sigma + \sigma_p + \frac{\sigma_x}{x}$$

$$\pi = \gamma\sigma_n + (\gamma-1)\sigma_\xi$$

$$\sigma_w = \frac{\pi}{\gamma} - \frac{\gamma-1}{\gamma} \sigma_\zeta$$

$$\mu_w = r + \pi\sigma_w$$

$$\mu_n = r + \frac{\gamma}{x^2}(\phi v)^2 + \pi\sigma_n$$

$$\tilde{\sigma_n} = \frac{\phi}{x}v$$

$$\mu_x = x\left(\mu_n - \hat{e} - \tau + \frac{a-\iota}{p} - r - \pi(\sigma+\sigma_p) - \frac{\gamma}{x}(\phi v)^2 + (\sigma + \sigma_p)^2 - \sigma_n(\sigma + \sigma_p)\right)$$

$$\mu_p = \frac{1}{p}\left( \mu_v\frac{\partial p}{\partial v} + \mu_x\frac{\partial p}{\partial x} + \frac{1}{2}\left( \sigma_v^2\frac{\partial^2 p}{\partial v^2} + 2\sigma_v\sigma_x\frac{\partial^2 p}{\partial v \partial x} + \sigma_x^2\frac{\partial^2 p}{\partial x^2} \right)\right)$$

$$\mu_\xi = \frac{1}{\xi}\left( \mu_v\frac{\partial \xi}{\partial v} + \mu_x\frac{\partial \xi}{\partial x} + \frac{1}{2}\left( \sigma_v^2\frac{\partial^2 \xi}{\partial v^2} + 2\sigma_v\sigma_x\frac{\partial^2 \xi}{\partial v \partial x} + \sigma_x^2\frac{\partial^2 \xi}{\partial x^2} \right)\right)$$

$$\mu_\zeta = \frac{1}{\zeta}\left( \mu_v\frac{\partial \zeta}{\partial v} + \mu_x\frac{\partial \zeta}{\partial x} + \frac{1}{2}\left( \sigma_v^2\frac{\partial^2 \zeta}{\partial v^2} + 2\sigma_v\sigma_x\frac{\partial^2 \zeta}{\partial v \partial x} + \sigma_x^2\frac{\partial^2 \zeta}{\partial x^2} \right)\right)$$


### Endogenous Equations
$$a - \iota = p(\hat{e}x + \hat{c}(1-x)) $$

$$\sigma + \sigma_p = \sigma_nx + \sigma_w(1-x)$$

$$\frac{a-\iota}{p} + g + \mu_p + \sigma\sigma_p - r = (\sigma + \sigma_p)\pi + \gamma\frac{1}{x}(\phi v)^2$$


### HJB Equations
$$\frac{\rho}{1-\psi} = \max \left\{\frac{\hat{e}^{1-\psi}}{1-\psi}\rho\xi^{\psi-1} + \frac{\tau}{1-\gamma}\left(\left(\frac{\zeta}{\xi} \right)^{1-\gamma}-1 \right) + \mu_n - \hat{e} + \mu_\xi - \frac{\gamma}{2}\left( \sigma_n^2 + \sigma_\xi^2 - 2\frac{1-\gamma}{\gamma}\sigma_n\sigma_\xi + \tilde{\sigma_n}^2 \right)\right\}$$

$$\frac{\rho}{1-\psi} = \max \left\{\frac{\hat{c}^{1-\psi}}{1-\psi}\rho\zeta^{\psi-1} + \mu_w - \hat{c} + \mu_\zeta - \frac{\gamma}{2}\left( \sigma_w^2 + \sigma_\zeta^2 - 2\frac{1-\gamma}{\gamma}\sigma_w\sigma_\zeta \right) \right\}$$

## Implementation

1. Import necessary packages
```py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from deep_macrofin import PDEModel
from deep_macrofin import ActivationType, OptimizerType, SamplingMethod, plot_loss_df, set_seeds
```

2. Define LaTex-variable mapping so that we can parse LaTex equations
```py
latex_var_mapping = {
    # variables
    r"\iota": "iota",
    r"\hat{e}": "e_hat",
    r"\hat{c}": "c_hat",
    r"\sigma_{x,1}": "sigxtop",
    r"\sigma_{x,2}": "sigxbot",
    r"\sigma_x": "sigx",
    r"\sigma_p": "sigp",
    r"\sigma_\xi": "sigxi",
    r"\sigma_\zeta": "sigzeta",
    r"\tilde{\sigma_n}": "signtilde",
    r"\sigma_n": "sign",
    r"\pi": "signxi",
    r"\sigma_w": "sigw",
    r"\mu_n": "mun",
    r"\mu_x": "mux",
    r"\mu_p": "mup",
    r"\mu_\xi": "muxi",
    r"\mu_\zeta": "muzeta",
    r"\mu_w": "muw",

    # agents
    r"\xi": "xi",
    r"\zeta": "zeta",

    # constants
    r"\bar{\sigma_v}": "sigv_mean",
    r"\sigma_v": "sigv",
    r"\mu_v": "muv",
    r"\sigma": "sigma",
    r"\lambda": "lbd",
    r"\bar{v}": "v_mean",
    r"\rho": "rho",
    r"\gamma": "gamma",
    r"\psi": "psi",
    r"\tau": "tau",
    r"\delta": "delta",
    r"\phi": "phi",
}
```

3. Define the problem. We train on a fixed grid of $50\times 50$ points.
```py
set_seeds(0)
pde_model = PDEModel("ditella", 
                     {"batch_size": 50, "num_epochs": 10000, "loss_log_interval": 100, 
                      "optimizer_type": OptimizerType.Adam, "sampling_method": SamplingMethod.FixedGrid}, 
                     latex_var_mapping)
pde_model.set_state(["x", "v"], {"x": [0.05, 0.95], "v": [0.05, 0.95]})
pde_model.add_agents(["xi", "zeta"], 
                     {"xi": {
                         "positive": True, 
                        }, 
                      "zeta": {
                          "positive": True, 
                          }
                     })
pde_model.add_endogs(["p", "r"], 
                     {"p": {
                         "positive": True, 
                         },
                     })
pde_model.add_params({
    "a": 1,
    "sigma": 0.0125,
    "lbd": 1.38,
    "v_mean": 0.25,
    "sigv_mean": -0.17,
    "rho": 0.0665,
    "gamma": 5,
    "psi": 0.5,
    "tau": 1.15,
    "phi": 0.2,

    "A": 53.2,
    "B": -0.8668571428571438,
    "delta": 0.05,
})
pde_model.add_equation(r"$g &= \frac{1}{2*A} * (p - B) - \delta$") # g &= \frac{1}{2*A} * (p - B) - \delta
pde_model.add_equation(r"$\iota &= A * (g+\delta)^2 + B * (g+\delta)$") # \iota &= A * (g+\delta)^2 + B * (g+\delta)
pde_model.add_equation(r"$\mu_v &= \lambda * (\bar{v} - v)$")
pde_model.add_equation(r"$\sigma_v &= \bar{\sigma_v} * \sqrt{v}$")
pde_model.add_equation(r"$\hat{e} &= \rho^{1/\psi} * \xi^{(\psi-1)/\psi}$")
pde_model.add_equation(r"$\hat{c} &= \rho^{1/\psi} * \zeta^{(\psi-1)/\psi}$")
pde_model.add_equation(r"$\sigma_{x,1} &= (1-x) * x * \frac{1-\gamma}{\gamma} * \left( \frac{1}{\xi} * \frac{\partial \xi}{\partial v} - \frac{1}{\zeta} * \frac{\partial \zeta}{\partial v} \right)$")
pde_model.add_equation(r"$\sigma_{x,2} &= 1 - (1-x) * x * \frac{1-\gamma}{\gamma} * \left( \frac{1}{\xi} * \frac{\partial \xi}{\partial x} - \frac{1}{\zeta} * \frac{\partial \zeta}{\partial x} \right)$")
pde_model.add_equation(r"$\sigma_x &= \frac{\sigma_{x,1}}{\sigma_{x,2}} * \sigma_v$")
pde_model.add_equation(r"$\sigma_p &= \frac{1}{p} * \left( \frac{\partial p}{\partial v} * \sigma_v + \frac{\partial p}{\partial x} * \sigma_x \right)$")
pde_model.add_equation(r"$\sigma_\xi &= \frac{1}{\xi} * \left( \frac{\partial \xi}{\partial v} * \sigma_v + \frac{\partial \xi}{\partial x} * \sigma_x \right)$")
pde_model.add_equation(r"$\sigma_\zeta &= \frac{1}{\zeta} * \left( \frac{\partial \zeta}{\partial v} * \sigma_v + \frac{\partial \zeta}{\partial x} * \sigma_x \right)$")
pde_model.add_equation(r"$\sigma_n &= \sigma + \sigma_p + \frac{\sigma_x}{x}$")
pde_model.add_equation(r"$\pi &= \gamma * \sigma_n + (\gamma-1) * \sigma_\xi$")
pde_model.add_equation(r"$\sigma_w &= \frac{\pi}{\gamma} - \frac{\gamma-1}{\gamma} *  \sigma_\zeta$")
pde_model.add_equation(r"$\mu_w &= r + \pi * \sigma_w$")
pde_model.add_equation(r"$\mu_n &= r + \frac{\gamma}{x^2} * (\phi * v)^2 + \pi * \sigma_n$")
pde_model.add_equation(r"$\tilde{\sigma_n} &= \frac{\phi}{x} * v$")
pde_model.add_equation(r"$\mu_x &= x * \left(\mu_n - \hat{e} - \tau + \frac{a-\iota}{p} - r - \pi * (\sigma+\sigma_p) - \frac{\gamma}{x} * (\phi * v)^2 + (\sigma + \sigma_p)^2 - \sigma_n * (\sigma + \sigma_p)\right)$")
pde_model.add_equation(r"$\mu_p &= \frac{1}{p} * \left( \mu_v * \frac{\partial p}{\partial v} + \mu_x * \frac{\partial p}{\partial x} + \frac{1}{2} * \left( \sigma_v^2 * \frac{\partial^2 p}{\partial v^2} + 2 * \sigma_v * \sigma_x * \frac{\partial^2 p}{\partial v \partial x} + \sigma_x^2 * \frac{\partial^2 p}{\partial x^2} \right)\right)$")
pde_model.add_equation(r"$\mu_\xi &= \frac{1}{\xi} * \left( \mu_v * \frac{\partial \xi}{\partial v} + \mu_x * \frac{\partial \xi}{\partial x} + \frac{1}{2} * \left( \sigma_v^2 * \frac{\partial^2 \xi}{\partial v^2} + 2 * \sigma_v * \sigma_x * \frac{\partial^2 \xi}{\partial v \partial x} + \sigma_x^2 * \frac{\partial^2 \xi}{\partial x^2} \right)\right)$")
pde_model.add_equation(r"$\mu_\zeta &= \frac{1}{\zeta} * \left( \mu_v * \frac{\partial \zeta}{\partial v} + \mu_x * \frac{\partial \zeta}{\partial x} + \frac{1}{2} * \left( \sigma_v^2 * \frac{\partial^2 \zeta}{\partial v^2} + 2 * \sigma_v * \sigma_x * \frac{\partial^2 \zeta}{\partial v \partial x} + \sigma_x^2 * \frac{\partial^2 \zeta}{\partial x^2} \right)\right)$")

pde_model.add_endog_equation(r"$a - \iota &= p * (\hat{e} * x + \hat{c} * (1-x))$")
pde_model.add_endog_equation(r"$\sigma + \sigma_p &= \sigma_n * x + \sigma_w * (1-x)$")
pde_model.add_endog_equation(r"$\frac{a-\iota}{p} + g + \mu_p + \sigma * \sigma_p - r &= (\sigma + \sigma_p) * \pi + \gamma * \frac{1}{x} * (\phi * v)^2$")

pde_model.add_hjb_equation(r"$\frac{\hat{e}^{1-\psi}}{1-\psi} * \rho * \xi^{\psi-1} + \frac{\tau}{1-\gamma} * \left(\left(\frac{\zeta}{\xi} \right)^{1-\gamma}-1 \right) + \mu_n - \hat{e} + \mu_\xi - \frac{\gamma}{2} * \left( \sigma_n^2 + \sigma_\xi^2 - 2 * \frac{1-\gamma}{\gamma} * \sigma_n * \sigma_\xi + \tilde{\sigma_n}^2 \right) - \frac{\rho}{1-\psi}$")
pde_model.add_hjb_equation(r"$\frac{\hat{c}^{1-\psi}}{1-\psi} * \rho * \zeta^{\psi-1} + \mu_w - \hat{c} + \mu_\zeta - \frac{\gamma}{2} * \left( \sigma_w^2 + \sigma_\zeta^2 - 2 * \frac{1-\gamma}{\gamma} * \sigma_w * \sigma_\zeta \right) - \frac{\rho}{1-\psi}$")
print(pde_model)
```

4. Train and evaluate the best model with min loss
```py
pde_model.train_model("./models/ditella_with_investments", "model.pt", True)
pde_model.load_model(torch.load("./models/ditella_with_investments/model_best.pt"))
pde_model.eval_model(True)
```

5. Plot the solutions, with additional variables defined by equations.
```py
pde_model.plot_vars([r"$\xi$", r"$\zeta$", "p", 
                     r"$\sigma+\sigma_p = \sigma + \sigma_p$", r"$\pi$", "r"], ncols=3)
```

6. Plot variables in 1D. Specifically, we plot $p$ (price of capital), $\sigma_x$ (volatility of $x$), $\Omega = \frac{\xi}{\zeta}$ (relative investment opportunities), $\sigma+\sigma_p$ (aggregate risk), $\pi$ (price of risk), and $r$ (risk-free rate) as functions of $v$ for $x=0.05, 0.1, 0.2$, and as functions of $x$ for $v=0.1, 0.25, 0.6$.
```py
fig, ax = plt.subplots(2, 3, figsize=(18, 12))
for x_val, linestyle in [(0.05, "-"), (0.1, ":"), (0.2, "--")]:
    sv = torch.ones((100, 2), device=pde_model.device) * x_val
    sv[:, 1] = torch.linspace(0.05, 0.95, 100)
    for i, sv_name in enumerate(pde_model.state_variables):
        pde_model.variable_val_dict[sv_name] = sv[:, i:i+1]
    pde_model.update_variables(sv)
    p = pde_model.variable_val_dict["p"]
    sigx = pde_model.variable_val_dict["sigx"]
    omega = pde_model.variable_val_dict["xi"] / pde_model.variable_val_dict["zeta"]
    ax[0][0].plot(sv[:, 1].detach().cpu().numpy(), p.detach().cpu().numpy().reshape(-1), linestyle=linestyle, label=f"x={x_val}")
    ax[0][1].plot(sv[:, 1].detach().cpu().numpy(), sigx.detach().cpu().numpy().reshape(-1), linestyle=linestyle, label=f"x={x_val}")
    ax[0][2].plot(sv[:, 1].detach().cpu().numpy(), omega.detach().cpu().numpy().reshape(-1), linestyle=linestyle, label=f"x={x_val}")
    ax[0][0].set_title(r"p")
    ax[0][1].set_title(r"$\sigma_x$")
    ax[0][2].set_title(r"$\Omega = \xi/\zeta$")
    ax[0][0].legend()
    ax[0][1].legend()
    ax[0][2].legend()

for v_val, linestyle in [(0.1, "-"), (0.25, ":"), (0.6, "--")]:
    sv = torch.ones((100, 2), device=pde_model.device) * v_val
    sv[:, 0] = torch.linspace(0.05, 0.95, 100)
    for i, sv_name in enumerate(pde_model.state_variables):
        pde_model.variable_val_dict[sv_name] = sv[:, i:i+1]
    pde_model.update_variables(sv)
    p = pde_model.variable_val_dict["p"]
    sigx = pde_model.variable_val_dict["sigx"]
    omega = pde_model.variable_val_dict["xi"] / pde_model.variable_val_dict["zeta"]
    ax[1][0].plot(sv[:, 0].detach().cpu().numpy(), p.detach().cpu().numpy().reshape(-1), linestyle=linestyle, label=f"v={v_val}")
    ax[1][1].plot(sv[:, 0].detach().cpu().numpy(), sigx.detach().cpu().numpy().reshape(-1), linestyle=linestyle, label=f"v={v_val}")
    ax[1][2].plot(sv[:, 0].detach().cpu().numpy(), omega.detach().cpu().numpy().reshape(-1), linestyle=linestyle, label=f"v={v_val}")
    ax[1][0].set_title(r"p")
    ax[1][1].set_title(r"$\sigma_x$")
    ax[1][2].set_title(r"$\Omega = \xi/\zeta$")
    ax[1][0].legend()
    ax[1][1].legend()
    ax[1][2].legend()
plt.subplots_adjust()
plt.show()

fig, ax = plt.subplots(2, 3, figsize=(18, 12))
for x_val, linestyle in [(0.05, "-"), (0.1, ":"), (0.2, "--")]:
    sv = torch.ones((100, 2), device=pde_model.device) * x_val
    sv[:, 1] = torch.linspace(0.05, 0.95, 100)
    for i, sv_name in enumerate(pde_model.state_variables):
        pde_model.variable_val_dict[sv_name] = sv[:, i:i+1]
    pde_model.update_variables(sv)
    sigsigp = pde_model.variable_val_dict["sigma"] + pde_model.variable_val_dict["sigp"]
    pi = pde_model.variable_val_dict["signxi"]
    r = pde_model.variable_val_dict["r"]
    ax[0][0].plot(sv[:, 1].detach().cpu().numpy(), sigsigp.detach().cpu().numpy().reshape(-1), linestyle=linestyle, label=f"x={x_val}")
    ax[0][1].plot(sv[:, 1].detach().cpu().numpy(), pi.detach().cpu().numpy().reshape(-1), linestyle=linestyle, label=f"x={x_val}")
    ax[0][2].plot(sv[:, 1].detach().cpu().numpy(), r.detach().cpu().numpy().reshape(-1), linestyle=linestyle, label=f"x={x_val}")
    ax[0][0].set_title(r"$\sigma+\sigma_p$")
    ax[0][1].set_title(r"$\pi$")
    ax[0][2].set_title("r")
    ax[0][0].legend()
    ax[0][1].legend()
    ax[0][2].legend()
for v_val, linestyle in [(0.1, "-"), (0.25, ":"), (0.6, "--")]:
    sv = torch.ones((100, 2), device=pde_model.device) * v_val
    sv[:, 0] = torch.linspace(0.05, 0.95, 100)
    for i, sv_name in enumerate(pde_model.state_variables):
        pde_model.variable_val_dict[sv_name] = sv[:, i:i+1]
    pde_model.update_variables(sv)
    sigsigp = pde_model.variable_val_dict["sigma"] + pde_model.variable_val_dict["sigp"]
    pi = pde_model.variable_val_dict["signxi"]
    r = pde_model.variable_val_dict["r"]
    ax[1][0].plot(sv[:, 0].detach().cpu().numpy(), sigsigp.detach().cpu().numpy().reshape(-1), linestyle=linestyle, label=f"v={v_val}")
    ax[1][1].plot(sv[:, 0].detach().cpu().numpy(), pi.detach().cpu().numpy().reshape(-1), linestyle=linestyle, label=f"v={v_val}")
    ax[1][2].plot(sv[:, 0].detach().cpu().numpy(), r.detach().cpu().numpy().reshape(-1), linestyle=linestyle, label=f"v={v_val}")
    ax[1][0].set_title(r"$\sigma+\sigma_p$")
    ax[1][1].set_title(r"$\pi$")
    ax[1][2].set_title("r")
    ax[1][0].legend()
    ax[1][1].legend()
    ax[1][2].legend()
plt.subplots_adjust()
plt.show()
```