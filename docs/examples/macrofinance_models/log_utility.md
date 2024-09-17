# Log Utility Problem

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/pymacrofin_eg/1d_problem_relu.ipynb" target="_blank">1d_problem.ipynb</a>. The split and merge solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/pymacrofin_eg/1d_problem_split.ipynb" target="_blank">1d_problem_split.ipynb</a>.

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

## Implementation (Split and Merge)

Instead of solving the problem on both regions as a whole, we can separately solve two problems and merge the solutions.

In the first region $\psi < 1$, we need to solve for both $q$ and $\psi$. The boundary conditions are

$$\begin{cases}
q(0)=-\kappa r + \sqrt{\kappa^2 r^2+1+2 \underline{a} \kappa}\\
q(1)=-\kappa \rho + \sqrt{\kappa^2 \rho^2+1+2 a \kappa}\\
\psi(0)=0\\
\psi(1)=1
\end{cases}$$


The equations to solve for equilibrium are

$$
\begin{cases}
(r(1-\eta) + \rho \eta) q = \psi a + (1-\psi) \underline{a} - \iota\\
(\sigma + \sigma_t^q) ^2 \frac{q (\psi - \eta)}{\eta (1-\eta)} = (a - \underline{a}) + (\underline{\delta} - \delta) q
\end{cases}
$$

In the second region $\psi = 1$, we only need to solve for $q$ with a single equation:

$$(r (1-\eta) + \rho \eta) q = a - \iota$$


1. Solve for region 1.
```py
set_seeds(0)
pde_model = PDEModel("BruSan14_log_utility", {"sampling_method": SamplingMethod.FixedGrid, "batch_size": 1000,
    "num_epochs": 20000, "loss_log_interval": 100, "optimizer_type": OptimizerType.Adam}, latex_var_mapping=latex_var_mapping)
pde_model.set_state(["e"], {"e": [0.001, 0.999]})
pde_model.add_endogs(["q", "psi"], configs={
    "q": {
        "positive": True,
        "activation_type": ActivationType.SiLU,
    },
    "psi": {
        "positive": True, 
        "activation_type": ActivationType.SiLU,
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
                              "q(SV)", 
                              {"SV": torch.zeros((1, 1))},
                              Comparator.EQ,
                              "-kappa*r + (kappa**2*r**2 + 1 + 2*ah*kappa)**0.5", {"r": 0.05, "ah": .07, "kappa": 2},
                              label="q_min", weight=100)
pde_model.add_endog_condition("q", 
                              "q(SV)", 
                              {"SV": torch.ones((1, 1))},
                              Comparator.EQ,
                              "-kappa*rho + (kappa**2*rho**2 + 1 + 2*a*kappa)**0.5", {"rho": 0.06, "a": .11, "kappa": 2},
                              label="q_max", weight=100)
pde_model.add_endog_condition("psi", 
                              "psi(SV)", 
                              {"SV": torch.zeros((1, 1))},
                              Comparator.EQ,
                              "0", {},
                              label="psi_min", weight=100)
pde_model.add_endog_condition("psi", 
                              "psi(SV)", 
                              {"SV": torch.ones((1, 1))},
                              Comparator.EQ,
                              "1", {},
                              label="psi_max", weight=100)

pde_model.add_equation(r"$\iota = \frac{q^2-1}{ 2 * \kappa}$")
pde_model.add_equation(r"$\sigma_t^q = \frac{\sigma}{1 - \frac{1}{q} * \frac{\partial q}{\partial \eta} * (\psi - \eta)} - \sigma$")
pde_model.add_equation(r"$\sigma_t^\eta = \frac{\psi - \eta}{\eta} * (\sigma + \sigma_t^q)$")
pde_model.add_equation(r"$\mu_t^\eta = (\sigma_t^\eta)^2 + \frac{a - \iota}{q} + (1-\psi) * (\underline{\delta} - \delta) - \rho$")

pde_model.add_endog_equation(r"$(r*(1-\eta) + \rho * \eta) * q = \psi * a + (1-\psi) * \underline{a} - \iota$", loss_reduction=LossReductionMethod.SSE)
pde_model.add_endog_equation(r"$(\sigma + \sigma_t^q) ^2 * \frac{q * (\psi - \eta)}{\eta * (1-\eta)} = (a - \underline{a}) + (\underline{\delta} - \delta) * q$", loss_reduction=LossReductionMethod.SSE, weight=2)
pde_model.train_model("./models/BruSan14_log_utility_relu_split", "region1.pt", True)
pde_model.load_model(torch.load("./models/BruSan14_log_utility_relu_split/region1_best.pt"))
pde_model.eval_model(True)
```

2. Compute and plot the results specifically for region 1. We can simply use `update_variables` function in `PDEModel` class to update all variables of interest.
```py
N = 1000
SV = torch.linspace(0, 1, N, device=pde_model.device).reshape(-1, 1)
x_plot = SV.detach().cpu().numpy().reshape(-1)
for i, sv_name in enumerate(pde_model.state_variables):
    pde_model.variable_val_dict[sv_name] = SV[:, i:i+1]
pde_model.update_variables(SV)
q_region1 = pde_model.variable_val_dict["q"].detach().cpu().numpy().reshape(-1)
psi_region1 = pde_model.variable_val_dict["psi"].detach().cpu().numpy().reshape(-1)
sigq_region1 = pde_model.variable_val_dict["sigq"].detach().cpu().numpy().reshape(-1)
esige_region1 = (SV * pde_model.variable_val_dict["sige"]).detach().cpu().numpy().reshape(-1)
emue_region1 = (SV * pde_model.variable_val_dict["mue"]).detach().cpu().numpy().reshape(-1)
er_region1 = (pde_model.variable_val_dict["psi"] / (SV * (pde_model.variable_val_dict["sig"] + pde_model.variable_val_dict["sigq"])**2)).detach().cpu().numpy().reshape(-1)

xlabel = "$\eta$"
plot_args = [
    {"y": q_region1, "ylabel": r"$q$", "title": r"$q$ vs. $\eta$"},
    {"y": psi_region1, "ylabel": r"$\psi$", "title": r"$\psi$ vs. $\eta$"},
    {"y": sigq_region1, "ylabel": r"$\sigma^q$", "title": r"$\sigma^q$ vs. $\eta$"},
    {"y": esige_region1, "ylabel": r"$\eta\sigma^{\eta}$", "title": r"$\eta\sigma^{\eta}$ vs. $\eta$"},
    {"y": emue_region1, "ylabel": r"$\eta\mu^{\eta}$", "title": r"$\eta\mu^{\eta}$ vs. $\eta$"},
    {"y": er_region1, "ylabel": r"$E[dr_t^k-dr_t]/dt$", "title": r"$E[dr_t^k-dr_t]/dt$ vs. $\eta$"},
]

fig, ax = plt.subplots(2, 3, figsize=(18, 12))
for i, plot_arg in enumerate(plot_args):
    row = i // 3
    col = i % 3
    curr_ax = ax[row, col]
    curr_ax.plot(x_plot, plot_arg["y"])
    curr_ax.set_xlabel(xlabel)
    curr_ax.set_ylabel(plot_arg["ylabel"])
    curr_ax.set_title(plot_arg["title"])
plt.tight_layout()
plt.show()
```

3. Solve for region 2.
```py
set_seeds(0)
pde_model = PDEModel("BruSan14_log_utility", {"sampling_method": SamplingMethod.FixedGrid, "batch_size": 1000,
    "num_epochs": 2000, "loss_log_interval": 100, "optimizer_type": OptimizerType.Adam}, latex_var_mapping=latex_var_mapping)
pde_model.set_state(["e"], {"e": [0.001, 0.999]})
pde_model.add_endogs(["q"], configs={
    "q": {
        "positive": True,
        "activation_type": ActivationType.SiLU,
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

pde_model.add_equation(r"$\iota = \frac{q^2-1}{ 2 * \kappa}$")
pde_model.add_equation(r"$\sigma_t^q = \frac{\sigma}{1 - \frac{1}{q} * \frac{\partial q}{\partial \eta} * (1 - \eta)} - \sigma$")
pde_model.add_equation(r"$\sigma_t^\eta = \frac{1 - \eta}{\eta} * (\sigma + \sigma_t^q)$")
pde_model.add_equation(r"$\mu_t^\eta = (\sigma_t^\eta)^2 + \frac{a - \iota}{q} - \rho$")

pde_model.add_endog_equation(r"$(r*(1-\eta) + \rho * \eta) * q = a - \iota$", loss_reduction=LossReductionMethod.SSE)

pde_model.train_model("./models/BruSan14_log_utility_relu_split", "region2.pt", True)
pde_model.load_model(torch.load("./models/BruSan14_log_utility_relu_split/region2_best.pt"))
pde_model.eval_model(True)
```

4. Compute and plot the results specifically for region 2.
```py
N = 1000
SV = torch.linspace(0, 1, N, device=pde_model.device).reshape(-1, 1)
x_plot = SV.detach().cpu().numpy().reshape(-1)
for i, sv_name in enumerate(pde_model.state_variables):
    pde_model.variable_val_dict[sv_name] = SV[:, i:i+1]
pde_model.update_variables(SV)
q_region2 = pde_model.variable_val_dict["q"].detach().cpu().numpy().reshape(-1)
sigq_region2 = pde_model.variable_val_dict["sigq"].detach().cpu().numpy().reshape(-1)
esige_region2 = (SV * pde_model.variable_val_dict["sige"]).detach().cpu().numpy().reshape(-1)
emue_region2 = (SV * pde_model.variable_val_dict["mue"]).detach().cpu().numpy().reshape(-1)
er_region2 = (1 / (SV * (pde_model.variable_val_dict["sig"] + pde_model.variable_val_dict["sigq"])**2)).detach().cpu().numpy().reshape(-1)

xlabel = "$\eta$"
plot_args = [
    {"y": q_region2, "ylabel": r"$q$", "title": r"$q$ vs. $\eta$"},
    {"y": np.ones_like(x_plot), "ylabel": r"$\psi$", "title": r"$\psi$ vs. $\eta$"},
    {"y": sigq_region2, "ylabel": r"$\sigma^q$", "title": r"$\sigma^q$ vs. $\eta$"},
    {"y": esige_region2, "ylabel": r"$\eta\sigma^{\eta}$", "title": r"$\eta\sigma^{\eta}$ vs. $\eta$"},
    {"y": emue_region2, "ylabel": r"$\eta\mu^{\eta}$", "title": r"$\eta\mu^{\eta}$ vs. $\eta$"},
    {"y": er_region2, "ylabel": r"$E[dr_t^k-dr_t]/dt$", "title": r"$E[dr_t^k-dr_t]/dt$ vs. $\eta$"},
]

fig, ax = plt.subplots(2, 3, figsize=(18, 12))
for i, plot_arg in enumerate(plot_args):
    row = i // 3
    col = i % 3
    curr_ax = ax[row, col]
    curr_ax.plot(x_plot, plot_arg["y"])
    curr_ax.set_xlabel(xlabel)
    curr_ax.set_ylabel(plot_arg["ylabel"])
    curr_ax.set_title(plot_arg["title"])
plt.tight_layout()
plt.show()
```

5. Merge the results using $\psi$ as the final solution.
```py
index_unconstrain = (psi_region1 < 1)
index_constrain = (psi_region1 >= 1)

q_nn = q_region1 * index_unconstrain + q_region2 * index_constrain
psi_nn = psi_region1 * index_unconstrain + 1 * index_constrain
sigq_nn = sigq_region1 * index_unconstrain + sigq_region2 * index_constrain
esig_nn = esige_region1 * index_unconstrain + esige_region2 * index_constrain
emue_nn = emue_region1 * index_unconstrain + emue_region2 * index_constrain
er_nn = er_region1 * index_unconstrain + er_region2 * index_constrain

xlabel = "$\eta$"
plot_args = [
    {"y": q_nn, "ylabel": r"$q$", "title": r"$q$ vs. $\eta$"},
    {"y": psi_nn, "ylabel": r"$\psi$", "title": r"$\psi$ vs. $\eta$"},
    {"y": sigq_nn, "ylabel": r"$\sigma^q$", "title": r"$\sigma^q$ vs. $\eta$"},
    {"y": esig_nn, "ylabel": r"$\eta\sigma^{\eta}$", "title": r"$\eta\sigma^{\eta}$ vs. $\eta$"},
    {"y": emue_nn, "ylabel": r"$\eta\mu^{\eta}$", "title": r"$\eta\mu^{\eta}$ vs. $\eta$"},
    {"y": er_nn, "ylabel": r"$E[dr_t^k-dr_t]/dt$", "title": r"$E[dr_t^k-dr_t]/dt$ vs. $\eta$"},
]

fig, ax = plt.subplots(2, 3, figsize=(18, 12))
for i, plot_arg in enumerate(plot_args):
    row = i // 3
    col = i % 3
    curr_ax = ax[row, col]
    curr_ax.plot(x_plot, plot_arg["y"])
    curr_ax.set_xlabel(xlabel)
    curr_ax.set_ylabel(plot_arg["ylabel"])
    curr_ax.set_title(plot_arg["title"])
plt.tight_layout()
plt.show()
```