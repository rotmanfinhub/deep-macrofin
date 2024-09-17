# PyMacroFin 2D Problem

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/pymacrofin_eg/2d_problem.ipynb" target="_blank">2d_problem.ipynb</a> and <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/time_step/2d_problem.ipynb" target="_blank">2d_problem (time stepping).ipynb</a>.

## Problem Setup
This is an extension of <a href="https://www.aeaweb.org/articles?id=10.1257/aer.104.2.379" target="_blank">Brunnermeier and Sannikov 2014</a>[^1] with two agents and time-varying aggregate volatility. The PyMacroFin setup can be found <a href="https://adriendavernas.com/pymacrofin/example.html#two-dimensional-problem"  target="_blank">here</a>.

[^1]: Brunnermeier, Markus K. and Sannikov, Yuliy, *"A Macroeconomic Model with a Financial Sector"*, SIAM Review, 104(2): 379–421, 2014

## Implementation (Loss Balancing)

With default training scheme, the loss does not converge to a small enough number, so we apply Relative Loss Balancing with Random Lookback (ReLoBRaLo) algorithm. The implementation can be found in <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/pymacrofin_eg/2d_problem.ipynb" target="_blank">2d_problem.ipynb</a>.

1. Import necessary packages
```py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from deep_macrofin import PDEModel, plot_loss_df
from deep_macrofin import ActivationType, Comparator, Constraint, System, set_seeds, OptimizerType, SamplingMethod
```

2. Define the problem, with loss balancing, $\mathbb{E}(\rho)=0.999$, $\mathcal{T}=0.1$, $\alpha=0.999$.
```py
set_seeds(0)
pde_model = PDEModel("BruSan", {"batch_size": 50, "num_epochs": 20000, "lr": 1e-3, "optimizer_type": OptimizerType.Adam, "sampling_method": SamplingMethod.FixedGrid,
                                "loss_balancing": True, "bernoulli_prob": 0.9999, "loss_balancing_temp": 0.1, "loss_balancing_alpha": 0.999})
pde_model.set_state(["e", "z"], {"e": [0.05, 0.95], "z": [0.05, 0.95]})
pde_model.add_endogs(["q", "psi", "mue", "sigqk", "sigqs"], configs={
    "q": {"positive": True},
    "psi": {"positive": True},
})
pde_model.add_agents(["vi", "vh"], configs={
    "vi": {"positive": True},
    "vh": {"positive": True},
})
pde_model.add_params({
    "gammai": 2,
    "gammah": 3,
    "ai": .1,
    "ah": .1,
    "rhoi": .04,
    "rhoh": .04,
    "sigz": .01,
    "sigbar": .5,
    "deltai": .04,
    "deltah": .04,
    "kappa_p": 2,
    "kappa_z": 5,
    "zetai": 1.15,
    "zetah": 1.15,
    "kappa_l": .9,
    "ebar": .5,
})

pde_model.add_equation("sigma = z")
pde_model.add_equation("wi = psi/e")
pde_model.add_equation("wh = (1-psi)/(1-e)")
pde_model.add_equation("ci = vi**((1-zetai)/(1-gammai))")
pde_model.add_equation("ch = vh**((1-zetah)/(1-gammah))")
pde_model.add_equation("iotai = (q-1)/kappa_p")
pde_model.add_equation("iotah = (q-1)/kappa_p")
pde_model.add_equation("phii = log(1+kappa_p*iotai)/kappa_p-deltai")
pde_model.add_equation("phih = log(1+kappa_p*iotah)/kappa_p-deltah")
pde_model.add_equation("muz = kappa_z*(sigbar-sigma)")
pde_model.add_equation("muk = psi*phii+(1-psi)*phih")
pde_model.add_equation("signis = wi*sigqs")
pde_model.add_equation("signhs = wh*sigqs")
pde_model.add_equation("signik = wi*(sigqk+sigma)")
pde_model.add_equation("signhk = wh*(sigqk+sigma)")
pde_model.add_equation("siges = e*(1-e)*(signis -sigqs)")
pde_model.add_equation("sigek = e*(1-e)*(signik - (sigqk+sigma))")
pde_model.add_equation("sigxik = vi_e/vi*sigek*e")
pde_model.add_equation("sigxhk = vh_e/vh*sigek*e")
pde_model.add_equation("sigxis = vi_e/vi*siges*e + vi_z/vi*sigz*z")
pde_model.add_equation("sigxhs = vh_e/vh*siges*e + vh_z/vh*sigz*z")
pde_model.add_equation("muee=mue*e")
pde_model.add_equation("muzz=muz*z")
pde_model.add_equation("sigee=(siges*e)**2+(sigek*e)**2")
pde_model.add_equation("sigzz=(sigz*z)**2")
pde_model.add_equation("sigcross=siges*e*sigz*z")
pde_model.add_equation("muq = q_e/q*muee + q_z/q*muzz + 1/2*q_ee/q*sigee + 1/2*q_zz/q*sigzz + q_ez/q*sigcross")
pde_model.add_equation("muri = (ai-iotai)/q + phii + muq + sigma*sigqk")
pde_model.add_equation("murh = (ah-iotah)/q + phih + muq + sigma*sigqk")
pde_model.add_equation("r = muri - gammai*wi*((sigqs**2)+(sigma+sigqk)**2) + sigqs*sigxis + (sigqk+sigma)*sigxik")
pde_model.add_equation("muni = r + wi*(muri-r)-ci")
pde_model.add_equation("munh = r + wh*(murh-r)-ch")
pde_model.add_equation("muxi = vi_e/vi*muee + vi_z/vi*muzz + 1/2*vi_ee/vi*sigee + 1/2*vi_zz/vi*sigzz + vi_ez/vi*sigcross")
pde_model.add_equation("muxh = vh_e/vh*muee + vh_z/vh*muzz + 1/2*vh_ee/vh*sigee + 1/2*vh_zz/vh*sigzz + vh_ez/vh*sigcross")

pde_model.add_endog_equation("kappa_l/e*(ebar-e)+(1-e)*(muni - muk - muq - sigma*sigqk + (sigqk+sigma)**2 + sigqs**2 - wi*sigqs**2 - wi*(sigqk+sigma)**2) - mue=0")
pde_model.add_endog_equation("(ci*e+ch*(1-e))*q - psi*(ai-iotai) - (1-psi)*(ah-iotah)=0")
pde_model.add_endog_equation("muri - murh + gammah*wh*((sigqs**2)+(sigqk+sigma)**2) - gammai*wi*((sigqs)**2+(sigqk+sigma)**2) + sigqs*sigxis + (sigqk+sigma)*sigxik - sigqs*sigxhs - (sigqk+sigma)*sigxhk=0")
pde_model.add_endog_equation("(sigz*z*q_z + siges*e*q_e)-sigqs*q=0")
pde_model.add_endog_equation("sigek*e*q_e - sigqk*q=0")

pde_model.add_hjb_equation("1/(1-1/zetai)*(ci-(rhoi+kappa_l)) + r - ci + gammai/2*(wi * sigqs)**2 + gammai/2*(wi*sigma+wi*sigqk)**2 + muxi / (1-gammai)")
pde_model.add_hjb_equation("1/(1-1/zetah)*(ch-(rhoh+kappa_l)) + r - ch + gammah/2*(wh * sigqs)**2 + gammah/2*(wh*sigma+wh*sigqk)**2 + muxh / (1-gammah)")
```

3. Train and evaluate
```py
pde_model.train_model("./models/2D_fixed_grid_loss_balance", "model.pt", True)
pde_model.load_model(torch.load("./models/2D_fixed_grid_loss_balance/model_best.pt"))
pde_model.eval_model(True)
```

4. Plot the solutions
```py
pde_model.plot_vars(["q", r"$\psi=psi$", r"$\mu^{\eta}=mue$", r"$\sigma^{q,k}=sigqk$",
                     r"$\sigma^{q,\sigma}=sigqs$", r"$\xi^i=vi$", r"$\xi^h=vh$"], ncols=4)
```

## Implementation (Time Stepping)

Here we show the time stepping scheme that is similar to what PyMacroFin implements, but in a neural network approach. The implementation can be found in <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/time_step/2d_problem.ipynb" target="_blank">2d_problem (time stepping).ipynb</a>.

1. Import necessary packages
```py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from deep_macrofin import PDEModelTimeStep, plot_loss_df
from deep_macrofin import ActivationType, Comparator, Constraint, System, set_seeds, OptimizerType, SamplingMethod
```

2. Define the problem. The HJB equations now involve a false transient term in time.
```py
set_seeds(0)
pde_model = PDEModelTimeStep("BruSan", {"batch_size": 20,
    "num_outer_iterations": 50,
    "num_inner_iterations": 5000,
    "lr": 1e-3,
    "loss_log_interval": 100,
    "optimizer_type": OptimizerType.Adam,
    "min_t": 0.0,
    "max_t": 1.0,
    "outer_loop_convergence_thres": 1e-4})
pde_model.set_state(["e", "z"], {"e": [0.05, 0.95], "z": [0.05, 0.95]})
pde_model.add_endogs(["q", "psi", "mue", "sigqk", "sigqs"], configs={
    "q": {"positive": True},
    "psi": {"positive": True},
})
pde_model.add_agents(["vi", "vh"], configs={
    "vi": {"positive": True},
    "vh": {"positive": True},
})
pde_model.add_params({
    "gammai": 2,
    "gammah": 3,
    "ai": .1,
    "ah": .1,
    "rhoi": .04,
    "rhoh": .04,
    "sigz": .01,
    "sigbar": .5,
    "deltai": .04,
    "deltah": .04,
    "kappa_p": 2,
    "kappa_z": 5,
    "zetai": 1.15,
    "zetah": 1.15,
    "kappa_l": .9,
    "ebar": .5,
})

pde_model.add_equation("sigma = z")
pde_model.add_equation("wi = psi/e")
pde_model.add_equation("wh = (1-psi)/(1-e)")
pde_model.add_equation("ci = vi**((1-zetai)/(1-gammai))")
pde_model.add_equation("ch = vh**((1-zetah)/(1-gammah))")
pde_model.add_equation("iotai = (q-1)/kappa_p")
pde_model.add_equation("iotah = (q-1)/kappa_p")
pde_model.add_equation("phii = log(1+kappa_p*iotai)/kappa_p-deltai")
pde_model.add_equation("phih = log(1+kappa_p*iotah)/kappa_p-deltah")
pde_model.add_equation("muz = kappa_z*(sigbar-sigma)")
pde_model.add_equation("muk = psi*phii+(1-psi)*phih")
pde_model.add_equation("signis = wi*sigqs")
pde_model.add_equation("signhs = wh*sigqs")
pde_model.add_equation("signik = wi*(sigqk+sigma)")
pde_model.add_equation("signhk = wh*(sigqk+sigma)")
pde_model.add_equation("siges = e*(1-e)*(signis -sigqs)")
pde_model.add_equation("sigek = e*(1-e)*(signik - (sigqk+sigma))")
pde_model.add_equation("sigxik = vi_e/vi*sigek*e")
pde_model.add_equation("sigxhk = vh_e/vh*sigek*e")
pde_model.add_equation("sigxis = vi_e/vi*siges*e + vi_z/vi*sigz*z")
pde_model.add_equation("sigxhs = vh_e/vh*siges*e + vh_z/vh*sigz*z")
pde_model.add_equation("muee=mue*e")
pde_model.add_equation("muzz=muz*z")
pde_model.add_equation("sigee=(siges*e)**2+(sigek*e)**2")
pde_model.add_equation("sigzz=(sigz*z)**2")
pde_model.add_equation("sigcross=siges*e*sigz*z")
pde_model.add_equation("muq = q_e/q*muee + q_z/q*muzz + 1/2*q_ee/q*sigee + 1/2*q_zz/q*sigzz + q_ez/q*sigcross")
pde_model.add_equation("muri = (ai-iotai)/q + phii + muq + sigma*sigqk")
pde_model.add_equation("murh = (ah-iotah)/q + phih + muq + sigma*sigqk")
pde_model.add_equation("r = muri - gammai*wi*((sigqs**2)+(sigma+sigqk)**2) + sigqs*sigxis + (sigqk+sigma)*sigxik")
pde_model.add_equation("muni = r + wi*(muri-r)-ci")
pde_model.add_equation("munh = r + wh*(murh-r)-ch")
pde_model.add_equation("rvi=-1*(1-gammai)*(1/(1-1/zetai)*(ci-(rhoi+kappa_l))+r-ci+gammai/2*(wi*(sigqs)**2 +wi*(sigqk+sigma)**2))")
pde_model.add_equation("rvh=-1*(1-gammah)*(1/(1-1/zetah)*(ch-(rhoh+kappa_l))+r-ch+gammah/2*(wh*(sigqs)**2 +wh*(sigqk+sigma)**2))")

pde_model.add_endog_equation("kappa_l/e*(ebar-e)+(1-e)*(muni - muk - muq - sigma*sigqk + (sigqk+sigma)**2 + sigqs**2 - wi*sigqs**2 - wi*(sigqk+sigma)**2) - mue=0")
pde_model.add_endog_equation("(ci*e+ch*(1-e))*q - psi*(ai-iotai) - (1-psi)*(ah-iotah)=0")
pde_model.add_endog_equation("muri - murh + gammah*wh*((sigqs**2)+(sigqk+sigma)**2) - gammai*wi*((sigqs)**2+(sigqk+sigma)**2) + sigqs*sigxis + (sigqk+sigma)*sigxik - sigqs*sigxhs - (sigqk+sigma)*sigxhk=0")
pde_model.add_endog_equation("(sigz*z*q_z + siges*e*q_e)-sigqs*q=0")
pde_model.add_endog_equation("sigek*e*q_e - sigqk*q=0")

pde_model.add_hjb_equation("muee*vi_e+muzz*vi_z+1/2*(sigee)**2*vi_ee+1/2*(sigzz)**2*vi_zz+sigcross*vi_ez+vi_t-rvi*vi")
pde_model.add_hjb_equation("muee*vh_e+muzz*vh_z+1/2*(sigee)**2*vh_ee+1/2*(sigzz)**2*vh_zz+sigcross*vh_ez+vh_t-rvh*vh")
```

3. Train and evaluate
```py
pde_model.train_model("./models/2D_timestep", "model.pt", True)
pde_model.load_model(torch.load("./models/2D_timestep/model_best.pt"))
pde_model.eval_model(True)
```

4. Plot the solutions
```py
pde_model.plot_vars(["q", r"$\psi=psi$", r"$\mu^{\eta}=mue$", r"$\sigma^{q,k}=sigqk$",
                     r"$\sigma^{q,\sigma}=sigqs$", r"$\xi^i=vi$", r"$\xi^h=vh$"], ncols=4)
```