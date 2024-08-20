# 2D Problem

The full solution can be found at <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/pymacrofin_eg/2d_problem5.ipynb" target="_blank">2d_problem.ipynb</a>.

## Problem Setup
This is an extension of <a href="https://www.aeaweb.org/articles?id=10.1257/aer.104.2.379" target="_blank">Brunnermeier and Sannikov 2014</a>[^1] with two agents and time-varying aggregate volatility. The PyMacroFin setup can be found <a href="https://adriendavernas.com/pymacrofin/example.html#two-dimensional-problem"  target="_blank">here</a>.

[^1]: Brunnermeier, Markus K. and Sannikov, Yuliy, *"A Macroeconomic Model with a Financial Sector"*, SIAM Review, 104(2): 379–421, 2014

## Implementation

1. Import necessary packages
```py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from deep_macrofin import PDEModel, plot_loss_df
from deep_macrofin import set_seeds, OptimizerType, SamplingMethod, LossReductionMethod
```

2. Define the problem
```py
set_seeds(0)
pde_model = PDEModel("BruSan", {"batch_size": 20, "num_epochs": 5000, "optimizer_type": OptimizerType.Adam, "sampling_method": SamplingMethod.FixedGrid})
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
pde_model.add_equation("muq = q_e/q*mue*e + q_z/q*muz*z + 1/2*q_ee/q*((siges*e)**2 + (sigek*e)**2) + 1/2*q_zz/q*(sigz*z)**2 + q_ez/q*siges*e*sigz*z")
pde_model.add_equation("muri = (ai-iotai)/q + phii + muq + sigma*sigqk")
pde_model.add_equation("murh = (ah-iotah)/q + phih + muq + sigma*sigqk")
pde_model.add_equation("r = muri - gammai*wi*((sigqs**2)+(sigma+sigqk)**2) + sigqs*sigxis + (sigqk+sigma)*sigxik")
pde_model.add_equation("muni = r + wi*(muri-r)-ci")
pde_model.add_equation("munh = r + wh*(murh-r)-ch")

pde_model.add_endog_equation("kappa_l/e*(ebar-e)+(1-e)*(muni - muk - muq\
                     - sigma*sigqk + (sigqk+sigma)**2 + sigqs**2 \
                     - wi*sigqs**2 - wi*(sigqk+sigma)**2) - mue=0", weight=2.0)
pde_model.add_endog_equation("(ci*e+ch*(1-e))*q - psi*(ai-iotai) - (1-psi)*(ah-iotah)=0")
pde_model.add_endog_equation("muri - murh + gammah*wh*((sigqs**2)+(sigqk+sigma)**2) - \
                     gammai*wi*((sigqs)**2+(sigqk+sigma)**2) + sigqs*sigxis + \
                     (sigqk+sigma)*sigxik - sigqs*sigxhs - (sigqk+sigma)*sigxhk=0", weight=2.0)
pde_model.add_endog_equation("(sigz*z*q_z + siges*e*q_e)-sigqs*q=0")
pde_model.add_endog_equation("sigek*e*q_e - sigqk*q=0", weight=2.0)

pde_model.add_hjb_equation("1/(1-1/zetai)*(ci-(rhoi+kappa_l))+r-ci+gammai/2*(wi*(sigqs)**2 +wi*(sigqk+sigma)**2)")
pde_model.add_hjb_equation("1/(1-1/zetah)*(ch-(rhoh+kappa_l))+r-ch+gammah/2*(wh*(sigqs)**2 +wh*(sigqk+sigma)**2)")

print(pde_model)
```

3. Train and evaluate
```py
pde_model.train_model("./models/2D_fixed_grid_mae", "model.pt", True)
pde_model.load_model(torch.load("./models/2D_fixed_grid_mae/model_best.pt"))
pde_model.eval_model(True)
```

4. Finetune with MAE loss on high residuals.
```py
pde_model.set_config({"num_epochs": 2000})
pde_model.set_loss_reduction("endogeq_2", LossReductionMethod.MAE)
pde_model.set_loss_reduction("endogeq_5", LossReductionMethod.MAE)
pde_model.load_model(torch.load("./models/2D_fixed_grid_mae/model_best.pt"))
pde_model.train_model("./models/2D_fixed_grid_mae", "model2.pt", True)
pde_model.load_model(torch.load("./models/2D_fixed_grid_mae/model2_best.pt"))
pde_model.eval_model(True)
```

5. Plot the solutions
```py
pde_model.plot_vars(["q", "psi", "mue", "vi", "vh",
                     "sigqk", "esigqk=e*sigqk", "sigqs", "esigqs=e*sigqs", "r"], ncols=5)
```