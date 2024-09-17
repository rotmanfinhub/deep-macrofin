# Basic Usage
This document shows how to configure and train a model using `deep_macrofin`.

## Define PDE System
Firstly, we import `PDEModel` ([API](./api/pde_model.md#pdemodel)) from the package to define the problem to be solved. The following code snippet defines a PDE model with default training settings and name `example`:

```py
from deep_macrofin import PDEModel
pde_model = PDEModel("model_name")
```

`PDEModel` takes three parameters: `name: str, config: Dict[str, Any] = DEFAULT_CONFIG, latex_var_mapping: Dict[str, str] = {}`. Only `name` is required. The trained models will be saved with filename `name` by default.

### Training Configs

The default training configs set batch size = 100, epochs = 1000, learning rate = $10^{-3}$, and AdamW optimizer. In this setting, loss will be logged to a csv file every 100 epochs during training (`loss_log_interval`), and data points are sampled randomly. 
```py
DEFAULT_CONFIG = {
    "batch_size": 100,
    "num_epochs": 1000,
    "lr": 1e-3,
    "loss_log_interval": 100,
    "optimizer_type": OptimizerType.AdamW,
    "sampling_method": SamplingMethod.UniformRandom,
    "refinement_sample_interval": int(0.2 * num_epochs),
    "loss_balancing": False,
    "bernoulli_prob": 0.9999,
    "loss_balancing_temp": 0.1,
    "loss_balancing_alpha": 0.999,
    "soft_adapt_interval": -1,
    "loss_soft_attention": False,
}
```

This dictionary can be imported:
```py
from deep_macrofin import DEFAULT_CONFIG
```

We can override any of the settings by passing a dictionary to the `config` parameter. The dictionary does not need to be complete. Any missing values are automatically replaced with default values. The following initialization will only change the number of training epoches to 10000, and keep everything else the same as default.
```py
from deep_macrofin import PDEModel
pde_model = PDEModel("model_name", {"num_epochs": 10000})
```

#### Optimizers

OptimizerType is a `Enum` object that can be imported from the package. Currently, we support the following optimizer types:
```py
from deep_macrofin import OptimizerType
# OptimizerType.Adam = "Adam"
# OptimizerType.AdamW = "AdamW"
# OptimizerType.LBFGS = "LBFGS"
```

> Note: when KAN is used as one of the learnable variables, only LBFGS is supported.

#### SamplingMethod

SamplingMethod is a `Enum` object that can be imported from the package. Currently, we support the following sampling methods:
```py
from deep_macrofin import SamplingMethod
# SamplingMethod.UniformRandom = "UniformRandom"
# SamplingMethod.FixedGrid = "FixedGrid"
# SamplingMethod.ActiveLearning = "ActiveLearning"
# SamplingMethod.RARG = "RAR-G"
# SamplingMethod.RARD = "RAR-D"
```

When sampling method is `ActiveLearning`, `RARG`, or `RARD`, additional points to help learning are sampled every `refinement_sample_interval` epochs. It is default to 200 epochs, which is 20% of total epochs.

> Note: For FixedGrid sampling, the batch size is applied to each dimension, and the final sample is of shape $(B^n, n)$, where $B$ is batch size, $n$ is number of state variables. Batch size should be set to a lower value than in uniform sampling.

> Note: RAR-G and RAR-D are implemented based on Wu et al. 2022[^1], the underlying sampling method for additional residual points is UniformRandom, and the base sampling method for training is FixedGrid. The total number of additional residual points sampled over the entire training period is $B^n$. In each addition, $\frac{B^n}{\text{refinement rounds}}$ points are added.

> Note: ActiveLearning is not yet implemented specifically, and currently uses the logic of RAR-G.

#### Dynamic Loss Weighting

**Loss Balancing** implements the Relative Loss Balancing with Random Lookback (ReLoBRaLo) algorithm in Bischof and Kraus 2021[^2]. The update follows the equations:

$$\lambda_i^{bal}(t,t') = m \frac{\exp\left( \frac{\mathcal{L}_i(t)}{\mathcal{T}\mathcal{L}_i(t')}\right)}{\sum_{j=1}^m \exp\left( \frac{\mathcal{L}_j(t)}{\mathcal{T}\mathcal{L}_j(t')}\right)}$$

$$\lambda_i^{hist}(t)=\rho \lambda_i (t-1) + (1-\rho) \lambda_i^{bal}(t,0)$$

$$\lambda_i(t) = \alpha \lambda_i^{hist} + (1-\alpha) \lambda_i^{bal}(t,t-1)$$


$m$ is the number of loss functions. $i\in \{1,...,m\}$ are indices for loss functions. $\mathcal{T}$ (`loss_balancing_temp`) is softmax temperature. $\rho$ is a Bernoulli random variable with $\mathbb{E}(\rho)\approx 1$ (`bernoulli_prob`). $\alpha$ is the exponential decay rate (`loss_balancing_alpha`).


**Soft Adapt** implements the algorithm in Heydari et al. 2019[^3]. It merges the loss Weighted and normalized approach. It also uses `loss_balancing_temp` parameter as the softmax temperature.

$$ns_i = \frac{s_i}{\sum_{j=1}^m |s_j|}$$

$$\alpha_i = \frac{\exp(\beta(ns_i - \max(ns_i)))}{\sum_{j=1}^m \exp(\beta(ns_j - \max(ns_j)))}$$

$$\alpha_i = \frac{f_i \alpha_i}{\sum_{j=1}^m f_j \alpha_j}$$

**Soft Attention** implements the algorithm in Song et al. 2024[^4]. Specifically, loss weights are linear neural networks applied to individual grid points in the training data.

### Latex Variable Map
Economic models may involve a large amount of variables and equations. Each variable can have super-/subscripts. To properly distinguish super-/subscripts from powers/derivatives and parse equations when LaTex formula are provided, we require a mapping from LaTex variables to Python strings. The keys are LaTex strings in raw format `r""`, and the values are the corresponding python string. The following dictionary maps LaTex string $\xi_t^h$ to Python string `"xih"`, and LaTex string $q_t^a$ to Python string `"qa"`.

```py
latex_var_mapping = {
    r"\xi_t^h": "xih",
    r"q_t^a": "qa"
} # define mapping

from deep_macrofin import PDEModel
pde_model = PDEModel("model_name", latex_var_mapping=latex_var_mapping) # enforce the mapping for formula parsing in the model
```

### Define Problem Domain
The problem dimension and domain are defined using state variables through `PDEModel.set_state` method ([API](./api/pde_model.md#set_state)).

The following defines a 1D problem, with state variable $x$, domain $x\in [0.01,0.99]$. 
```py
pde_model.set_state(["x"], {"x": [0.01, 0.99]})
```

The following defines a 2D problem, with state variable $(x,y)$, domain $(x,y)\in [-1,1]\times[-1,1]$ (default domain). 
```py
pde_model.set_state(["x", "y"])
```

The following defines a 2D problem, with state variable $(x,y)$, domain $(x,y)\in [0,1]\times[0,\pi]$. 
```py
pde_model.set_state(["x", "y"], {"x": [0,1], "y": [0, np.pi]}) # torch.pi is also acceptable (They are the same constant)
```

## Define Agent/EndogVar
Agents and EndogVar ([LearnableVar](./api/models.md#learnablevar)) can only be added to a PDEModel after the states are defined, so that the functions to compute derivatives can be constructed properly.

To add a single agent or endogenous variable to a model with default settings, we can use `add_agent` ([API](./api/pde_model.md#add_agent)) and `add_endog` ([API](./api/pde_model.md#add_endog)):
```py
pde_model.add_agent("xih") # this adds an agent with name: xih
pde_model.add_endog("qa") # this adds an agent with name: qa
```
If the latex_variable_mapping is properly setup in the [previous step](#latex-variable-map), `xih` is associated with $\xi_t^h$ and `qa` is associated with $q_t^a$ in formula parsing and evaluations.

Suppose there are two state variables $x,y$, then the derivatives `xih_x` ($\frac{\partial \xi_t^h}{\partial x}$), `xih_y` ($\frac{\partial \xi_t^h}{\partial y}$), `xih_xx` ($\frac{\partial^2 \xi_t^h}{\partial x^2}$), `xih_yy` ($\frac{\partial^2 \xi_t^h}{\partial y^2}$), `xih_xy` ($\frac{\partial^2 \xi_t^h}{\partial y\partial x}$), and `xih_yx` ($\frac{\partial^2 \xi_t^h}{\partial x\partial y}$) are computed for $\xi_t^h$. Similar for $q_t^a$.


### Model Configs
The default learnable variable configs sets training device to `"cuda"` if it is available. Otherwise the models will be trained on CPU. However, we recommend using CPU for training, because the models are not usually computationally expensive, while the synchronization (evaluating string formula on CPU and transferring computation to GPU) can significantly bottleneck the computation speed of GPUs. 
The models are default to 4-layer MLPs. Each layer contains 30 hidden units, with $\tanh(x)$ activation. 
By default, `positive=False`, each model can output any values $(-\infty,\infty)$. Derivatives will be taken up to second order (`derivative_order=2`).
```py
DEFAULT_LEARNABLE_VAR_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "hidden_units": [30, 30, 30, 30],
    "layer_type": LayerType.MLP,
    "activation_type": ActivationType.Tanh,
    "positive": False,
    "derivative_order": 2,
}
```

This dictionary can be imported:
```py
from deep_macrofin import DEFAULT_LEARNABLE_VAR_CONFIG
```

LayerType and ActivationType are `Enum` objects that can be imported from the package. Currently, we support the following layers and activation types:
```py
from deep_macrofin import ActivationType, LayerType
# ActivationType.ReLU = "relu" (nn.ReLU)
# ActivationType.SiLU = "silu" (nn.SiLU)
# ActivationType.Sigmoid = "sigmoid" (nn.Sigmoid)
# ActivationType.Tanh = "tanh" (nn.Tanh)
# ActivationType.Wavelet="wavelet" (w1*sin(x)+w2*cos(x), where w1 and w2 are learnable)

# LayerType.MLP = "MLP"
# LayerType.KAN = "KAN"
```

Similar to PDEModel defintion, we can override any of the settings by passing a dictionary to the `config` parameter.

In the following setup, `xih` is defined to be a 2-layer MLP. Each layer contains 40 hidden units and is activated by SiLU.
Its output is restricted to be positive only by SoftPlus.
Suppose the state variable is $x$, then four derivatives are computed: `xih_x`, `xih_xx`, `xih_xxx`, `xih_xxxx`. 
`qa` is defined as a 2-layer KAN. The input and output sizes are restricted to 1, and there are two layers of width 5.
```py
pde_model.add_agent("xih", 
    config={
        "hidden_units": [40, 40], 
        "activation_type": ActivationType.SiLU, 
        "positive": True, 
        "derivative_order": 4}
) 
pde_model.add_endog("qa", 
    config={
        "hidden_units": [1, 5, 5, 1], 
        "layer_type": LayerType.KAN}
)
```

### Defining Multiple Agents/EndogVars
Multiple agents and endogenous variables with different configurations can be added to PDEModel at the same time using `add_agents` ([API](./api/pde_model.md#add_agents)) and `add_endogs` ([API](./api/pde_model.md#add_endogs)). The following code adds two agents `"xii", "xih"`, and five endogenous variables `"mue", "qa", "wia", "wha", "sigea"` to a PDEModel.
```py
pde_model.add_agents(["xii", "xih"], 
    {
        "xii": {"positive": True}, 
        "xih": {"positive": True}
    }
)
pde_model.add_endogs(["mue", "qa", "wia", "wha", "sigea"], 
    {"qa": {"positive": True}})
```

The parameter `configs` is now a multi-level dictionary, `{var1_name: {var1_configs}, var2_name: {var2_configs}, ...}`. Any configurations not provided by the user are set as default. 


### Conditions for Agent/EndogVar
`add_agent_condition` ([API](./api/pde_model.md#add_agent_condition)) and `add_endog_condition` ([API](./api/pde_model.md#add_endog_condition)) can be used for adding conditions (initial and boundary conditions, and any point that requires specific values)

The following code adds a condition on a 1D endogenous variable $y$: $y(0)=1$. The key `SV` can be replaced by any variable that is not used in the system. For example, if `lhs` is defined as `"y(X)"`, then `lhs_state` should be replaced with `{"X": torch.zeros((1, 1))}`. Because rhs is a constant 1, there is no need for any `rhs_state` definition. An empty dictionary suffices. `weight=2` means the computed loss will have weight 2 in the PDEModel.
```py
pde_model.add_endog_condition("y", 
    "y(SV)", {"SV": torch.zeros((1, 1))}, 
    Comparator.EQ, 
    "1", {}, 
    weight=2)
```

`Comparator` is a `Enum` object that can be imported from the package.
```py
from deep_macrofin import Comparator
# Comparator.LEQ = "<="
# Comparator.GEQ = ">="
# Comparator.LT = "<"
# Comparator.GT = ">"
# Comparator.EQ = "="
```

The following code adds a condition on a 2D endogenous variable $z$ s.t. $z(0, y)\leq y$ $\forall y\in[0,1]$. Batch size is set to 100 here, but can be finetuned to get better accuracy. Each element in the batch requires two state variables. The first variable is $x=0$, and the second variable $y$ is set to be a linear space from 0 to 1. RHS is replaced with `zero_x[:, 1:2]` as $y$.
```py
zero_x = torch.zeros((100, 2)) 
zero_x[:, 1] = torch.tensor(np.linspace(0, 1, num=100))
pde_model.add_endog_condition("y", 
    "y(SV)", {"SV": zero_x}, 
    Comparator.LEQ, 
    "rhs", {"rhs": zero_x[:, 1:2]}
)
```

## Add Equations, etc
**Note**: 
- When an equation is provided using LaTex, it must be enclosed by `$` and defined as a raw string `r""`
- Derivatives should be provided as `"y_x"` or `"y_xx"` in raw python strings, or `"\frac{\partial y}{\partial x}"` or `"\frac{\partial^2 y}{\partial x^2}"` in LaTex strings.

`add_param` ([API](./api/pde_model.md#add_param)) and `add_params` ([API](./api/pde_model.md#add_params)) are used to define constant parameters.  
`add_equation` ([API](./api/pde_model.md#add_equation)) defines new variables through an equation.  

The following adds two parameters `zetai=1.0005` ($\zeta^i$) and `gammai=2` ($\gamma^i$) to the system, and defines a new variable $y=\zeta^i + \gamma^i$ in the system. 
```py
from deep_macrofin import PDEModel
latex_var_mapping = {
    r"\zeta^i": "zetai",
    r"\gamma^i": "gammai"
} # define mapping
pde_model = PDEModel("model_name", latex_var_mapping=latex_var_mapping)
pde_model.add_params({
    "zetai": 1.0005,
    "gammai": 2,
})
# The following two equations are equivalent.
# DO NOT copy and run the following code, because double definition could cause errors in the model
pde_model.add_equation(r"$y=\zeta^i + \gamma^i$") # latex version
pde_model.add_equation("y=zetai+gammai") # pure python version
```

`add_endog_equation` ([API](./api/pde_model.md#add_endog_equation)) defines endogenous equations (algebraic equations) that must be satisfied by the system.  
`add_hjb_equation` ([API](./api/pde_model.md#add_hjb_equation)) defines HJB equation to maximize/minimize for each agent.
`add_system` ([API](./api/pde_model.md#add_system)) defines constraint-activated systems to handle different components of a system.

Endogenous equations, constraints, HJB equations and systems are used to compute losses.


The following code defines an endogenous equation

$$y'' + 6y' + 5*y = 0$$

```py
pde_model.add_endog_equation(r"$\frac{\partial^2 y}{\partial x^2} + 6 * \frac{\partial y}{\partial x} + 5 * y =0$") # latex version
pde_model.add_endog_equation("y_xx+6*y_x+5*y=0") # pure python version
```

The following code defines an HJB equation to maximize

$$\frac{\rho^i}{1-\frac{1}{\zeta^i}} * \left( \left(\frac{c_t^i}{\xi_t^i} \right)^{1-1/\zeta^i}-1 \right) + \mu_t^{\xi i} +  \mu_t^{ni} - \frac{\gamma^i}{2} * (\sigma_t^{nia})^2  - \frac{\gamma^i}{2} * (\sigma_t^{\xi ia})^2$$

```py
# Note: proper parsing of variables are required. See 1D Economic Problem for full example.
pde_model.add_hjb_equation(r"$\frac{\rho^i}{1-\frac{1}{\zeta^i}} * \left( \left(\frac{c_t^i}{\xi_t^i} \right)^{1-1/\zeta^i}-1 \right) + \mu_t^{\xi i} +  \mu_t^{ni} - \frac{\gamma^i}{2} * (\sigma_t^{nia})^2  - \frac{\gamma^i}{2} * (\sigma_t^{\xi ia})^2$")
```

The systems ([API](./api/evaluations.md#system)) are defined using activation constraints.   
The following code defines a system that is activated when $\frac{\partial y}{\partial x} \geq 0$ and $x \in [0,0.5]$, and computes $z=x^2y$, requiring $y+z = 1$ in this case.

```py
from deep_macrofin import System, Constraint
# because the systems are separated from the main PDEModel when being defined, labels are required when initializing the constraints and the system.
s = System([
    Constraint("y_x", Comparator.GEQ, "0"), # y_x>=0
    Constraint("x", Comparator.GEQ, "0"), # x>= 0
    Constraint("x", Comparator.LEQ, "0.5"), # x<=0.5
    ], "sys1")
s.add_equation("z=x**2 * y") 
s.add_endog_equation("y+z = 1")
pde_model.add_system(s) # add the system to PDEModel
```


### About Labels
For all equations, conditions, etc added to the model, there is an optional parameter `label` used to identify the equations. This can be automatically set based on the sequence the equation is added to the system, and usually doesn't need to be set by the user. 
```py
pde_model.add_equation("y=x+1") # eq_1 is y=x+1
pde_model.add_equation("z=y*2") # eq_2 is z=y*2
```


## Training and Evaluation

After the model is defined, `train_model` ([API](./api/pde_model.md#train_model)) can be used to train the models, and `eval_model` ([API](./api/pde_model.md#eval_model)) can be used to eval models. 
`load_model` ([API](./api/pde_model.md#load_model)) can be used to load a trained model. It will overwrite existing agent/endogenous variables in the PDEModel.

`train_model` will print out the full model configurations and save the best and final models. Losses are logged every `loss_log_interval` epochs during training in `modelname_loss.csv` file for plotting. Minimum (converging) losses are logged in a separate `modelname_min_loss.csv` file. Examples can be found in [Basic Examples](./examples/approx/discont.md).

### Print the Model
For easier debugging on the model setup and equation typing, `print(pde_model)` prints out a detailed configuration of the model. The following is a sample print out of [log utility problem](./examples/macrofinance_models/log_utility.md). The same summary is logged in the log file for each model. 

```
=====================Summary of Model BruSan14_log_utility======================
Config: {
 "batch_size": 100,
 "num_epochs": 100,
 "lr": 0.001,
 "loss_log_interval": 10,
 "optimizer_type": "Adam"
}
Latex Variable Mapping:
{
 "\\sigma_t^q": "sigq",
 "\\sigma_t^\\theta": "sigtheta",
 "\\sigma_t^\\eta": "sige",
 "\\mu_t^\\eta": "mue",
 "\\mu_t^q": "muq",
 "\\mu_t^\\theta": "mutheta",
 "\\rho": "rho",
 "\\underline{a}": "ah",
 "\\underline{\\delta}": "deltah",
 "\\delta": "deltae",
 "\\sigma": "sig",
 "\\kappa": "kappa",
 "\\eta": "e",
 "\\theta": "theta",
 "\\psi": "psi",
 "\\iota": "iota",
 "\\Phi": "phi"
}
User Defined Parameters:
{
 "sig": 0.1,
 "deltae": 0.05,
 "deltah": 0.05,
 "rho": 0.06,
 "r": 0.05,
 "a": 0.11,
 "ah": 0.07,
 "kappa": 2
}

================================State Variables=================================
e: [0.0, 1.0]

=====================================Agents=====================================

================================Agent Conditions================================

==============================Endogenous Variables==============================
Endogenous Variable Name: q
EndogVar(
  (model): Sequential(
    (linear_0): Linear(in_features=1, out_features=30, bias=True)
    (activation_0): Tanh()
    (linear_1): Linear(in_features=30, out_features=30, bias=True)
    (activation_1): Tanh()
    (linear_2): Linear(in_features=30, out_features=30, bias=True)
    (activation_2): Tanh()
    (linear_3): Linear(in_features=30, out_features=30, bias=True)
    (activation_3): Tanh()
    (final_layer): Linear(in_features=30, out_features=1, bias=True)
    (positive_act): Softplus(beta=1.0, threshold=20.0)
  )
)
Num parameters: 2881
--------------------------------------------------------------------------------
Endogenous Variable Name: psi
EndogVar(
  (model): Sequential(
    (linear_0): Linear(in_features=1, out_features=30, bias=True)
    (activation_0): Tanh()
    (linear_1): Linear(in_features=30, out_features=30, bias=True)
    (activation_1): Tanh()
    (linear_2): Linear(in_features=30, out_features=30, bias=True)
    (activation_2): Tanh()
    (linear_3): Linear(in_features=30, out_features=30, bias=True)
    (activation_3): Tanh()
    (final_layer): Linear(in_features=30, out_features=1, bias=True)
    (positive_act): Softplus(beta=1.0, threshold=20.0)
  )
)
Num parameters: 2881
--------------------------------------------------------------------------------

========================Endogenous Variables Conditions=========================
endogvar_q_cond_q_min: q(SV)=(2*ah*kappa + (kappa*r)**2 + 1)**0.5 - kappa*r with LHS evaluated at SV=[[0.0]] and RHS evaluated at sig=0.1deltae=0.05deltah=0.05rho=0.06r=0.05a=0.11ah=0.07kappa=2
Loss weight: 1.0
--------------------------------------------------------------------------------

===================================Equations====================================
eq_1: 
Raw input: $\iota = \frac{q^2-1}{ 2 * \kappa}$
Parsed: iota=(q**(2)-1)/( 2 * kappa)
eq_2: 
Raw input: $\sigma_t^q = \frac{\sigma}{1 - \frac{1}{q} * \frac{\partial q}{\partial \eta} * (\psi - \eta)} - \sigma$
Parsed: sigq=(sig)/(1 - (1)/(q) * q_e * (psi - e)) - sig
eq_3: 
Raw input: $\sigma_t^\eta = \frac{\psi - \eta}{\eta} * (\sigma + \sigma_t^q)$
Parsed: sige=(psi - e)/(e) * (sig + sigq)
eq_4: 
Raw input: $\mu_t^\eta = (\sigma_t^\eta)^2 + \frac{a - \iota}{q} + (1-\psi) * (\underline{\delta} - \delta) - \rho$
Parsed: mue=(sige)**(2) + (a - iota)/(q) + (1-psi) * (deltah - deltae) - rho

==============================Endogenous Equations==============================
endogeq_1: 
Raw input: $(\sigma + \sigma_t^q) ^2 * (\psi / \eta - (1-\psi) / (1-\eta)) = \frac{a - \underline{a}}{q} + \underline{\delta} - \delta$
Parsed: (sig + sigq) **(2) * (psi / e - (1-psi) / (1-e))=(a - ah)/(q) + deltah - deltae
Loss weight: 1.0
--------------------------------------------------------------------------------

==================================Constraints===================================
constraint_1: psi<=1
Loss weight: 1.0
--------------------------------------------------------------------------------

=================================HJB Equations==================================

====================================Systems=====================================
non-opt: 
Activation Constraints:
non-opt: psi<1
===============Equations================

==========Endogenous Equations==========
system_non-opt_endogeq_1: 
Raw input: $(r*(1-\eta) + \rho * \eta) * q = \psi * a + (1-\psi) * \underline{a} - \iota$
Parsed: (r*(1-e) + rho * e) * q=psi * a + (1-psi) * ah - iota
Loss weight: 1.0
----------------------------------------

System loss weight: 1.0
--------------------------------------------------------------------------------
opt: 
Activation Constraints:
opt: psi>=1
===============Equations================

==========Endogenous Equations==========
system_opt_endogeq_1: 
Raw input: $(r*(1-\eta) + \rho * \eta) * q = a - \iota$
Parsed: (r*(1-e) + rho * e) * q=a - iota
Loss weight: 1.0
----------------------------------------

System loss weight: 1.0
--------------------------------------------------------------------------------
```

When errors are found in equation validation before training begins, the system will save an error log.

## Plot
Once the models are trained, there are several values that can be plotted. However, plottings are only supported for 1D and 2D models.

### Plot Learnable Variables
Each learnable variable has a plot function ([API](./api/models.md#plot)) to plot the function itself and associated derivatives.

The following code plots only the original function $y$ in an individual matplotlib figure on the domain $[0, 1]$.
```py
pde_model.agents["y"].plot("y", domain=[0, 1], ax=None)
```

The following code plots $y, y', y''$ in one row three column:
```py
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
pde_model.endog_vars["y"].plot("y", {"x": [0, 1]}, ax=ax[0])
pde_model.endog_vars["y"].plot("y_x", {"x": [0, 1]}, ax=ax[1])
pde_model.endog_vars["y"].plot("y_xx", {"x": [0, 1]}, ax=ax[2])
plt.subplots_adjust()
plt.show()
```

The following code plots $u(x,y)$ in 2D:
```py
fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={"projection": "3d"})
pde_model.endog_vars["u"].plot("u", {"x": [0, 1.], "y": [0, 1.]}, ax=ax)
plt.subplots_adjust()
plt.show()
```

### Plot Variables Defined by Equations
PDEModel has a `plot_vars` function ([API](./api/pde_model.md#plot_vars)), which can plot any variables that are already defined in a `PDEModel` or additional variables of interests using new equations.

The following code will plot $q_t^a, \sigma_t^{qa}, w_t^{ia}, w_t^{ha}$ and risk premium (rp) in a four-column format. The risk premium is a new variable defined by $r_t^{ka} - r_t$.
```py
pde_model.plot_vars([
    r"$q_t^a$", 
    r"$\sigma_t^{qa}$", 
    r"$w_t^{ia}$", 
    r"$w_t^{ha}$",
    r"$rp = r_t^{ka} - r_t $"])
```

The following code will plot $q_t^a, \sigma_t^{qa}, w_t^{ia}, w_t^{ha}$ in a 2x2 grid.
```py
pde_model.plot_vars([
    r"$q_t^a$", 
    r"$\sigma_t^{qa}$", 
    r"$w_t^{ia}$", 
    r"$w_t^{ha}$"], ncols=2)
```

### Plot Loss
`plot_loss_df` ([API](./api/utils.md#plot_loss_df)) plots losses in the logged csv file.

A general usage to plot all losses in the file `./models/1d_prob/1d_prob_loss.csv`:
```py
from deep_macrofin import plot_loss_df
plot_loss_df(fn="./models/1d_prob/1d_prob_loss.csv", loss_plot_fn="./models/1d_prob/1d_prob_loss.png")
```

[^1]: Chenxi Wu, and Min Zhu, and Qinyang Tan, and Yadhu Kartha, and Lu Lu, *"A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks"*, 2022-07-21, <a href="https://arxiv.org/abs/2207.10289" target="_blank">arXiv:2207.10289</a>  

[^2]: Rafael Bischof, and Michael Kraus, *"Multi-Objective Loss Balancing for Physics-Informed Deep Learning"*, 2021-10-19, <a href="https://arxiv.org/abs/2110.09813">arXiv:2110.09813</a>

[^3]: A. Ali Heydari, and Craig A. Thompson, and Asif Mehmood, *"SoftAdapt: Techniques for Adaptive Loss Weighting of Neural Networks with Multi-Part Loss Functions"*, 2019-12-27, <a href="https://arxiv.org/abs/1912.12355">arXiv:1912.12355</a>

[^4]: Yanjie Song, and He Wang, and He Yang, and Maria Luisa Taccari, and Xiaohui Chen, *"Loss-attentional physics-informed neural networks"*, Journal of Computational Physics, Volume 501, 2024, <a href="https://www.sciencedirect.com/science/article/pii/S0021999124000305">link</a>