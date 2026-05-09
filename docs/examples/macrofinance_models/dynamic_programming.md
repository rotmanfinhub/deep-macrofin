# Dynamic Programming

Our library can also adapt to the solution technique, introduced and described in [Duarte, Fonesca, Goodman, and Parker (2021)](https://www.nber.org/system/files/working_papers/w29559/w29559.pdf). We provide two examples replicated from the [nndp](https://github.com/marcdelabarrera/nndp/tree/main) package:

- <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/macro_problems/nndp/nndp_cake_eating.ipynb" target="_blank">Cake Eating</a>
- <a href="https://github.com/rotmanfinhub/deep-macrofin/blob/main/examples/macro_problems/nndp/nndp_income_fluctuation.ipynb" target="_blank">Income Fluctuation</a>

## Problem Formulation

This solution technique applies to the problems of the following form:

$$
\begin{align*}
V(s_0) &= \max_{a_t\in\Gamma(s_t)} E_0\left[\sum_{t=0}^T u(s_t,a_t)\right]\\
    s_{t+1} &= m(s_{t}, a_{t}, \epsilon_t)\\
    s_0 &\sim \text{Some initial distribution}
\end{align*}
$$

The state vector is denoted by $s_t=(k_t, x_t)$, where $k_t$ are exogenous states and $x_t$ are endogenous states. We adopt the convention that the first exogenous state in $k_t$ is $t$. The goal is to find a policy function $\pi(s_t)$ that satisfies:

$$
\begin{align*}
\hat{V}(s_0) &= \max_{\pi \in \Pi} E_0 \left[\sum_{t=0}^T u(s_t, \pi(s_t))\right]\\
    s_{t+1} &= m(s_{t}, \pi(s_t), \epsilon_t)\\
    V(s_0) &= \hat{V}(s_0, \pi)
\end{align*}
$$

## Implementation

1. Sample initial state $s_0$:  
In this step, we can redefine the sampling function to adapt to any initial distributions.  
```python
from deep_macrofin import PDEModel

class PDEModelCustomSample(PDEModel):
    def __init__(self, name, config, latex_var_mapping={}):
        super().__init__(name, config, latex_var_mapping)
        self.sample = self.sample_custom

    def sample_custom(self, epoch):
        N = self.batch_size
        '''
        Define your custom state vectors. 
        The sequence should follow the state variable definition in the problem
        '''
        # t = torch.zeros(N, device=self.device)
        return None # replace with your desired variables
```
2. Define utility function and value function:  
The value function should have an internal time iteration to compute $\sum_{t=0}^T$. Here, we assume a discounted CRRA utility $u(t, c)=\beta^t \frac{c^{1-\gamma}}{1-\gamma}$. The value function arguments should be properly defined as needed.  
**Note**: in each timestep, the state variables should not be detached. Otherwise the gradient computation will be incomplete.  
```python
def compute_u(t, c, T, beta, gamma):
    # compute a discounted CRRA utility 
    crra = (beta**t)*(c**(1-gamma))/(1-gamma)
    return torch.where(t<=T, crra, 0)

def compute_value_function(SV: torch.Tensor, compute_pi, compute_u, *args): # replace *args with needed arguments
    B = SV.shape[0]
    # For each initial value, perform N_simul monte carlo samples
    SV = torch.repeat_interleave(SV, repeats=N_simul, dim=0) 
    V = torch.zeros((SV.shape[0], 1), device=SV.device) # (B*N_simul, 1)
    curr_state = SV
    # iterate through time
    for time_iter in range(T + 1):
        # 1. get the current state variables
        # 2. compute policy and value function
        action = compute_pi(curr_state)
        V += compute_u(t, c, T, beta, gamma)
        # 3. update to next state
        curr_state = next_state
    # average across all monte carlo samples to find the expectation
    V = V.reshape(B, N_simul, 1).mean(dim=1) 
    # the final loss is the average across all batches, we want to maximize the value function, so negative loss
    return -torch.mean(V) 
```
3. Define the problem and models using Deep-MacroFin:  
The `CONFIGS` are consistent with all the other PDE models. The convention for `STATE_VARIABLE` is to start with the time dimension `t` as in [nndp](https://github.com/marcdelabarrera/nndp/tree/main). `STATE_VARIABLE_CONSTRAINTS` does not matter much now, because we overwrite the sampling function.  
```python
from deep_macrofin import Comparator, OptimizerType, LossReductionMethod, set_seeds
set_seeds(0)
model = PDEModelCustomSample("model_name", config=CONFIGS)
model.set_state(STATE_VARIABLES, STATE_VARIABLE_CONSTRAINTS)
# define the policy function. It could be either agent or endogenous variable.
# we set derivative_order to zero so that we don't waste time to compute any derivatives like dpi/dt, 
# which are not used in the optimization
model.add_agent("pi", config={"hidden_units": model_size, "derivative_order": 0, "sigmoid": True})
# define any parameters as usual
model.add_params(params)
# register the previously defined utility and value function
model.register_functions([
    compute_u, compute_value_function
])
# use the value function, replace the args with proper sequence of variables
model.add_equations([
    "V=compute_value_function(*args)"
])
# since we already computed -torch.mean(V) in the value function, we do not apply additional MSE
model.add_hjb_equation("V", loss_reduction=LossReductionMethod.NONE)
if not os.path.exists(f"{model_path}/model.pt"):
    model.train_model(model_path, "model.pt", True)
model.load_model(torch.load(f"{model_path}/model_best.pt", weights_only=False))
```
4. Iterative simulation and plotting:  
To plot the results of this kind of problems, we need to perform iterative sampling, as in the training process.   
```python
for t in range(T+1):
    # sample state variables for the current time step
    SV = ...
    action = model.agents["pi"](SV).item()
    # compute the scaled policy function or value function according to the model setup
```
