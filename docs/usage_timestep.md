# Time Stepping Scheme

The time stepping scheme implementation in `PDEModelTimeStep` is mostly identical to the basic `PDEModel`. Here, we only highlight the differences.

## Define PDE System
Firstly, we import `PDEModelTimeStep` ([API](./api/pde_model_time_step.md#pdemodeltimestep)) from the package to define the problem to be solved. The following code snippet defines a PDE model with default training settings and name `example`:

```py
from deep_macrofin import PDEModelTimeStep
pde_model = PDEModelTimeStep("model_name")
```

`PDEModelTimeStep` takes three parameters: `name: str, config: Dict[str, Any] = DEFAULT_CONFIG_TIME_STEP, latex_var_mapping: Dict[str, str] = {}`. Only `name` is required. The trained models will be saved with filename `name` by default.

### Training Configs

The default training configs set batch size = 100, learning rate = $10^{-3}$, and Adam optimizer. In this setting, data points are sampled from a fixed grid. 
```py
DEFAULT_CONFIG_TIME_STEP = {
    "batch_size": 100,
    "time_batch_size": None,
    "num_outer_iterations": 100,
    "num_inner_iterations": 5000,
    "min_inner_iterations": 1000,
    "lr": 1e-3,
    "optimizer_type": OptimizerType.Adam,
    "min_t": 0.0,
    "max_t": 1.0,
    "time_boundary_loss_reduction": LossReductionMethod.MSE,
    "outer_loop_convergence_thres": 1e-4,
    "sampling_method": SamplingMethod.UniformRandom,
    "refinement_rounds": 5,
    "loss_balancing": False,
    "bernoulli_prob": 0.9999,
    "loss_balancing_temp": 0.1,
    "loss_balancing_alpha": 0.999,
}
```

This dictionary can be imported:
```py
from deep_macrofin import DEFAULT_CONFIG_TIME_STEP
```

## Set Initial Guess

```py
# This sets an initial guess of 7 uniformly on the domain for variable p.
pde_model.set_initial_guess({"p": 7.0})

# This sets an initial guess using a function for variable q.
def q_init(SV):
    x = SV[:, :-1]
    return torch.sum(x**2, dim=1, keepdim=True)
pde_model.set_initial_guess({"q": q_init})
```

Set the initial guess (uniform value across the state variable domain) for agents or endogenous variables. This is the boundary condition at $t=T$ in the first time iteration.

Note that in most cases, the initial guess has no actual effects on the training. The default value is 1 for any neural network models.

## Differences from PDEModel

- Only Fixed Grid Sampling and Uniform Random Sampling are supported in time stepping scheme.

- The name `t` is reserved for implicit time dimension (a state variable). Therefore, it cannot be used as user defined variables. It should not be passed into the state variable list by the user either. The only thing the user can change for `t` is `min_t` and `max_t`, defining the size of the finite time interval to simulate infinite horizon.

- With the additional `t`, the actual problem dimension is $N+1$, where $N$ is the number of state variables defined by the user. 
    - For fixed grid sampling: The final sample is of shape $(B^{N+1}, N+1)$.
    - For uniform sampling with a non-nagative `time_batch_size`: The final sample is of shape $(B*B_t, N+1)$ , where $B_t$ is the `time_batch_size` (default to $B$ when `None` is provided).
    - For uniform sampling with a `time_batch_size`$\leq 1$: The final sample is of shape $(B, N+1)$. In this case, the timesteps are uniformly iid sampled together with the state variables.
    - For RARG sampling, it is default to uniform sampling a `time_batch_size`$\leq 1$. Additional points to help learning are sampled `refinement_rounds` times. The sampling epochs are equally spaced over the inner loops.

- Currently, only loss balancing is implemented for time stepping scheme.

- During training, three models are saved
   - `{file_prefix}_temp_best.pt`: the best model within current outer loop
   - `{file_prefix}_best.pt`: the best model over all the past outer loops
   - `{file_prefix}.pt`: the final model in the final outer loop.

- Additional parameter `variables_to_track` for `train_model`:
   - By default, the changes in each timestep for all endogenous and value variables are tracked.
   - This additional parameter allows the user to specify additional variables to track. The variables to track must have already been defined by `equations` and available in `variable_val_dict` of the `PDEModelTimeStep` object.