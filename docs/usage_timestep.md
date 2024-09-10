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

The default training configs set batch size = 100, learning rate = $10^{-3}$, and Adam optimizer. In this setting, loss will be logged to a csv file every 100 epochs during training (`loss_log_interval`), and data points are sampled from a fixed grid. 
```py
DEFAULT_CONFIG_TIME_STEP = {
    "batch_size": 100,
    "num_outer_iterations": 100,
    "num_inner_iterations": 5000,
    "lr": 1e-3,
    "loss_log_interval": 100,
    "optimizer_type": OptimizerType.Adam,
    "min_t": 0.0,
    "max_t": 1.0,
    "outer_loop_convergence_thres": 1e-4
}
```

This dictionary can be imported:
```py
from deep_macrofin import DEFAULT_CONFIG_TIME_STEP
```

### Differences from PDEModel

- Only Fixed Grid Sampling is supported in time stepping scheme.

- The name `t` is reserved for implicit time dimension (a state variable). Therefore, it cannot be used as user defined variables. It should not be passed into the state variable list by the user either. The only thing the user can change for `t` is `min_t` and `max_t`, defining the size of the finite time interval to simulate infinite horizon.

- With the additional `t`, the actual problem dimension is $N+1$, where $N$ is the number of state variables defined by the user. The final sample is of shape $(B^{N+1}, N+1)$

- Currently, no loss weight adjustment algorithm is implemented for time stepping scheme.
