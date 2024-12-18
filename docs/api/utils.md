# deep_macrofin.utils

## set_seeds
```py
def set_seeds(seed)
```

Set the random seeds of `random`, `numpy`, and `torch` to the provided seed. It is used by default before training loop.

## plot_loss_df
```py
def plot_loss_df(fn: str=None, loss_df: pd.DataFrame=None, losses_to_plot: list=None, loss_plot_fn: str= "./plot.jpg")
```

Plot the provided loss df, with all losses listed in the losses_to_plot.

**Parameters**:

- fn: **str**, the relative path to loss df csv, default: None
- loss_df: **pd.DataFrame**, the loaded loss df, default: None, at least one of fn and loss_df should not be None.
- losses_to_plot: **List[str]**, the losses to plot, if None, all losses in the df will be plotted, default: None
- loss_plot_fn: **str**, the path to save the loss plot, default: "./plot.jpg"

## Constants

```py
class OptimizerType(str, Enum):
    Adam = "Adam"
    AdamW = "AdamW"
    LBFGS = "LBFGS"

class SamplingMethod(str, Enum):
    UniformRandom = "UniformRandom"
    FixedGrid = "FixedGrid"
    ActiveLearning = "ActiveLearning"
    RARG = "RAR-G"
    RARD = "RAR-D"

DEFAULT_CONFIG = {
    "batch_size": 100,
    "num_epochs": 1000,
    "lr": 1e-3,
    "loss_log_interval": 100,
    "optimizer_type": OptimizerType.AdamW,
    "sampling_method": SamplingMethod.UniformRandom,
    "refinement_rounds": 5,
    "loss_balancing": False,
    "bernoulli_prob": 0.9999,
    "loss_balancing_temp": 0.1,
    "loss_balancing_alpha": 0.999,
    "soft_adapt_interval": -1,
    "loss_soft_attention": False,
}

DEFAULT_CONFIG_TIME_STEP = {
    "batch_size": 100,
    "time_batch_size": None,
    "num_outer_iterations": 100,
    "num_inner_iterations": 5000,
    "lr": 1e-3,
    "optimizer_type": OptimizerType.Adam,
    "min_t": 0.0,
    "max_t": 1.0,
    "outer_loop_convergence_thres": 1e-4,
    "sampling_method": SamplingMethod.FixedGrid,
    "refinement_rounds": 5,
    "loss_balancing": False,
    "bernoulli_prob": 0.9999,
    "loss_balancing_temp": 0.1,
    "loss_balancing_alpha": 0.999,
}

DEFAULT_LEARNABLE_VAR_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "hidden_units": [30, 30, 30, 30],
    "output_size": 1,
    "layer_type": LayerType.MLP,
    "activation_type": ActivationType.Tanh,
    "positive": False,
    "derivative_order": 2,
    "batch_jac_hes": False,
}
```