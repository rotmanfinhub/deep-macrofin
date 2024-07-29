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
    "refinement_sample_interval": 200,
}

DEFAULT_LEARNABLE_VAR_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "hidden_units": [30, 30, 30, 30],
    "layer_type": LayerType.MLP,
    "activation_type": ActivationType.Tanh,
    "positive": False,
    "derivative_order": 2,
}
```