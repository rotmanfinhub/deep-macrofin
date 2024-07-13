# deep_macrofin.utils

## set_seeds

## plot_loss_df

## Constants

```py
DEFAULT_CONFIG = {
    "batch_size": 100,
    "num_epochs": 1000,
    "lr": 1e-3,
    "loss_log_interval": 100,
    "optimizer_type": OptimizerType.AdamW,
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