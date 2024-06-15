import random
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .models import ActivationType, LayerType


class OptimizerType(str, Enum):
    Adam = "Adam"
    AdamW = "AdamW"
    LBFGS = "LBFGS"

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_loss_df(fn: str=None, loss_df: pd.DataFrame=None, losses_to_plot: list=None, loss_plot_fn: str= "./plot.jpg"):
    '''
    Plot the provided loss df, with all losses listed in the losses_to_plot

    Inputs:
    - fn: **str**, the relative path to loss df csv, default: None
    - loss_df: **pd.DataFrame**, the loaded loss df, default: None
    - losses_to_plot: **List[str]**, the losses to plot, 
    if None, all losses in the df will be plotted, default: None
    - loss_plot_fn: **str**, the path to save the loss plot, default: "./plot.jpg"
    '''
    assert fn is not None or loss_df is not None, "one of fn or loss_df should not be None"
    if fn is not None:
        loss_df = pd.read_csv(fn)
    loss_df = loss_df.dropna().reset_index(drop=True)
    epochs = loss_df["epoch"]
    if losses_to_plot is None:
        losses_to_plot = list(loss_df.columns)
        losses_to_plot.remove("epoch")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for loss in losses_to_plot:
        ax.plot(epochs, loss_df[loss], label=loss)
    
    ax.set_yscale('log')
    ax.set_xlabel("epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Loss Plot")
    plt.tight_layout()
    plt.savefig(loss_plot_fn)
    plt.show()

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