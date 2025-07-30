import json
import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from deep_macrofin import (ActivationType, Comparator, OptimizerType, PDEModel, LayerType,
                           set_seeds)

plt.rcParams["font.size"] = 20
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["lines.markersize"] = 10

n_dim = 2     # number of assets
BASE_DIR = "./models/bs"
PLOT_DIR = os.path.join(BASE_DIR, "plots")
PREFIX = f"bs_{n_dim}assets"
os.makedirs(PLOT_DIR, exist_ok=True)

CONFIGS = {
    "MLP": {"optimizer_type": OptimizerType.Adam, "num_epochs": 5000},
}
MODEL_CONFIGS = {
    "MLP": [
        {
            "activation_type": ActivationType.SiLU, 
            "batch_jac_hes": True
        },
        {
            "layer_type": LayerType.DGM,
            "activation_type": ActivationType.ReLU,
            "hidden_units": [30, 30, 30],
            "batch_jac_hes": True,
        },
        {
            "layer_type": LayerType.ResNet,
            "activation_type": ActivationType.ReLU,
            "hidden_units": [32, 32, 32],
            "batch_jac_hes": True,
        },
    ]
}

def mc_simulation(sigma, r, K, rho, S_plot, n_simulations = 100000):
    S0 = np.ones(n_dim)
    option_values = np.zeros_like(S_plot)
    for i in range(len(S_plot)):
        S0[0] = S_plot[i]
        # Correlation matrix and Cholesky decomposition
        correlation_matrix = rho * np.ones((n_dim, n_dim)) + (1 - rho) * np.eye(n_dim)
        L = np.linalg.cholesky(correlation_matrix)

        # Generate correlated random normal variables
        z = np.random.normal(size=(n_simulations, n_dim))
        correlated_z = z @ L.T

        # Simulate asset prices at maturity
        drift = (r - 0.5 * sigma**2) * T
        diffusion = sigma * np.sqrt(T) * correlated_z
        S_T = S0 * np.exp(drift + diffusion)

        # Compute the payoff for a basket call option
        basket_prices = S_T.mean(axis=1)  # Average price of the basket
        payoffs = np.maximum(basket_prices - K, 0)

        # Discount the average payoff to present value
        option_values[i] = np.exp(-r * T) * np.mean(payoffs)
    return {
        "S": S_plot,
        "V": option_values
    }

def get_model(model_config, sigma, r, K, rho, n_dim=10):
    set_seeds(0)
    model = PDEModel(PREFIX, {"batch_size": n_dim * 100, "optimizer_type": OptimizerType.Adam, "num_epochs": 5000})
    sv_list = [f"S{i}" for i in range(n_dim)] + ["t"]
    sv_constraints = {f"S{i}": [0., 1.] for i in range(n_dim)}
    sv_constraints["t"] = [0., 1.]
    model.set_state(sv_list, sv_constraints)
    model.add_endog("V", model_config)
    model.add_params({
        "sig": sigma,
        "r": r,
        "K": K
    })
    model.add_equation(f"rho = (torch.ones(({n_dim}, {n_dim}), device='{model.device}') * {rho}).fill_diagonal_(1)")
    model.add_equation("S = SV[:, :-1]")
    model.add_equation("V_S = V_Jac[..., :-1]") # (B, O, I-1)
    model.add_equation("V_t = V_Jac[:, 0, -1:]") # (B, 1)
    model.add_equation("V_SS = V_Hess[:, :, :-1, :-1]") # (B, O, I-1, I-1)
    model.add_endog_equation(r"V_t + r*torch.einsum('bi,bki->bk', S, V_S) + sig**2/2 * torch.einsum('bi,il,bklj,bj->bk', S, rho, V_SS, S) - r * V=0")

    maturity_sv = torch.ones((n_dim * 100, n_dim + 1))
    maturity_sv[:, 0:-1] = torch.Tensor(np.random.uniform(low=[0] * n_dim, high=[1] * n_dim, size=(n_dim * 100, n_dim)))
    maturity_prices = torch.nn.functional.relu(torch.mean(maturity_sv[:, 0:-1], dim=1, keepdim=True) - K)

    model.add_endog_condition("V", 
                            "V(SV)", {"SV": maturity_sv.to(model.device)},
                            Comparator.EQ,
                            "maturity_prices", {"maturity_prices": maturity_prices.to(model.device)},
                            label="bc1")
    return model

def plot_solution(plot_dict_mc: Dict[str, Any],
                plot_dict_mlp: Dict[str, Any]):
    # Plotting the result
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for plot_dict, label, color, ls, marker in [(plot_dict_mc, "MC Simulation", "red", "-", "o"),
                                                (plot_dict_mlp, "MLP", "blue", "-", "x")]:
        ax.plot(plot_dict["S"], plot_dict["V"], label=label, color=color, marker=marker, linestyle=ls)
    ax.set_xlabel("$S$")
    ax.set_ylabel("$V(0, S)$")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{PREFIX}_V.jpg"))

def compute_errors(plot_dict_analytic: Dict[str, Any], 
                  plot_dict_mlp: Dict[str, Any]):
    error_dict = {}
    for plot_dict, label in [(plot_dict_mlp, "MLP")]:
        mse = np.mean((plot_dict["V"].reshape(-1) - plot_dict_analytic["V"]) ** 2)
        error_dict[label] = mse
    with open(os.path.join(BASE_DIR, f"mse_{n_dim}assets.json"), "w") as f:
        f.write(json.dumps(error_dict, indent=True))

def plot_loss():
    LOSS_LABELS = {
        "endogeq_1": "PDE Loss",
        "endogvar_V_cond_bc1": "Boundary Condition",
        "total_loss": "Total Loss"
    }
    loss_df = pd.read_csv(os.path.join(BASE_DIR, f"MLP_{n_dim}asset", "model_loss.csv"))
    loss_df = loss_df.dropna().reset_index(drop=True)
    try:
        epochs = loss_df["epoch"]
    except:
        epochs = loss_df["index"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    losses_to_plot = list(loss_df.columns)
    losses_to_plot.remove("epoch")
    for loss in losses_to_plot:
        ax.plot(epochs, loss_df[loss], label=LOSS_LABELS[loss])
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{PREFIX}_loss.jpg"))
    plt.close()
    
if __name__ == "__main__":
    print("{0:=^80}".format(f"Black-Scholes {n_dim+1}D ({n_dim} Assets)"))
    S_max = 1    # Maximum stock price
    K = 0.5        # Strike price
    T = 1          # Time to maturity
    r = 0.05       # Risk-free rate
    sigma = 0.2    # Volatility
    rho = 0.5      # Process Correlations
    S_plot = np.linspace(0.01, 0.99, 50)

    zero_t = torch.zeros((S_plot.shape[0], n_dim + 1))
    zero_t[:, 0] = torch.Tensor(S_plot)
    zero_t[:, 1:-1] = torch.ones((S_plot.shape[0], n_dim-1))

    print("{0:=^40}".format("DeepMacroFin"))
    for i, model_config in enumerate(MODEL_CONFIGS["MLP"]):
        PREFIX = f"bs_{n_dim}assets_{i}"
        print("{0:=^40}".format(f"Training {model_config}"))
        curr_dir = os.path.join(BASE_DIR, f"MLP_{n_dim}asset")
        os.makedirs(curr_dir, exist_ok=True)
        model = get_model(model_config, sigma, r, K, rho, n_dim)
        if not os.path.exists(f"{curr_dir}/model_{i}.pt"):
            model.train_model(curr_dir, f"model_{i}.pt", True)
            model.load_model(torch.load(f"{curr_dir}/model_{i}_best.pt", weights_only=False))
        else:
            model.load_model(torch.load(f"{curr_dir}/model_{i}_best.pt", weights_only=False, map_location=model.device))
        plot_dicts = {
            "S": S_plot,
            "V": model.endog_vars["V"].forward(zero_t).detach().cpu().numpy().reshape(-1)
        }

        print("{0:=^40}".format("Monte-Carlo Simulation"))
        plot_dicts_mc = mc_simulation(sigma, r, K, rho, S_plot)

        print("{0:=^40}".format("Plotting"))
        plot_solution(plot_dicts_mc, plot_dicts)
        compute_errors(plot_dicts_mc, plot_dicts)
        plot_loss()