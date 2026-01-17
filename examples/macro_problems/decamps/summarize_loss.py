import os
from itertools import product

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def format_sci(x):
    sci_str = f"{x:.2e}"  # Convert to scientific notation
    base, exp = sci_str.split("e")  # Split into base and exponent
    exp = int(exp)  # Convert exponent to integer to remove leading zeros and '+'
    if exp == 0:
        return f"{base}"
    else:
        return f"${base} \\times 10^{{{exp}}}$"

def generate_table(
    output_dir_base: str,
    plot_dir: str,
    model_type="refinance",  
):
    os.makedirs(plot_dir, exist_ok=True)
    base_nn32_mse = pd.read_csv(f"{output_dir_base}/decamp_{model_type}_251206/mse.csv")
    active_nn32_mse = pd.read_csv(f"{output_dir_base}/decamp_{model_type}_251206_active/mse.csv")
    base_nn64_mse = pd.read_csv(f"{output_dir_base}/decamp_{model_type}_251207/mse.csv")
    active_nn64_mse = pd.read_csv(f"{output_dir_base}/decamp_{model_type}_251207_active/mse.csv")

    first_layer = ["Baseline (32bits)", "Active (32bits)", "Baseline (64bits)", "Active (64bits)",]
    second_layer = ["MSE($F$)", "MAE($c^*$)"]
    cols = [(f, s) for f in first_layer for s in second_layer]
    res_df = pd.DataFrame(index=[0, 1], columns= pd.MultiIndex.from_tuples(cols))
    raw_df_map = {
        "Baseline (32bits)": base_nn32_mse, 
        "Active (32bits)": active_nn32_mse, 
        "Baseline (64bits)": base_nn64_mse, 
        "Active (64bits)": active_nn64_mse
    }
    mse_map = {
        "MSE($F$)": "MSE", 
        "MAE($c^*$)": "MAE"
    }
    for col in cols:
        res_df.loc[0, col] = format_sci(raw_df_map[col[0]].loc[0, mse_map[col[1]]])
        res_df.loc[1, col] = format_sci(raw_df_map[col[0]].loc[1, mse_map[col[1]]])

    ltx = res_df.style.to_latex(column_format="l" + "c"*len(cols), hrules=True, multicol_align="c")
    with open(f"{plot_dir}/loss_{model_type}.tex", "w") as f:
        f.write(ltx)

def loss_overlay(
    output_dir_base: str, # models/decamp
    plot_dir: str,
    model_type="refinance",
    suffix="_251206",
    baseline=False,
):
    fn = "model_baseline_loss.csv" if baseline else "model_loss.csv"
    out_fn = f"{model_type}_model_baseline_loss{suffix}.png" if baseline else f"{model_type}_model_loss{suffix}.png"
    os.makedirs(plot_dir, exist_ok=True)
    loss_dict = pd.read_csv(f"{output_dir_base}/decamp_{model_type}{suffix}/{fn}")
    loss_dict_active = pd.read_csv(f"{output_dir_base}/decamp_{model_type}{suffix}_active/{fn}")
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    ax[0].plot(loss_dict["epoch"], loss_dict["hjbeq_1"], label="Basic NN", linestyle="--", color="#5492ab")
    ax[0].plot(loss_dict_active["epoch"], loss_dict_active["hjbeq_1"], label="Active sampling", linestyle="-", color="#000000")
    ax[0].set_title("HJB Loss", fontsize=16)
    ax[1].plot(loss_dict["epoch"], loss_dict["hjbeq_2"], label="Basic NN", linestyle="--", color="#5492ab")
    ax[1].plot(loss_dict_active["epoch"], loss_dict_active["hjbeq_2"], label="Active sampling", linestyle="-", color="#000000")
    ax[1].set_title("Loss at c*", fontsize=16)

    for a in ax:
        a.set_yscale("log")

    ax[1].legend(loc="upper right", frameon=False, fontsize=14)
    for a in ax:
        a.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{out_fn}")
    
if __name__ == "__main__":
    output_dir_base = "models"
    plot_dir = "models/plots"
    os.makedirs(plot_dir, exist_ok=True)
    generate_table(output_dir_base, plot_dir, "refinance")
    generate_table(output_dir_base, plot_dir, "liquidation")

    for model_type, suffix in product(["refinance", "liquidation"], ["_251206", "_251207"]):
        loss_overlay(output_dir_base, plot_dir, model_type, suffix, True)
        loss_overlay(output_dir_base, plot_dir, model_type, suffix, False)