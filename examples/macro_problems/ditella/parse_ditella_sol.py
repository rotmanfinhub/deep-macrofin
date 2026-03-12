import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import json

# The grids are all the same in the provided solution.
v_grid = np.array([0.05, 0.055341152015933544, 0.07130608928453945, 0.0977198966122253, 
      0.13429317879846414, 0.18062523131017238, 0.23620843048442625, 
      0.3004337951595406, 0.3725976588001132, 0.4519093790148387, 0.5375, 
      0.6284317730010949, 0.7237084304844262, 0.8222861014526848, 
      0.923084748314038, 1.025, 1.1269152516859622, 1.2277138985473155, 
      1.3262915695155737, 1.4215682269989052, 1.5125, 1.5980906209851613, 
      1.6774023411998868, 1.7495662048404594, 1.8137915695155737, 
      1.8693747686898277, 1.9157068212015358, 1.9522801033877748, 
      1.9786939107154606, 1.9946588479840666, 2.])
x_grid = np.array([0.05, 
      0.05246514708427702, 0.059833579669787446, 0.0720245676671809, 
      0.08890454406082961, 0.11028856829700262, 0.13594235253127362, 
      0.1655848285351726, 0.19889122713851376, 0.2354966364683871, 
      0.27499999999999997, 0.3169685106158899, 0.3609423525312736, 
      0.40643973913200826, 0.4529621915295559, 0.49999999999999994, 
      0.547037808470444, 0.5935602608679917, 0.6390576474687263, 
      0.6830314893841101, 0.725, 0.7645033635316129, 0.8011087728614862, 
      0.8344151714648274, 0.8640576474687263, 0.8897114317029974, 
      0.9110954559391703, 0.9279754323328191, 0.9401664203302125, 
      0.9475348529157229, 0.95])

required_vars_ditella = {
    "interP": "p", # price
    r"inter\[Sigma]x": "sigx", # diffusion of wealth share
    "omega": "omega", # ratio of value function
    r"inter\[Sigma]p": "sigsigp", # price return diffusion partial
    "interpi": "signxi", # price of risk
    "interr": "r", # risk free rate
    "intere": "e_hat",
    "interc": "c_hat",
}

with open("models/ditella_sol", "r") as f:
    data_lines = f.readlines()

equations: list[str] = []
curr_eq = ""
for line in data_lines:
    if line == " \n":
        equations.append(curr_eq)
        curr_eq = ""
    else:
        curr_eq += line.strip()

x_vals = []
needed_eq = {}

for eq in equations:
    lhs, rhs = eq.split("=")
    lhs = lhs.strip()
    rhs = rhs.strip()
    if "inter" not in lhs or not rhs.startswith("InterpolatingFunction"):
        continue
    # grid_start = rhs.find(",{{") + 1
    # grid_end = rhs.find("}}, {Developer", grid_start) + 2
    # grid = rhs[grid_start:grid_end]
    # grid = grid.replace("{", "[").replace("}", "]").replace("2.]", "2.0]")

    vals_start = rhs.find("961},{") + 5
    vals_end = rhs.find("}", vals_start)
    vals = rhs[vals_start:vals_end+1]
    vals = vals.replace("{", "[").replace("}", "]")
    vals = json.loads(vals)

    res = {"original": vals}
    # grid = json.loads(grid)
    # grid_arr = np.array(grid)
    vals_arr = np.array(vals)
    vals_grid = vals_arr.reshape((len(v_grid), len(x_grid)))
    interp = RegularGridInterpolator((v_grid, x_grid), vals_grid, bounds_error=False, fill_value=None)

    for v in [0.1, 0.25, 0.6]:
        interpolated = interp((np.ones(31) * v, x_grid))
        res[v] = interpolated
    needed_eq[lhs] = res

ditella_res_dict = {}
ditella_res_dict["x_plot"] = x_grid
for i, (k, var_name) in enumerate(required_vars_ditella.items()):
    for v_val in [0.1, 0.25, 0.6]:
        if k == "omega":
            val = needed_eq[r"inter\[Xi]"][v_val] / needed_eq[r"inter\[Zeta]"][v_val]
        elif k == r"inter\[Sigma]p":
            val = 0.0125 + needed_eq[k][v_val]
        else:
            val = needed_eq[k][v_val]
        ditella_res_dict[f"{var_name}_{v_val}"] = val

for k, var in required_vars_ditella.items():
    renamed = {}
    if k == "omega":
        renamed["original"] = np.array(needed_eq[r"inter\[Xi]"]["original"]) / np.array(needed_eq[r"inter\[Zeta]"]["original"])
    elif k == r"inter\[Sigma]p":
        renamed["original"] = 0.0125 + np.array(needed_eq[k]["original"])
    else:
        renamed["original"] = np.array(needed_eq[k]["original"])
    needed_eq[var] = renamed