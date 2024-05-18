from pyDOE import lhs
import numpy as np
import torch

torch.set_default_dtype(torch.float32)
params = {
    "gammai": 1.0,
    "gammah":1.0,
    "rhoi": 0.05,
    "rhoh": 0.05,
    "siga" : 0.2, #\sigma^{a}
    "mua":0.04,
    "muO":0.03,
    "n_pop": 3,
    "zetai":1.00005,
    "zetah":1.00005,
    "aa":0.1,
    "batchSize":100,
    "nn_width":30,
    "nn_num_layers":4,
    "neta":100,
    "start_eta":0.01,
    "end_eta":0.99,
    "kappa":10000
}
    
class Net1(torch.nn.Module):
    def __init__(self, nn_width, nn_num_layers,positive=False):
        super(Net1, self).__init__()
        layers = [torch.nn.Linear(1, nn_width), torch.nn.Tanh()]
        for i in range(1, nn_num_layers):
            layers.append(torch.nn.Linear(nn_width, nn_width))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(nn_width, 1))
        self.positive = positive
        self.net = torch.nn.Sequential(*layers)
        torch.nn.init.xavier_normal_(self.net[0].weight)  # Initialize the first linear layer weights

    def forward(self, X):
        output = self.net(X)
        if self.positive: output = torch.nn.functional.softplus(output)  # Apply softplus to the output
        return output
    
def get_derivs_1order(y, x, idx):
    """ Returns the first order derivatives,
        Automatic differentiation used
    """
    dy_dx = torch.autograd.grad(y, x, 
                                create_graph=True, 
                                retain_graph=True, 
                                grad_outputs=torch.ones_like(y))[0][:, idx:idx+1]
    return dy_dx ## Return 'automatic' gradient.
    
print(params['n_pop'])
print(params["batchSize"])


x_np = np.random.uniform(low=[-1., 0., -2.], 
                         high=[1., 3., 2], 
                         size=(params["batchSize"], params["n_pop"]))

all_vars = ["x", "y", "z"]
level_derivatives = {i: {} for i in range(1, params["n_pop"] + 1)}
# first order
for i, var in enumerate(all_vars):
    # note that we must do idx=i as an additional variable, 
    # otherwise, python always capture the "variable", not its "value" at the time of creation.
    # https://stackoverflow.com/questions/33983980/lambda-in-for-loop-only-takes-last-value
    level_derivatives[1][f"f_{var}"] = lambda output, input, idx=i: get_derivs_1order(output, input, idx)

# recursively define higher order derivatives
for derivative_order in range(2, params["n_pop"] + 1):
    prev_derivatives = level_derivatives[derivative_order - 1]
    for prev_str, prev_val in prev_derivatives.items():
        for i, var in enumerate(all_vars):
            # same here, we must pass prev_fun as an additional variable
            # otherwise, we will get recursive overflow
            new_val = lambda output, input, idx=i, prev_fun=prev_val: get_derivs_1order(prev_fun(output, input), input, idx)
            level_derivatives[derivative_order][f"{prev_str}{var}"] = new_val

all_derivatives = {}
for level_derivative in level_derivatives.values():
    all_derivatives.update(level_derivative)

'''
f(x,y,z) = x*y+z
f_x = y
f_y = x
f_z = 1
f_xx = f_yy = 0
f_xy = f_yx = 1
f_xz = f_yz = f_zz = f_zx = f_zy = 0
'''

X = torch.Tensor(x_np).requires_grad_()
f = lambda x :(x[:, 0] * x[:, 1] + x[:, 2]).unsqueeze(-1)
f_eval = f(X)

f_x = all_derivatives["f_x"](f_eval, X)
f_y = all_derivatives["f_y"](f_eval, X)
f_z = all_derivatives["f_z"](f_eval, X)
assert torch.allclose(f_x, X[:, 1:2])
assert torch.allclose(f_y, X[:, 0:1])
assert torch.allclose(f_z, torch.ones((params["batchSize"], 1)))

f_xx = all_derivatives["f_xx"](f_eval, X)
f_xy = all_derivatives["f_xy"](f_eval, X)
f_xz = all_derivatives["f_xz"](f_eval, X)
assert torch.allclose(f_xx, torch.zeros((params["batchSize"], 1)))
assert torch.allclose(f_xy, torch.ones((params["batchSize"], 1)))
assert torch.allclose(f_xz, torch.zeros((params["batchSize"], 1)))

f_yx = all_derivatives["f_yx"](f_eval, X)
f_yy = all_derivatives["f_yy"](f_eval, X)
f_yz = all_derivatives["f_yz"](f_eval, X)
assert torch.allclose(f_yx, torch.ones((params["batchSize"], 1)))
assert torch.allclose(f_yy, torch.zeros((params["batchSize"], 1)))
assert torch.allclose(f_yz, torch.zeros((params["batchSize"], 1)))

f_zx = all_derivatives["f_zx"](f_eval, X)
f_zy = all_derivatives["f_zy"](f_eval, X)
f_zz = all_derivatives["f_zz"](f_eval, X)
assert torch.allclose(f_zx, torch.zeros((params["batchSize"], 1)))
assert torch.allclose(f_zy, torch.zeros((params["batchSize"], 1)))
assert torch.allclose(f_zz, torch.zeros((params["batchSize"], 1)))

# print(y.shape)
# df_d = get_derivs_1order(y, X)
# df_dx = df_d[:, 0:1]
# df_dy = df_d[:, 1:2]
# print(df_dx.shape)
# print(df_dy.shape)

# df_dxd = get_derivs_1order(df_dx, X)
# df_dxdx = df_dxd[:, 0:1]
# df_dxdy = df_dxd[:, 1:2]
# print(df_dxdx.shape)
# print(df_dxdy.shape)

# df_dyd = get_derivs_1order(df_dy, X)
# df_dydx = df_dxd[:, 0:1]
# df_dydy = df_dxd[:, 1:2]
# print(df_dydx.shape)
# print(df_dydy.shape)