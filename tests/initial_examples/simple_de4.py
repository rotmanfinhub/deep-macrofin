from basic_utils import *
from tqdm import tqdm


params = {
    "batchSize": 50,
    "n_pop": 2
}

'''
This example tries to solve the following heat equation:
du/dt = 0.4 d^2u/dx^2, x\in [0,1], t\in[0,1], 
u(0,t)=u(1,t)=0, u(x,0)=sin(\pi x)

The solution is u(x,t)=e^{-\pi^2*0.4t} sin(\pi x)
'''
f = Net1(1, 30, 4, input_names=["x", "t"], model_name="u")
f.train()
optimizer = torch.optim.AdamW(f.parameters(), lr=1e-3)

print("{0:=^80}".format("Training"))
for epochs in tqdm(range(1000)):
    optimizer.zero_grad()
    x_np = np.random.uniform(low=[0., 0.], 
                         high=[1., 1.], 
                         size=(params["batchSize"], params["n_pop"]))
    X = torch.Tensor(x_np)
    loss_1 = f.derivatives["u_t"](X) - 0.4 * f.derivatives["u_xx"](X) # "u_t(X) = 0.4 u_xx(X)"

    t_space = torch.Tensor(np.random.uniform(low=0., high=1., size=params["batchSize"]))
    x_1 = torch.zeros((params["batchSize"], params["n_pop"]))
    x_1[:, 1] = t_space
    x_2 = torch.ones((params["batchSize"], params["n_pop"]))
    x_2[:, 1] = t_space
    loss_2 = f(x_1) # u(0,t) = 0
    loss_3 = f(x_2) # u(1,t) = 0

    x_space = torch.Tensor(np.random.uniform(low=0., high=1., size=params["batchSize"]))
    x_3 = torch.zeros((params["batchSize"], params["n_pop"]))
    x_3[:, 0] = x_space
    loss_4 = f(x_3) - torch.sin(torch.pi * x_3[:, 0:1])

    loss = torch.mean(torch.square(loss_1)) + torch.mean(torch.square(loss_2)) \
            + torch.mean(torch.square(loss_3)) + torch.mean(torch.square(loss_4))
    loss.backward()
    optimizer.step()

print("{0:=^80}".format("Evaluation"))
f.eval()
x_np = np.random.uniform(low=[0., 0.], 
                         high=[1., 1.], 
                         size=(params["batchSize"], params["n_pop"]))
X = torch.Tensor(x_np)
loss_1 = f.derivatives["u_t"](X) - 0.4 * f.derivatives["u_xx"](X) # "u_t(X) = 0.4 u_xx(X)"

t_space = torch.Tensor(np.random.uniform(low=0., high=1., size=params["batchSize"]))
x_1 = torch.zeros((params["batchSize"], params["n_pop"]))
x_1[:, 1] = t_space
x_2 = torch.ones((params["batchSize"], params["n_pop"]))
x_2[:, 1] = t_space
loss_2 = f(x_1) # u(0,t) = 0
loss_3 = f(x_2) # u(1,t) = 0

x_space = torch.Tensor(np.random.uniform(low=0., high=1., size=params["batchSize"]))
x_3 = torch.zeros((params["batchSize"], params["n_pop"]))
x_3[:, 0] = x_space
loss_4 = f(x_3) - torch.sin(torch.pi * x_3[:, 0:1])

loss = torch.mean(torch.square(loss_1)) + torch.mean(torch.square(loss_2)) \
        + torch.mean(torch.square(loss_3)) + torch.mean(torch.square(loss_4))
print("Loss 1: ", torch.mean(torch.square(loss_1)))
print("Loss 2: ", torch.mean(torch.square(loss_2)))
print("Loss 3: ", torch.mean(torch.square(loss_3)))
print("Loss 4: ", torch.mean(torch.square(loss_4)))

torch.save(f.state_dict(), "simple_de4.pt")

