from basic_utils import *
from tqdm import tqdm

params = {
    "batchSize": 50,
    "n_pop": 1
}

'''
This example tries to solve the following ODE:
dy/dx = 2*x, y(0)=1

The solution is y=x^2+1
'''
f = Net1(1, 30, 4)
f.train()
optimizer = torch.optim.AdamW(f.parameters(), lr=1e-3)

print("{0:=^80}".format("Training"))
for epochs in tqdm(range(1000)):
    optimizer.zero_grad()
    x_np = np.random.uniform(low=[-2.,], 
                         high=[2.,], 
                         size=(params["batchSize"], params["n_pop"]))
    X = torch.Tensor(x_np)
    loss_1 = f.derivatives["f_x"](X) - 2 * X # "f_x(X)-2*X"
    loss_2 = f(torch.zeros((1, 1))) - 1

    loss = torch.mean(torch.square(loss_1)) + torch.mean(torch.square(loss_2))
    loss.backward()
    optimizer.step()

print("{0:=^80}".format("Evaluation"))
f.eval()
x_np = np.random.uniform(low=[-1.,], 
                         high=[1.,], 
                         size=(params["batchSize"], params["n_pop"]))
X = torch.Tensor(x_np)
loss_1 = torch.mean(torch.square(f.derivatives["f_x"](X) - 2 * X))
loss_2 = torch.mean(torch.square(f(torch.zeros((1, 1))) - 1))
print("Loss 1: ", loss_1)
print("Loss 2: ", loss_2)

torch.save(f.state_dict(), "simple_de1.pt")

