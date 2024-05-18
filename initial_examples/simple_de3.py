from basic_utils import *
from tqdm import tqdm

params = {
    "batchSize": 50,
    "n_pop": 2
}

'''
This example tries to solve the following PDE:
\nabla^2 T = T_xx + T_yy = 0, T(x,0) = T(x,pi) = 0, T(0, y)=1

The solution is T(x,y) = 2/pi arctan((sin y) / (sinhx))
'''
model = Net1(1, 30, 4, input_names=["x", "y"], model_name="T")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("{0:=^80}".format("Training"))
for epochs in tqdm(range(1000)):
    optimizer.zero_grad()
    x_np = np.random.uniform(low=[0., 0.], 
                         high=[3., np.pi], 
                         size=(params["batchSize"], params["n_pop"]))
    X = torch.Tensor(x_np)
    loss = 0
    # nabla^2 T = 0
    nabla_T = model.derivatives["T_xx"](X) + model.derivatives["T_yy"](X)
    loss += torch.mean(torch.square(nabla_T)) 
    
    # T(x,0) = 0
    all_x = np.linspace(0, 3., params["batchSize"])
    bd1 = torch.zeros((params["batchSize"], params["n_pop"]))
    bd1[:, 0] = torch.Tensor(all_x)

    # T(x, pi) = 0
    bd2 = torch.zeros((params["batchSize"], params["n_pop"]))
    bd2[:, 0] = torch.Tensor(all_x)
    bd2[:, 1] = torch.pi

    loss += torch.mean(torch.square(model(bd1)))
    loss += torch.mean(torch.square(model(bd2)))

    # T(0, y) = 1
    all_y = np.linspace(0, np.pi, params["batchSize"])
    bd3 = torch.zeros((params["batchSize"], params["n_pop"]))
    bd3[:, 1] = torch.Tensor(all_y)
    bd3_pred = model(bd3)
    loss += torch.mean(torch.square(bd3_pred - torch.ones_like(bd3_pred)))

    loss.backward()
    optimizer.step()

print("{0:=^80}".format("Evaluation"))
model.eval()
x_np = np.random.uniform(low=[0., 0.], 
                         high=[3., np.pi], 
                         size=(params["batchSize"], params["n_pop"]))
X = torch.Tensor(x_np)
loss = 0
# nabla^2 T = 0
nabla_T = model.derivatives["T_xx"](X) + model.derivatives["T_yy"](X)
loss += torch.mean(torch.square(nabla_T)) 
print("Loss nabla:", loss)

# T(x,0) = 0
all_x = np.linspace(0, 3, params["batchSize"])
bd1 = torch.zeros((params["batchSize"], params["n_pop"]))
bd1[:, 0] = torch.Tensor(all_x)
bd1_loss = torch.mean(torch.square(model(bd1)))

# T(x, pi) = 0
bd2 = torch.zeros((params["batchSize"], params["n_pop"]))
bd2[:, 0] = torch.Tensor(all_x)
bd2[:, 1] = torch.pi
bd2_loss = torch.mean(torch.square(model(bd2)))

# T(0, y) = 1
all_y = np.linspace(0, np.pi, params["batchSize"])
bd3 = torch.zeros((params["batchSize"], params["n_pop"]))
bd3[:, 1] = torch.Tensor(all_y)
bd3_pred = model(bd3)
bd3_loss = torch.mean(torch.square(bd3_pred - torch.ones_like(bd3_pred)))
loss += bd1_loss + bd2_loss + bd3_loss
print("bd1 loss:", bd1_loss)
print("bd2 loss:", bd2_loss)
print("bd3 loss:", bd3_loss)

torch.save(model.state_dict(), "simple_de3.pt")

