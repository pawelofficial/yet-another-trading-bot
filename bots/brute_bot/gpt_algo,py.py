import numpy as np 
import torch
import matplotlib.pyplot as plt 

class FitModel(torch.nn.Module):
    def __init__(self, num_units=10):
        super(FitModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, num_units),
            torch.nn.ReLU(),
            torch.nn.Linear(num_units, num_units)
            ,torch.nn.ReLU()
            ,torch.nn.Linear(num_units, 1)
        )
    def forward(self, x):
        return self.model(x)



x=np.linspace(0,3,10)
y=[i**1.1 for i in x]
# Define the number of units per layer

# Create the model
model = FitModel()

# Define the loss function
loss_fn = torch.nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
optimizer = torch.optim.SGD(model.parameters(),lr=0.001 )

# Convert the data to PyTorch tensors
X = torch.from_numpy(np.array(x).astype(np.float32))
Y = torch.from_numpy(np.array(y).astype(np.float32))

# Reshape the data for the model
Xt = X.view(X.shape[0], 1)
Yt = Y.view(Y.shape[0], 1)

# Train the model
for epoch in range(10000):
    y_pred = model(Xt)
    l = loss_fn(y_pred, Yt)
    l.backward()
    optimizer.zero_grad()
    optimizer.step()

    if epoch/100==epoch//100:
        print(l.item())

print(x)
print(model(Xt).detach().numpy())
print(Yt)

print(l.item())
if 1:
    plt.plot(x,y)
    y_pred=model(Xt).detach().numpy()
    plt.plot(x,y_pred)
    plt.show()