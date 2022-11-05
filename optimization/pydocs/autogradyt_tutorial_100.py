
# https://pytorch.org/tutorials/beginner/introyt/autogradyt_tutorial.html
import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import torch.nn as nn 

BATCH_SIZE = 16
DIM_IN = 1000
HIDDEN_SIZE = 100
DIM_OUT = 10

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.layer1 = torch.nn.Linear(1000, 100)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)

model = TinyModel()


optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
prediction = model(some_input)
loss_fn=nn.MSELoss()

for i in range(0, 500):
    prediction = model(some_input)
    loss=loss_fn(prediction,ideal_output)
    print(loss)
    loss.backward()
    optimizer.step()
    print(loss)
