# here is a sample code which lets you control what do you want to fit in pytorch with weights
# aka the labels are two parabolas on the same x-axis and you can choose if you want your 
# net to fit the upper parabola or lower parabola by multiplying the output that corresponds to each parabola
# by a penalty 
# why would you do that you might ask! the answer is i don't know, i  was just wondering how to suppress
# significance of some values with respect to others that are within same label 
import numpy as np 
import torch
import matplotlib.pyplot as plt 
import torch.nn as nn 

# model 
class FitModel(torch.nn.Module):
    def __init__(self, num_units=10):
        super(FitModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, num_units)
            ,torch.nn.ReLU()
            ,torch.nn.Linear(num_units,num_units)
            ,torch.nn.ReLU()
            ,torch.nn.Linear(num_units,1)
        )
    def forward(self, x):
        return self.model(x)

# first x-axis 
N=100
x=np.linspace(0,10,N)
# two parabolas next to each other 
y=[i**2 for i in x] + [2*i**2 for i in x]
#extending x-axis t oaccomodate for second parabola 
x=np.array(list(x)+list(x))


penalty1=1 # penalty for upper parabola 
penalty2=1     # penalty  for lower parabola  
# building weights 
weights=[penalty1 for i in range(len(x)//2)] +  [penalty2 for i in range(len(x)//2)]


# make some tensors 
W=torch.from_numpy(np.array(weights).astype(np.float32))
X=torch.from_numpy(np.array(x).astype(np.float32))
Y=torch.from_numpy(np.array(y).astype(np.float32))
Xi=X.view(X.shape[0],1)
Yi=Y.view(Y.shape[0],1)
Wi=W.view(W.shape[0],1)

loss = nn.MSELoss()
#loss=nn.CrossEntropyLoss()
n_samples,n_features=Xi.shape
model=FitModel(num_units=n_samples)
optimizer = torch.optim.SGD(model.parameters(),lr=0.00025 )  # notice optimizer uses model.parameter() rather than w tensor, which is not used ata ll 

for epoch in range(1000):
    # prediction 
    y=model(Xi)*Wi   # weights here 
    # loss calculation
    l=loss(y,Yi)
    # graduebt 
    l.backward()    # using backward method to calcylate dl/dw gradient 
    # update weights 
    optimizer.step()
    optimizer.zero_grad()
print(l.item())



if 1:
    predicted = model(Xi).detach().numpy()
    plt.plot(Xi, Yi, 'ro')
    plt.plot(Xi, predicted, 'ob')
    plt.show()