# this is a wrap up of 05 lesson of pytorch tutorial which is important, here
# example shows how to find linear function coefficient with machine learning - a trivial task but methodology is important 
# numpy based machine learning loop
# step by step torchification of the numpy manual loop:
#   - torch.backward() metod -> getting rid of explicit gradient 
#   - torch.loss() metod -> getting rid of explicit loss function
#   - torch.model()       -> getting rid of explicit mathematical model 
#   - optimizer.step()   -> getting rid of explicit gradient calculation 
#   - optimizer.zero_Grad() -> gettinng rid of explitic weights gradient zeroing 

# numpy based machine learning 
import numpy as np 
import os 
cls = lambda : os.system('CLS') # call to clear outputs in console 

# linear model 
f= lambda x: 2*x
# dataset  
Xii=np.linspace(1,5,5)   # i means it's a dataset
Yii=[f(x) for x in Xii]

# ------------------------------------------------------------------------ numpy based trivial machine learning 44
Xi=Xii
Yi=Yii


# model prediction - forward pass 
def forward(w,x):
    return w*x 
# loss function -> MSE := 1/N * (w*x -y_pred)**2 
def loss(y,yi): 
    return ((y-yi)**2).mean()
# dJ/dw -> d/dw  1/N * (w*x -y_pred)**2  = 1/N 2x(wx-y)
def gradient(x,y,yi):
    return np.dot(2*x,y-yi).mean()

# ml loop params
lr=0.001    # learning rate 
n_iters=100 # iterations 
w=0         # initial weights 

for epoch in range(n_iters):
    # prediction 
    y=forward(w,Xi)
    # loss calculation
    l=loss(y,Yi)
    # graduebt 
    dw = gradient(Xi,y,Yi)
    # update weights 
    w = w- lr *dw 
    
print(l)
# ------------------------------------------------------------------------ torch based example above 
import torch 
# now you need tensors 
Xi=torch.tensor(Xii,dtype=torch.float32)
Yi=torch.tensor(Yii, dtype=torch.float32)


w=torch.tensor([0],dtype=torch.float32,requires_grad=True)



for epoch in range(n_iters*2):
    # prediction 
    y=forward(w,Xi)
    # loss calculation
    l=loss(y,Yi)
    # graduebt 
    l.backward()    # using backward method to calcylate dl/dw gradient 
    # update weights 
    with torch.no_grad():
        w -= lr * w.grad
     # zero the gradients 
    w.grad.zero_()
print(l.item())
    
# ------------------------------------------------------------------------ torchification - using optimizer 
import torch 
import torch.nn as nn 
# now you need tensors 
Xi=torch.tensor(Xii,dtype=torch.float32)
Yi=torch.tensor(Yii, dtype=torch.float32)
w=torch.tensor([0],dtype=torch.float32,requires_grad=True)

optimizer = torch.optim.SGD([w],lr=lr )
for epoch in range(n_iters*2):
    # prediction 
    y=forward(w,Xi)
    # loss calculation
    l=loss(y,Yi)
    # graduebt 
    l.backward()    # using backward method to calcylate dl/dw gradient 
    # update weights 
    optimizer.step()
    optimizer.zero_grad()
print(l.item())

# ------------------------------------------------------------------------ torchification - using loss 
import torch 
import torch.nn as nn 
# now you need tensors 
Xi=torch.tensor(Xii,dtype=torch.float32)
Yi=torch.tensor(Yii, dtype=torch.float32)
w=torch.tensor([0],dtype=torch.float32,requires_grad=True)

optimizer = torch.optim.SGD([w],lr=lr )
loss = nn.MSELoss()
for epoch in range(n_iters*2):
    # prediction 
    y=forward(w,Xi)
    # loss calculation
    l=loss(y,Yi)
    # graduebt 
    l.backward()    # using backward method to calcylate dl/dw gradient 
    # update weights 
    optimizer.step()
    optimizer.zero_grad()
print(l.item())

# ------------------------------------------------------------------------ torchification - using torch linear model instead of explicit one  
import torch 
import torch.nn as nn 
# now you need tensors 
Xi=torch.tensor([[xi] for xi in Xii],dtype=torch.float32)
Yi=torch.tensor([[yi] for yi in Yii], dtype=torch.float32)

loss = nn.MSELoss()
n_samples,n_features=Xi.shape
input_size=n_features
output_size=n_features
model=nn.Linear(input_size,output_size)
optimizer = torch.optim.SGD(model.parameters(),lr=lr )  # notice optimizer uses model.parameter() rather than w tensor, which is not used ata ll 

for epoch in range(n_iters*10):
    # prediction 
    y=model(Xi)
    # loss calculation
    l=loss(y,Yi)
    # graduebt 
    l.backward()    # using backward method to calcylate dl/dw gradient 
    # update weights 
    optimizer.step()
    optimizer.zero_grad()
print(l.item())


# ------------------------------------------------------------------------ torchification -  better data and plotting 
from sklearn import datasets
import torch.nn as nn 
import matplotlib.pyplot as plt
# now you need tensors 
X,Y= datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)
X=torch.from_numpy(X.astype(np.float32))
Y=torch.from_numpy(Y.astype(np.float32))

Xi=X
Yi=Y.view(Y.shape[0],1)


loss = nn.MSELoss()
n_samples,n_features=Xi.shape
input_size=n_features
output_size=n_features
model=nn.Linear(input_size,output_size)
optimizer = torch.optim.SGD(model.parameters(),lr=lr )  # notice optimizer uses model.parameter() rather than w tensor, which is not used ata ll 

for epoch in range(n_iters*10):
    # prediction 
    y=model(Xi)
    # loss calculation
    l=loss(y,Yi)
    # graduebt 
    l.backward()    # using backward method to calcylate dl/dw gradient 
    # update weights 
    optimizer.step()
    optimizer.zero_grad()
print(l.item())

predicted = model(X).detach().numpy()
plt.plot(Xi, Yi, 'ro')
plt.plot(Xi, predicted, 'b')
plt.show()