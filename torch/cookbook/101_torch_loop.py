# this script shows a basic learning loop achieved in pytorch  
import sys 
sys.path.append("..")
from myfuns import myfuns as mf
from sklearn import datasets
import torch 
import numpy as np 
import torch.nn as nn 
import matplotlib.pyplot as plt

#1.  get data 
X,Y= datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)
#2.  put data into torch tensors 
X=torch.from_numpy(X.astype(np.float32)) # torch.Size([100, 1])
Y=torch.from_numpy(Y.astype(np.float32)) # torch.Size([100])
#3. make a view to fit shapes of labels to features 
Xi=X
Yi=Y.view(Y.shape[0],1) # torch.Size([100, 1])
#4. define loop parameters
n_iters=1000                        # iterations 
lr=0.01                             # kearbubg rate 
loss = nn.MSELoss()                 # loss function 
n_samples,n_features=Xi.shape       # no of features 
model=nn.Linear(in_features=n_features,out_features=n_features)  # pytorch model type 
optimizer = torch.optim.SGD(model.parameters(),lr=lr )           # optimizer type 

# 5. learning loop 
for epoch in range(n_iters): 
    y=model(Xi)             # make prediction 
    l=loss(y,Yi)            # calculate loss - loss(input,target)
    l.backward()            # do backward step - dl/dw  
    optimizer.step()        # optimize weights 
    optimizer.zero_grad()   # clear optimizer grad


# take a look at results 
print(f'loss: {l}')
predicted = model(X).detach().numpy()
plt.plot(Xi, Yi, 'ro')
plt.plot(Xi, predicted, 'b')
plt.show()