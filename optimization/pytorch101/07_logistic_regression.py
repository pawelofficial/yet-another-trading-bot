import torch 
import torch.nn as nn 
import numpy as np 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

 
# prepare data : 
bc=datasets.load_breast_cancer() # binary classification problem 
X,Y=bc.data, bc.target 
n_samples, n_features = X.shape 

X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=1234)

# scale features 
sc=StandardScaler() # Z-scoring our features 
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# convert to tensors 
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test  = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

# reshape 
Y_train=Y_train.view(Y_train.shape[0],1)
Y_test = Y_test.view(Y_test.shape[0],1)


# end of preparing data 
# 1) making model 
# f = wx + b, sigmoid at the end 

class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear=nn.Linear(n_input_features,1)
        
    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x) )
        return y_predicted


model=LogisticRegression(n_features)

# 2) loss and optimizer
criterion = nn.BCELoss() # binary cross entropy 
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)


for epoch in range(300):
    # prediction 
    y=model(X_train) # calling the module class invokes forward function fren ! loss calculation
    l=criterion(y,Y_train) # order here is important ! 
    # graduebt 
    l.backward()    # using backward method to calcylate dl/dw gradient 
    # update weights 
    optimizer.step()
    optimizer.zero_grad()
print(l.item())




with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(Y_test).sum() / float(Y_test.shape[0])
    
print(f'{acc:.4f}')