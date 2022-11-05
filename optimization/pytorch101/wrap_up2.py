#another wrap up to wrap up cool stuff in one script 

from email import header
import torch 
import torchvision 
from torch.utils.data import Dataset, DataLoader 
import numpy as np 
from sklearn import datasets
import math 
import torch.nn as nn 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
import datetime 

class WineDataset(Dataset):
    def __init__(self) -> None:
        data=np.loadtxt('./wine.csv',delimiter=",",dtype=np.float32, skiprows=1)
        self.x=data[:,1:] # first column is y 
        self.y=data[:,0]
        self.x=torch.from_numpy(self.x.astype(np.float32))             # torch.Size([178, 13])
        self.y=torch.from_numpy(self.y.astype(np.float32))           # torch.Size([178])
        self.n_features=data.shape[1]-1 # 13 
        self.n_samples=data.shape[0]        # 178 
    def __getitem__(self, index):
        return self.x[index],self.y[index] # returns a tuple  
    
    def __len__(self): # len(dataset )
        return self.n_samples 
    
class BCDataset(Dataset):
    def __init__(self) -> None:
        data=datasets.load_breast_cancer() # binary classification problem 
        X,Y=data.data, data.target  
        self.x=torch.from_numpy(X.astype(np.float32))        # torch.Size([569, 30])
        self.y=torch.from_numpy(Y.astype(np.float32))                     # torch.Size([569])
        
        
        
        self.n_features=self.x.shape[1] # 30 
        self.n_samples=self.y.shape[0]      # 569

        
    def __getitem__(self, index):
        return self.x[index],self.y[index] # returns a tuple  
    
    def __len__(self): # len(dataset )
        return self.n_samples 
    
class LogisticRegression(nn.Module):
    def __init__(self,n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear=nn.Linear(n_input_features,1)
        
    def forward(self,x):
        y_predicted = torch.sigmoid(self.linear(x) )
        return y_predicted


if __name__ == '__main__':
    nw=datetime.datetime.now()
    how_long = lambda : (datetime.datetime.now() - nw ).total_seconds()
    
    dataset=WineDataset()
    dataset=BCDataset()
    print(dataset.n_features,dataset.n_samples) # 13 178

    #dataset=BCDataset()
    dataloader=DataLoader(dataset=dataset,batch_size=1000,shuffle=True,num_workers=1)
    model=LogisticRegression(dataset.n_features)
    criterion = nn.BCELoss() # binary cross entropy 
    optimizer = torch.optim.SGD(model.parameters(),lr=0.025)
    
    for epoch in range(10):
        print(how_long())
        for i,(features,labels) in enumerate(dataloader):
            print('loop', how_long())
            y=model(features) # calling the module class invokes forward function fren ! loss calculatio

            labels=labels.view(labels.shape[0],1)

            l=criterion(y,labels) # order here is important ! 
            # graduebt 
            l.backward()    # using backward method to calcylate dl/dw gradient 
            # update weights 
            optimizer.step()
            optimizer.zero_grad()


    with torch.no_grad():
        y_predicted = model(features)
        y_predicted_cls = y_predicted.round()
        acc = y_predicted_cls.eq(labels).sum() / float(labels.shape[0])
    print(acc)        
    