# epoch -> one epoch is a forward and backward pass of ALL training samples
# batch_size = number of training samples in one forward and backward pass 
# no of iterations - number of passes where each pass uses [batch_size] number of samples
#   if we have 100 samples and batch_Size = 20 --> 5 iterations for 1 epoch  


from email import header
import torch 
import torchvision 
from torch.utils.data import Dataset, DataLoader 
import numpy as np 
import math 


class WineDataset(Dataset):
    def __init__(self):# data loading 
        xy=np.loadtxt('./wine.csv',delimiter=",",dtype=np.float32, skiprows=1)
        # split dataset 
        self.x = xy[:,1:] # skip first column 
        self.y=xy[:,[0]] # all samples, first column -> n_samples x 1 column          
        self.x=torch.from_numpy(self.x)
        self.y=torch.from_numpy(self.y)
        self.n_samples = xy.shape[0]

        
    def __getitem__(self, index):# dataset [0]
        return self.x[index],self.y[index] # returns a tuple  
    
    def __len__(self): # len(dataset )
        return self.n_samples 
    
    
    
if __name__ == '__main__':
    dataset=WineDataset()
    dataloader=DataLoader(dataset=dataset,batch_size=1000,shuffle=True,num_workers=1)

    dataiter = iter(dataloader)
    data=dataiter.next()
    features,labels=data  # batch size 4 -> 4 feature vectors and 4 labels 
#    print(features,labels)
    print(features,labels)
    print(dataiter)
    print(features.shape,labels.shape)
    
    
    
    
# dummy training loop 
    num_epochs = 2 
    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples / 4 ) 
    print(total_samples)
    print(n_iterations)
    
    
    
    
    for epoch in range(num_epochs):
        for i,(inputs,labels) in enumerate(dataloader):
            # forward pass 
            # backward pass 
            # update weights 
            if (i+1)%5 == 0: 
                print(f'epoch {epoch + 1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}   ')
    

# vision datasets:    
#torchvision.datasets.MNIST()
# fashion-mnist, cifar, colo 