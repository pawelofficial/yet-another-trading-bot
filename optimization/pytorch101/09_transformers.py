import torch 
import torchvision 
from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np 

# check ouyt pytorch.org./docs/stable/torchvision/transforms.html

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)


# our own custom wine dataset 
class WineDataset(Dataset):
    def __init__(self, transform = None ):# data loading 
        xy=np.loadtxt('./wine.csv',delimiter=",",dtype=np.float32, skiprows=1)
        # split dataset 
        self.x = xy[:,1:] # skip first column 
        self.y=xy[:,[0]] # all samples, first column -> n_samples x 1 column          
        # cast to tensors 
        if 0: # in this tutorial we write our own transformation that casts stuff to tensor 
            self.x=torch.from_numpy(self.x)
            self.y=torch.from_numpy(self.y)
        self.n_samples = xy.shape[0]
        
        self.transform=transform 
        
        
    def __getitem__(self, index):
        sample =  self.x[index],self.y[index]  
        if self.transform: 
            sample = self.transform(sample)
        return sample 
     
    
    def __len__(self): # len(dataset )
        return self.n_samples 
    
    
    
class myToTensor():
    def __call__(self,sample): # 23 have to implement __call__ method 
        inputs, targets = sample
        return torch.from_numpy(inputs),torch.from_numpy(targets)
        
# pass our own transformation to dataset 

dataset=WineDataset(transform=myToTensor())
first_data =dataset[0]
features,labels=first_data

print(type(features),type(labels))

class MulTransform():
    def __init__(self,factor):
        self.factor=factor 
    def __call__(self,sample):
        inputs,target=sample 
        inputs *=self.factor
        return inputs,target


# let's apply our very own transform class to our very own dataset 
dataset=WineDataset(transform=None)
first_data =dataset[0]
features,labels=first_data
print(features)
print(type(features),type(labels))

# let's do composed transform 
composed_transform=torchvision.transforms.Compose( [myToTensor(),MulTransform(2)])

dataset2=WineDataset(transform=composed_transform)
first_data =dataset2[0]
features,labels=first_data
print(type(features),type(labels))
print(features) # labels should be multiplied 