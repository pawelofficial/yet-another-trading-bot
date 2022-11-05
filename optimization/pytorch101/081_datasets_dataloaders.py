# here is this tutorial 
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html


# Dataset class stores samples and their labels
# Dataloader wraps iterable around Dataset 

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}


if 0:# plot 9 images ( torch tensors as images) and their respective labels 
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx] # use i instead of sample_idx to have images constant 
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")    
    if 0:
        plt.show()
    
    
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file) # file with annotations to the images 
        self.img_dir = img_dir                          # directory where images are stored 
        self.transform = transform                      # transform  ? 
        self.target_transform = target_transform        # target tramsfpr,  

    def __len__(self):                                  # returns number of samples in a dataset 
        return len(self.img_labels)

    def __getitem__(self, idx):                         # loads and returns a sample from the dataset at the given index idx
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# iterating through data loader 
# Display image and label.
if 1: # access features/labels via iterator on top of a dataloader:
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.title(f'{labels_map[label.item()]}')
    plt.title=title=label.item()
    print(f"Label: {label} {labels_map[label.item()]} ")
    plt.show()
    
# toch data:
torchvision.datasets.MNIST()    
