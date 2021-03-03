import os

import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(Dataset):  

    def __init__(self, images, xs , ys):
        self.images = images
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx],dtype=torch.float)#.to(device)
        x = torch.tensor(self.xs[idx],dtype=torch.float)#.to(device)
        y = torch.tensor(self.ys[idx] , dtype=torch.float)#.to(device)  
        return (image, x, y)