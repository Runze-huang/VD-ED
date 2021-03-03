import os
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(Dataset):   
    def __init__(self, images, numbers, lebals):
        self.images = images
        #self.starts = starts
        self.number = numbers
        self.lebal = lebals

    def __len__(self):
        return len(self.number)
    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx],dtype=torch.float)#.to(device)
        number = torch.tensor(self.number[idx] , dtype=torch.float)#.to(device)  
        lebal = self.lebal[idx].float()#.to(device)
        return (image,number,lebal)