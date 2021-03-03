import os

import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(Dataset):   
    def __init__(self, inputs, images):
        self.image = images
        #self.starts = starts
        self.input = inputs

    def __len__(self):
        return len(self.input)
    def __getitem__(self, idx):
        image = torch.tensor(self.image[idx],dtype=torch.float)#.to(device)
        input = torch.tensor(self.input[idx] , dtype=torch.float)#.to(device)  
        return (input,image)