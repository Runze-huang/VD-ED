import os

import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(Dataset):   

    def __init__(self, X, source ,length):
        self.X = X
        self.source = source
        self.length = length

    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        Xs = torch.tensor(self.X[idx],dtype=torch.float).to(device)
        sources = torch.tensor(self.source[idx],dtype=torch.float).to(device)
        lengths = torch.tensor(self.length[idx] , dtype=torch.float).to(device)
        return (Xs,sources,lengths)