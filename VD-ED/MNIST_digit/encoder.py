import torch
import torch.nn as nn


class Flatten(nn.Module):       
    def forward(self, input):
        return input.view(input.size(0), -1)   

class Unflatten(nn.Module):
    def __init__(self, channel, height, width):     
        super(Unflatten, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.height, self.width)


class ConvEncoder(nn.Module):

    def __init__(self, input_channels=1, x_dim=25, y_dim=25):
        super(ConvEncoder, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, out_channels=8, kernel_size=3, stride=1, padding=1),   
			nn.ReLU(),  
            nn.Conv2d(8, 32 , kernel_size=4, stride=2, padding=1), 
			nn.ReLU(),  
            nn.Conv2d(32, 64 , kernel_size=4, stride=2, padding=1),   
			nn.ReLU(),  
			nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),   
			nn.ReLU(), 
			Flatten(),            
			nn.Linear(4096, 1024), 
			nn.ReLU()
		)

        self.initial_point1 = nn.Linear(1024,256)
        self.initial_point2 = nn.Linear(256,self.x_dim)
        self.segment1 = nn.Linear(1024,256)
        self.segment2 = nn.Linear(256,self.y_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.encoder(x)
        point1 = self.initial_point1(h)  
        point2 = self.relu(point1)
        point3 = self.initial_point2(point2)

        segment1 = self.segment1(h)
        segment2 = self.relu(segment1)
        segment3 = self.segment2(segment2)
        return point3*10 , segment3*10
