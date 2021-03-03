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


class ConvDecoder(nn.Module):

    def __init__(self,  z_dim = 75):
        super(ConvDecoder, self).__init__()
        self.z_dim = z_dim
        
        self.decoder = nn.Sequential(
			nn.Linear(self.z_dim , 1024),
			nn.ReLU(),
			nn.Linear(1024, 4096),
			nn.ReLU(),
			Unflatten(256, 4, 4),
			nn.ReLU(),
			nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1),  
			nn.ReLU(),
			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
			nn.ConvTranspose2d(32, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
			nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1),  
        
			nn.Sigmoid()      
		)

    def forward(self, x):
        #x = x /10
        flat_x = x.view(x.size(0), -1)
        image = self.decoder(flat_x)
        return image 
