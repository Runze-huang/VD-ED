import argparse
import torch
import torch.optim as  optim
from torchvision.utils import save_image
from utils import prepare_MNIST
import torch.nn.functional as F
import os
import time
import math
import pickle
from decoder import *
import shutil
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
import time
from dataset_2 import Dataset
#import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cuda = torch.cuda.is_available()
#device = torch.device("cuda" if cuda else "cpu")
device = torch.device("cpu")

parser = argparse.ArgumentParser(description='Convolutional VAE MNIST Example')
parser.add_argument('--result_dir', type=str, default='./decoder_result', metavar='DIR',    
					help='output directory')
parser.add_argument('--save_dir', type=str, default='./decoder_model', metavar='DIR',              
					help='model saving directory')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',                
					help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',                     
					help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--model', default='./model/mmm/epoch_19.pt', type=str, metavar='PATH',   
					help='path to latest checkpoint (default: None')
parser.add_argument('--test_every', default=5, type=int, metavar='N',    
					help='test after every epochs')
parser.add_argument('--num_worker', type=int, default=2, metavar='N',     
					help='num_worker')
parser.add_argument('--data_dir', type=str, default='./mydata/', metavar='DIR',    
					help='data directory')

parser.add_argument('--lr', type=float, default=1e-3,
					help='learning rate')                            
parser.add_argument('--line_dim', type=int, default=25, metavar='N',
					help='attribute vector size of encoder')            
parser.add_argument('--input_dim', type=int, default=3 * 25 , metavar='N',
					help='input dimension (28*28 for MNIST)')        

args = parser.parse_args()
kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}

def load_data(input_data, image_data, batch_size):
    data_loader = None
    if input_data != '':    
        inputs = pickle.load(open(input_data,'rb'))
        images = pickle.load(open(image_data, 'rb')) 

        data = Dataset(inputs, images)  

        data_loader = DataLoader(data, batch_size=batch_size, shuffle = True)
    return data_loader


def save_checkpoint(state, is_best, outdir):         
    if not os.path.exists(outdir):    
        os.makedirs(outdir)

    checkpoint_file = os.path.join(outdir, 'checkpoint.pth')   
    best_file = os.path.join(outdir, 'model_best.pth')
    torch.save(state, checkpoint_file)         
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)   


def loss_func(outputs, sources):   

    loss = F.binary_cross_entropy(outputs, sources, reduction='sum')
    return loss 

def train():
    model = ConvDecoder( z_dim= 75 ).cuda()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    
    if not os.path.exists(args.result_dir):        
        os.makedirs(args.result_dir)

    data_loader = load_data( args.data_dir + 'processed_inputs.pickle', 
                             args.data_dir + 'processed_sources.pickle',

                             args.batch_size)

    for epoch in range(start_epoch, args.epochs):     
        for i, data in enumerate(data_loader):   
            inputs = data[0]
            sources = data[1]
            model.cuda()
            outputs = model(inputs)

            loss  = loss_func(outputs, sources) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss {:.2f}"  .format(epoch + 1, args.epochs, i + 1, len(data_loader), loss.item()   )  ) 
            if (epoch + 1 ) % 1 == 0 and i == 1220 :

                x_concat = torch.cat([sources.view(128, 1, 28, 28), outputs.view(128, 1, 28, 28)], dim=3) 
                save_image(x_concat, ("%s/reconstructed-%d.png" % (result_dir, epoch + 1))) 

        if (epoch + 1 ) % 5 == 0 :
            epoch_num  = 'epoch_{}'.format(epoch)
            if os.path.exists(model_path):
               shutil.rmtree(model_path)
            os.makedirs(model_path)
            torch.save(model.cpu().state_dict(), os.path.join(model_path,"%s.pt" % (epoch_num)))
            print('Model saved!')

if __name__ == '__main__':
    sample_num = 20 
    resolution = 1.5
    threshold = 0.13
    test_num = 15

    dis_para = 10
    std_para = 100

    date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    result_dir = os.path.join(args.result_dir,date_time)
    if not os.path.exists(result_dir):     
        os.makedirs(result_dir)
    model_path = os.path.join(args.save_dir,date_time)
    if not os.path.exists(model_path):      
        os.makedirs(model_path)
    line_dim = args.line_dim
    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('CPU mode')
    train()