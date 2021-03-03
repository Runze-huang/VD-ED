import argparse
import torch
import torch.optim as  optim
from torchvision.utils import save_image
from utils import prepare_MNIST
import torch.nn.functional as F
import os
import time
import math
import random
import pickle
from encoder import *
import shutil
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
import time
from dataset import Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cuda = torch.cuda.is_available()
device = torch.device("cpu")

parser = argparse.ArgumentParser(description='Convolutional VAE MNIST Example')
parser.add_argument('--result_dir', type=str, default='./result', metavar='DIR',   
					help='output directory')
parser.add_argument('--save_dir', type=str, default='./mydata', metavar='DIR',             
					help='model saving directory')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',                  
					help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',                      
					help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--model', default='./model/ooo/epoch_19.pt', type=str, metavar='PATH',   
					help='path to latest checkpoint (default: None')
parser.add_argument('--test_every', default=5, type=int, metavar='N',   
					help='test after every epochs')
parser.add_argument('--num_worker', type=int, default=2, metavar='N',    
					help='num_worker')
parser.add_argument('--data_dir', type=str, default='./mydata/', metavar='DIR',   
					help='data directory')

# model options
parser.add_argument('--lr', type=float, default=1e-4,
					help='learning rate')                           
parser.add_argument('--line_dim', type=int, default=25, metavar='N',
					help='attribute vector size of encoder')           
parser.add_argument('--input_dim', type=int, default=28 * 28, metavar='N',
					help='input dimension (28*28 for MNIST)')       
parser.add_argument('--input_channel', type=int, default=1, metavar='N',
					help='input channel (1 for MNIST)')            

args = parser.parse_args()
kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}
def load_data(image_data,start_data,number_data,lebal_data,thick_data,segment_data, batch_size):
    data_loader = None
    if number_data != '':    
        image = pickle.load(open(image_data,'rb'))
        number = pickle.load(open(number_data, 'rb')) 
        lebal = pickle.load(open(lebal_data, 'rb'))
        data = Dataset(image ,number,lebal)  
        data_loader = DataLoader(data, batch_size=batch_size, shuffle = False)
    return data_loader


def filter(images,positions,sums):
    rate = 0.0
    lists = [0,0,0,0,0,0,0,0,0,0]
    sources, lines = [], []
    for i in range(len(images)):
        threshold = 2
        position = np.round( positions[i] ).astype(int)
        pos = (position[0],position[1])
        image = images[i][0].cpu().numpy()
        value = image[pos]
        valid = ( value < 0.25) * 1 
        invalid = np.sum(valid)
        index = sums[i].int()
        if index == 8 or index == 9 : threshold = 3
        if index == 1 : threshold = 1
        if invalid >= threshold:
            rate += 1
            lists[index] += 1
            continue
        sources.append(image)
        lines.append(positions[i])
    
    rate = rate / len(images)
    print("rate is :",rate)

    print("got filter :",len(sources))
    
    return sources, lines
    
def get_more_sample(sources,lines):
    lens = len(sources)
    images = []
    new_lines = []
    test = [0,0,0,0]
    for i in range(lens):
        image = sources[i]

        pos  = np.nonzero(image)
        pos = list(pos)
    
        images.append(sources[i])
        new_lines.append(lines[i])

        right_up , right_down , left_up , left_down = 0,0,0,0

        x_min = np.min(pos[1])
        x_max = 27 - np.max(pos[1])
        y_min = np.min(pos[0])
        y_max = 27 - np.max(pos[0])

        right_up = min( y_min, x_max )
        right_down = min( x_max , y_max )
        left_down = min ( x_min , y_max )
        left_up =  min ( x_min , y_min )
    
        ll = random.sample(range(0,4),2)
        for index in ll:
            if index == 0 and right_up >=2 :
                test[0]+=1
                add_new = image[ 2: , :26 ]
                add_new = np.pad(add_new,((0,2),(2,0)),'constant', constant_values=(0,0)) 
                line = lines[i].copy()
                line[1]+=2
                line[0]-=2

                images.append( add_new )
                new_lines.append( line )
                continue

            if index == 1 and right_down >=2:
                test[1]+=1
                add_new = image[ :26 , :26 ]
                add_new = np.pad(add_new,((2,0),(2,0)),'constant', constant_values=(0,0)) 
                line = lines[i].copy()
                line[1]+=2
                line[0]+=2

                images.append( add_new )
                new_lines.append( line )
                continue

            if index == 2 and left_down >=2:
                test[2]+=1
                add_new = image[ :26 , 2: ]
                add_new = np.pad(add_new,((2,0),(0,2)),'constant', constant_values=(0,0)) 
                line = lines[i].copy()
                line[1]-=2
                line[0]+=2

                images.append( add_new )
                new_lines.append( line )
                continue

            if index == 3 and left_up >=2:
                test[3]+=1
                add_new = image[ 2: , 2: ]
                add_new = np.pad(add_new,((0,2),(0,2)),'constant', constant_values=(0,0)) 
                line = lines[i].copy()
                line[1]-=2
                line[0]-=2

                images.append( add_new )
                new_lines.append( line )

    print(test)
    return images, new_lines


def get_thickness( images , positions):
    lists = []
    for j in range(len(images)): 
        position = np.round( positions[j] ).astype(int) 
        image = images[j]
        thicks= np.array([])

        for i in range(len(position[0])) :
            y = position[0,i]
            x = position[1,i]
            slash_pos =min([x,y])
            slash_pos2 =min ([27-x,y])
            left, right,up,down,upleft,downleft,upright,downright = 0,0,0,0,0,0,0,0

            right = image[y , x:].argmin(axis = 0)
            left = image[y , x::-1].argmin(axis = 0)
            horizontal = right + left -1

            up = image  [y::-1 , x].argmin(axis = 0)
            down = image[y:, x].argmin(axis = 0)
            vertical = up + down -1
 
            upleft = image.diagonal(x-y)[slash_pos::-1].argmin(axis = 0)
            downright = image.diagonal(x-y)[slash_pos:].argmin(axis = 0)
            diagonal = upleft + downright
    
            upright = np.fliplr(image).diagonal(27-x-y)[slash_pos2::-1].argmin(axis = 0)
            downleft = np.fliplr(image).diagonal(27-x-y)[slash_pos2:].argmin(axis = 0)
            back_diagonal = upright + downleft
            data =np.array([horizontal,vertical,diagonal,back_diagonal])
            thick = np.min(data)
            thicks = np.append(thicks,thick)
        pos_thick = np.vstack((positions[j],thicks)) 
        lists.append(pos_thick)

    return lists


def train():
    model = ConvEncoder(input_channels=args.input_channel, x_dim= line_dim , y_dim = line_dim ).cuda()
    model.load_state_dict(torch.load(args.model))
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    if not os.path.exists(args.result_dir):         
        os.makedirs(args.result_dir)

    data_loader = load_data( args.data_dir + 'all_image.pickle', 
                             args.data_dir + 'all_start.pickle',
                             args.data_dir + 'all_data.pickle', 
                             args.data_dir +'all_sum.pickle',
                             args.data_dir +'all_thick.pickle',
                             args.data_dir +'all_segment.pickle',
                             args.batch_size )

    positions, sources  ,digit= [],[],[]
    with torch.no_grad():
        for i, data in enumerate(data_loader):  
            inputs = data[0]
            inputer = inputs.clone()
            inputer[:,0:2,0:2] =0 ; inputer[:,0:2,26:] = 0 ; inputer[:,26:,0:2] = 0 ; inputer[:,26:,26:] = 0

            images = data[1]
            sums = data[2]
            inputs = inputs.unsqueeze(1)

            model.cuda()
            x_line,y_line = model(inputs)  
            x_line = x_line.cpu().detach().numpy()
            y_line = y_line.cpu().detach().numpy()

            for j in range(len(images)):
                x = x_line[j]
                y = y_line[j]
                pos = np.stack((y, x))
                positions.append(pos)
                sources.append(images[j])
                digit.append(sums[j])

        print('len of positions is :',len(positions))
        print('len of images is :',len(sources))

        sources, lines = filter(sources,positions,digit)
        pos_thick = get_thickness( sources , lines)
        print('got thickness')
        new_images , new_lines =  get_more_sample(sources,pos_thick)
        print('added sample:',len(new_images))

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        pickle.dump(new_images,open(args.save_dir+'/processed_sources.pickle', 'wb'))
        pickle.dump(new_lines,open(args.save_dir+'/processed_inputs.pickle', 'wb'))
        print("data saved")
        

if __name__ == '__main__':
    sample_num = 20 
    resolution = 1.5
    test_num = 15

    dis_para = 10
    std_para = 100
    line_dim = args.line_dim
    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('CPU mode')
    train()