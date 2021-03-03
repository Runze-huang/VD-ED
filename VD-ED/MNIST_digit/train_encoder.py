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
from encoder import *
import shutil
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
import time
from dataset_encoder import Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cuda = torch.cuda.is_available()
device = torch.device("cpu")

parser = argparse.ArgumentParser(description='Convolutional VAE MNIST Example')
parser.add_argument('--result_dir', type=str, default='./result', metavar='DIR',    
					help='output directory')
parser.add_argument('--save_dir', type=str, default='./model', metavar='DIR',         
					help='model saving directory')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',                
					help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',                     
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
        data_loader = DataLoader(data, batch_size=batch_size, shuffle = True)
    return data_loader

def txt_save(content,filename):
    z = np.array(content.cpu().detach())
    np.savetxt(filename,z)

def sum_save(content,filename):
    sums = np.array(content.cpu().detach())
    np.savetxt(filename,sums)

def save_checkpoint(state, is_best, outdir):        
    if not os.path.exists(outdir):     
        os.makedirs(outdir)

    checkpoint_file = os.path.join(outdir, 'checkpoint.pth')   
    best_file = os.path.join(outdir, 'model_best.pth')
    torch.save(state, checkpoint_file)        
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)   

def build_image(x,y,thick):  
    x_max = math.ceil(max(x)) +2
    if x_max >27: x_max =27
    y_max = math.ceil(max(y)) +2
    if y_max >27: y_max =27
    x_min = math.floor(min(x)) -2
    if x_min <0: x_min =0
    y_min = math.floor(min(y)) -2
    if y_min <0: y_min = 0
    
    ones = torch.ones(( y_max - y_min + 1 , x_max - x_min + 1)).cuda().requires_grad_()
    dot_pos = ( ones > 0).nonzero().float().requires_grad_() 
    dot_pos = dot_pos + torch.tensor([y_min, x_min]).cuda()
    numbers = dot_pos.shape[0]
    x_matrix = x.repeat(numbers,1)
    y_matrix = x.repeat(numbers,1)
    x_dot = torch.unsqueeze(dot_pos[:,1],1)
    y_dot = torch.unsqueeze(dot_pos[:,0],1)
    x_square = (x_matrix - x_dot).pow(2)
    y_square = (y_matrix - y_dot).pow(2)
    sum_squares = x_square + y_square
    sum_squares = ( (sum_squares < thick ) * 1 ).float().requires_grad_()
    dots = sum_squares.sum(1)
    dots = ((dots > 0) * 1).float().requires_grad_()
    dots_matrix = dots.reshape((y_max - y_min + 1 , x_max - x_min + 1))
    modify_matrix = F.pad(dots_matrix, pad=(x_min, 27-x_max , y_min, 27-y_max ), mode="constant",value=0.0)
    return modify_matrix


def getcoords(point ,line):
    x_lists = []
    y_lists = []
    for i in range(len(line)):  
        x = torch.tensor( [ point[i][1] ] ).cuda()
        y = torch.tensor( [ point[i][0] ] ).cuda()
        x_list = 1.5 * torch.cos(-line[i])
        y_list = 1.5 * torch.sin(-line[i])
        x_list = torch.cat((x ,x_list))
        y_list = torch.cat((y ,y_list))
        x_list =x_list.cumsum(0)
        y_list =y_list.cumsum(0)
        x_lists.append(x_list)
        y_lists.append(y_list)
    return x_lists, y_lists

def distance(x_line,y_line,x_dot,y_dot):
    numbers = x_dot.shape[0]
    x_matrix = x_line.repeat(numbers,1)
    y_matrix = y_line.repeat(numbers,1)
    x_square = (x_matrix - x_dot).pow(2)
    y_square = (y_matrix - y_dot).pow(2)
    sum_squares = (x_square + y_square).sqrt()

    column_min = torch.min(sum_squares,0)[0]

    min_list = torch.min(sum_squares,1)[0]

    diff1 = column_min - 0.7  #
    diff1[diff1 < 0] = 0
    diff1 *= 3
    column_min = column_min + diff1

    diff = min_list - 0.7
    diff[diff < 0] = 0
    diff *= 3
    min_list = min_list + diff

    sums2 = torch.sum(column_min)
    sums1 = torch.sum(min_list)   

def loss_func(inputs, x_line, y_line, sums):     
    x_1 = x_line[:,:-1]
    x_2 = x_line[:, 1:]
    y_1 = y_line[:,:-1]
    y_2 = y_line[:, 1:]
    distances =  ( (x_1 - x_2).pow(2) + (y_1 - y_2).pow(2) ).sqrt()
    distances = distances - 0  
    max_dis = torch.max(distances[:,1:],1)[1] +1
    inputs = torch.squeeze(inputs,dim = 1)
    batch = inputs.shape[0]
    all_sum1 = torch.tensor([0.0]).cuda()
    all_sum2 = torch.tensor([0.0]).cuda()

    for i in range(batch): 
        if sums[i] ==4 or sums[i] == 7 :
           dis_mean = ( torch.sum(distances[i]) - distances[i,max_dis[i]] ) / 23
           distances[i,max_dis[i]] = dis_mean

        dot_pos = (inputs[i] > 0).nonzero()
        x_dot = torch.unsqueeze(dot_pos[:,1],1)
        y_dot = torch.unsqueeze(dot_pos[:,0],1)
        sums1,sums2 = distance(x_line[i], y_line[i], x_dot, y_dot) 
        all_sum1 += sums1
        all_sum2 += sums2
    dis_std = torch.std(distances , 1)
    std_loss = torch.mean (dis_std)
    diff = distances - 2.8 
    diff[diff < 0] = 0
    diff *= 10
    distances = distances + diff

    mean_dis = torch.mean(distances)

    distance1 = all_sum1 / batch
    distance2 = all_sum2 / batch

    loss = distance1 + distance2  + (dis_para * mean_dis) + (std_para * std_loss)
    #print(loss)
    return loss , distance1, distance2 , mean_dis , std_loss

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
                             args.batch_size)

    for epoch in range(start_epoch, args.epochs):    
        for i, data in enumerate(data_loader):  
            inputs = data[0]
            inputer = inputs.clone()
            inputer[:,0:2,0:2] =0 ; inputer[:,0:2,26:] = 0 ; inputer[:,26:,0:2] = 0 ; inputer[:,26:,26:] = 0
 
            sources = data[1]
            sums = data[2]
            inputs = inputs.unsqueeze(1)
            model.cuda()
            x_line,y_line = model(inputs)  
            loss , distance1_loss , distance2_loss , dis_loss, std_loss = loss_func(inputer, x_line, y_line, sums) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 1 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss {:.2f}, distance1 {:.2f}, distance2 {:.2f}, mean_dis {:.2f}, std_loss {:.2f}"  .format(epoch + 1, args.epochs, i + 1, len(data_loader), loss.item(),distance1_loss.item(),distance2_loss.item(),dis_loss.item(),std_loss.item() )) 
                        if (epoch + 1 ) % 1 == 0 and i == 440 :
                images = torch.zeros(x_line.shape[0],28,28).cuda()
                x_line = torch.round(x_line)
                y_lien = torch.round(y_line)

                for i in range(x_line.shape[0]):
                    for j in range(line_dim):
                        xx = int(x_line[i][j])
                        if xx >27 : xx = 27
                        if xx <0 :  xx = 0
                        yy = int(y_line[i][j])
                        if yy >27 : yy = 27
                        if yy <0 :  yy = 0
                        images [ i, yy , xx ] = 1

                x_concat = torch.cat([inputs.view(128, 1, 28, 28), images.view(128, 1, 28, 28)], dim=3) 
                save_image(x_concat, ("%s/reconstructed-%d.png" % (result_dir, epoch + 1))) 
                txt_save(x_line,("%s/x_line-%d.txt" % (result_dir, epoch + 1)))
                txt_save(y_line,("%s/y_line-%d.txt" % (result_dir, epoch + 1)))

                inputser = np.array(inputs.cpu().detach()) 
                sources = np.array(sources.cpu())
                np.save(("%s/sources-%d" % (result_dir, epoch + 1)),sources)
                np.save(("%s/images-%d" % (result_dir, epoch + 1)),inputser)

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