import argparse
import torch
import torch.optim as  optim
from torchvision.utils import save_image
from VAE import *
from peel import peel_1, peel_2
from utils import prepare_MNIST
import torch.nn.functional as F
import os
import time
import math
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cuda = torch.cuda.is_available()
#device = torch.device("cuda" if cuda else "cpu")
device = torch.device("cpu")

parser = argparse.ArgumentParser(description='Convolutional VAE MNIST Example')
parser.add_argument('--save_as_pickle', type=str, default=True, metavar='DIR',    
					help='if save data')
parser.add_argument('--save_dir', type=str, default='./mydata', metavar='DIR',              
					help='saving directory')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',                   
					help='input batch size for training (default: 128)')
parser.add_argument('--num_worker', type=int, default=2, metavar='N',    
					help='num_worker')


args = parser.parse_args()

kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}


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

def modify(num,segment):
    if   num == 3:
        return round(1.25 * segment)
    elif num == 4:
        return round(1.25 * segment)
    elif num == 5:
        return round(1.07 * segment)
    elif num == 2:
        return round(1.16 * segment)
    elif num == 8:
        return round(1.28 * segment)   
    elif num == 6:
        return round(1.16 * segment)   
    elif num == 9:
        return round(1.16 * segment) 
    elif num == 0:
        return round(1.1 * segment)   
    else :
        return segment

def thickness(pos,image):
    num = sample_num
    diff = pos.shape[0] /num
    thicks= np.array([])
    linespace = np.linspace(1,pos.shape[0]-1,num)
    for i in range(num) :
        index = int(linespace[i])
        y = pos[index][0]
        x = pos[index][1]
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
        data [data == 1] = 5
        modify = np.clip(data, a_min=2, a_max=5)
        thick = np.min(modify)
        thicks = np.append(thicks,thick)
    return np.mean(thicks)

def get_start(sums,image):
    if sums ==   0:
        point = np.array([-7.0,13.5])
    elif sums == 1:
        point = np.array([-7.0,15.0])
    elif sums == 2:
        point = np.array([20.0,27.0])
    elif sums == 3:
        point = np.array([20.0,0.0])
    elif sums == 4:
        point = np.array([35.0,16.0])
    elif sums == 5:
        point = np.array([6.0,27.0])
    elif sums == 6:
        point = np.array([-7.0,13.5])
    elif sums == 7:
        point = np.array([37.0,12.0])
    elif sums == 8:
        point = np.array([5.0,27.0])
    elif sums == 9:
        point = np.array([37.0,13.5])
    position = np.argwhere(image > 0).astype(float) 
    y = position[:,0]
    x = position[:,1]
    distance = np.square( y - point[0] ) + np.square( x - point[1] )
    pos = np.argmin(distance)
    return position[pos]



def reprocess():

    trainloader, testloader, classes = prepare_MNIST(args.batch_size, args.num_worker)
    all_segment=[]
    all_sum=[]
    all_data = []
    all_thick=[]
    all_image = []
    all_start = []
    for i, data in enumerate(trainloader):
        inputs = data[0]        
        sums = data[1]
        inputs = inputs.numpy()
        
        for number in range(inputs.shape[0]):
            inputer = inputs[number] 
            sum = sums[number]   
            input = np.squeeze(inputer)  
            input = (input > threshold) * 1
            position = np.argwhere(input > 0) 

            image1 = peel_1(input)
            '''
            if number == 34:
               plt.imshow(image1, cmap=plt.get_cmap('gray_r'))  #画图
               plt.show()
            '''
            image2 = peel_2(image1)

            if image2 is  None :
                print("NONE NONE")
                continue
            '''
            print(number,":",sum)
            plt.figure(dpi = 150)
            plt.subplot(1,2,1)
            plt.imshow(inputer[0], cmap=plt.get_cmap('gray_r'))  #画图
            #plt.imshow(input, cmap=plt.get_cmap('gray_r'))  #画图
            #plt.imshow(image1, cmap=plt.get_cmap('gray_r'))  #画图
            plt.subplot(1,2,2)
            plt.imshow(image2, cmap=plt.get_cmap('gray_r'))  #画图
            plt.show()
            '''

            image2[0:2,0:2] =sum ; image2[0:2,26:] = sum ; image2[26:,0:2] = sum ; image2[26:,26:] = sum
            all_image.append(image2)
            all_data.append(inputer)
            all_sum.append(sum)
        print(i)
    print("all_image :",len(all_image))
    print("all_data :",len(all_data))
    print("all_sum :",len(all_sum))

    if args.save_as_pickle:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        pickle.dump(all_image,open(args.save_dir+'/all_image.pickle', 'wb'))
        pickle.dump(all_data,open(args.save_dir+'/all_data.pickle', 'wb'))
        pickle.dump(all_sum,open(args.save_dir+'/all_sum.pickle', 'wb'))

if __name__ == '__main__':
    sample_num = 20
    resolution = 1.5
    threshold = 0.5
    test_num = 15
    reprocess()

