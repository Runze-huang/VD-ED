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
from dataset_pre import Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cuda = torch.cuda.is_available()
#device = torch.device("cuda" if cuda else "cpu")
device = torch.device("cpu")

parser = argparse.ArgumentParser(description='Convolutional VAE MNIST Example')
parser.add_argument('--result_dir', type=str, default='./pre_result', metavar='DIR',   
					help='output directory')
parser.add_argument('--save_dir', type=str, default='./pre_model', metavar='DIR',             
					help='model saving directory')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',                  
					help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',                        
					help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',   
					help='path to latest checkpoint (default: None')
parser.add_argument('--test_every', default=5, type=int, metavar='N',    
					help='test after every epochs')
parser.add_argument('--num_worker', type=int, default=2, metavar='N',     
					help='num_worker')
parser.add_argument('--data_dir', type=str, default='./number/', metavar='DIR',    
					help='data directory')

# model options
parser.add_argument('--lr', type=float, default=1e-3,
					help='learning rate')                            
parser.add_argument('--line_dim', type=int, default=25, metavar='N',
					help='attribute vector size of encoder')            
parser.add_argument('--input_dim', type=int, default=28 * 28, metavar='N',
					help='input dimension (28*28 for MNIST)')        
parser.add_argument('--input_channel', type=int, default=1, metavar='N',
					help='input channel (1 for MNIST)')             

args = parser.parse_args()
kwargs = {'num_workers': 2, 'pin_memory': True} if cuda else {}


def get_more_sample(image ,x ,y,digit):
    lens = len(image)
    images = []
    xs = []
    ys = []
    for i in range(lens):

        sum = digit[i]
        pos  = np.nonzero(image[i])
        pos = list(pos)

        image[i][0:2,0:2] = sum ; image[i][0:2,26:] = sum ; image[i][26:,0:2] = sum ; image[i][26:,26:] = sum
        images.append(image[i])
        xs.append(x[i])
        ys.append(y[i])
        left ,right ,up, down = 3,3,3,3
        right_up , right_down , left_up , left_down = 2,2,2,2

        x_min = np.min(pos[1])
        x_max = np.max(pos[1])
        y_min = np.min(pos[0])
        y_max = np.max(pos[0])
        if x_min <= 2 :
            left = x_min ; right = 6 - left
        if y_min <= 2 :
            up = y_min ; down = 6 - up
        if x_max >= 25 :
            right = 27 - x_max ; left = 6 - right
        if y_max >= 25 :
            down = 27 - y_max ; up = 6 - down

        for j in range(1,left+1): 
            left_move = [ np.copy(pos[0]) , np.copy(pos[1]) ]
            left_move[1] -= j
            left_move = (left_move[0],left_move[1])
            add_new = np.zeros((28,28))
            add_new [left_move] = 1
            add_new[0:2,0:2] = sum ; add_new[0:2,26:] = sum ; add_new[26:,0:2] = sum ; add_new[26:,26:] = sum

            images.append( add_new )
            xs.append(x[i] - j)
            ys.append(y[i])

        for j in range(1,right+1): 
            right_move = [ np.copy(pos[0]) , np.copy(pos[1]) ]
            right_move[1] += j
            right_move = (right_move[0],right_move[1])
            add_new = np.zeros((28,28))
            add_new [right_move] = 1
            add_new[0:2,0:2] = sum ; add_new[0:2,26:] = sum ; add_new[26:,0:2] = sum ; add_new[26:,26:] = sum

            images.append( add_new )
            xs.append(x[i] + j)
            ys.append(y[i])

        for j in range(1,up+1): 
            up_move = [ np.copy(pos[0]) , np.copy(pos[1]) ]
            up_move[0] -= j
            up_move  = (up_move[0],up_move[1])
            add_new = np.zeros((28,28))
            add_new [up_move] = 1
            add_new[0:2,0:2] = sum ; add_new[0:2,26:] = sum ; add_new[26:,0:2] = sum ; add_new[26:,26:] = sum
            
            images.append( add_new )
            xs.append(x[i])
            ys.append(y[i]-j)

        for j in range(1, down +1): 
            down_move = [ np.copy(pos[0]) , np.copy(pos[1]) ]
            down_move[0] += j
            down_move = (down_move[0],down_move[1])
            add_new = np.zeros((28,28))
            add_new [down_move] = 1
            add_new[0:2,0:2] = sum ; add_new[0:2,26:] = sum ; add_new[26:,0:2] = sum ; add_new[26:,26:] = sum

            images.append( add_new )
            xs.append(x[i])
            ys.append(y[i]+j)

        if min(right,up) <= 1 : 
            right_up = 2 -  min(right,up) ; left_down = 4 - right_up
        if min(right,down) <= 1 : 
            right_down = 2 -  min(right,down) ; left_up = 4 - right_down
        if min(left,down) <= 1 : 
            left_down = 2 -  min(left,down) ; right_up = 4 - left_down
        if min(left,up) <= 1 : 
            left_up = 2 -  min(left,up) ; right_down = 4 - left_up
        
        for j in range(1, right_up +1): 
            move = [ np.copy(pos[0]) , np.copy(pos[1]) ]
            move[0] -= j  
            move[1] += j  
            move = (move[0],move[1])
            add_new = np.zeros((28,28))
            add_new [move] = 1
            add_new[0:2,0:2] = sum ; add_new[0:2,26:] = sum ; add_new[26:,0:2] = sum ; add_new[26:,26:] = sum

            images.append( add_new )
            xs.append(x[i]+j)
            ys.append(y[i]-j)

        for j in range(1, right_down +1): 
            move = [ np.copy(pos[0]) , np.copy(pos[1]) ]
            move[0] += j  
            move[1] += j 
            #if np.max(move[1]) >= 28 : break
            move = (move[0],move[1])
            add_new = np.zeros((28,28))
            #print(move,j,right_down,i)
            #print(right,down)
            add_new [move] = 1
            add_new[0:2,0:2] = sum ; add_new[0:2,26:] = sum ; add_new[26:,0:2] = sum ; add_new[26:,26:] = sum

            images.append( add_new )
            xs.append(x[i]+j)
            ys.append(y[i]+j)

        for j in range(1, left_down +1): 
            move = [ np.copy(pos[0]) , np.copy(pos[1]) ]
            move[0] += j  
            move[1] -= j  
            move = (move[0],move[1])
            add_new = np.zeros((28,28))
            add_new [move] = 1
            add_new[0:2,0:2] = sum ; add_new[0:2,26:] = sum ; add_new[26:,0:2] = sum ; add_new[26:,26:] = sum

            images.append( add_new )
            xs.append(x[i]-j)
            ys.append(y[i]+j)

        for j in range(1, left_up +1): 
            move = [ np.copy(pos[0]) , np.copy(pos[1]) ]
            move[0] -= j  
            move[1] -= j  
            move = (move[0],move[1])
            add_new = np.zeros((28,28))
            add_new [move] = 1
            add_new[0:2,0:2] = sum ; add_new[0:2,26:] = sum ; add_new[26:,0:2] = sum ; add_new[26:,26:] = sum

            images.append( add_new )
            xs.append(x[i]-j)
            ys.append(y[i]-j)
        
    return images, xs , ys 



def load_data(path, batch_size): 
    data_loader = None 
    if path != '':    
        image,x,y,digit = [],[],[],[]
        for i in range(10): 
            for j in range(1,15): 
                if os.path.exists("./number/%d_%d.txt" % (i,j)) is False : continue
                num_data = np.loadtxt("./number/%d_%d.txt" % (i,j))
                num_data[0:2,0:2] =0 ; num_data[0:2,26:] = 0 ; num_data[26:,0:2] = 0 ; num_data[26:,26:] = 0
                image.append( num_data ) 
                x.append ( np.loadtxt("./number/%d_%d_x.txt" % (i,j)) ) 
                y.append ( np.loadtxt("./number/%d_%d_y.txt" % (i,j)) )
                digit.append(i) 
        print("had loaded data")
        #print(len(image),len(x))
        #get_more_sample(image,x,y,digit)
        #plt.imshow(image[1], cmap=plt.get_cmap('gray_r'))
        #plt.show()
        images ,xs, ys = get_more_sample(image,x,y,digit)
        print(len(images),len(xs))
        #plt.imshow(image[0], cmap=plt.get_cmap('gray_r'))
        #print(images[1506][0][0])
        #plt.imshow(images[1506], cmap=plt.get_cmap('gray_r'))
        #plt.show()
        data = Dataset(images ,xs , ys)  
        data_loader = DataLoader(data, batch_size=batch_size, shuffle = True) 
    return data_loader

def txt_save(content,filename):
    # Try to save a list variable in txt file.
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
    #dot_pos = np.argwhere(ones > 0)
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

def loss_func( x_line, y_line, xs , ys):    
    regression_loss = torch.nn.MSELoss(reduce=True, reduction='sum')
    loss = regression_loss (x_line , xs) + regression_loss (y_line , ys)

    return loss

def train():
    model = ConvEncoder(input_channels=args.input_channel, x_dim= line_dim , y_dim = line_dim ).cuda()
                                                                                   
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    if not os.path.exists(args.result_dir):        
        os.makedirs(args.result_dir)
    data_loader = load_data( args.data_dir ,args.batch_size)
    for epoch in range(start_epoch, args.epochs):    
        for i , data in enumerate(data_loader):  
            inputs = data[0]
            xs = data[1]
            ys = data[2]

            inputs = torch.unsqueeze(inputs,1)

            model.cuda()
            x_line,y_line = model(inputs)  
            loss  = loss_func(x_line, y_line, xs, ys) 
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
			
            if (epoch + 1) % 1 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss {:.2f}"  .format(epoch + 1, args.epochs, i + 1, len(data_loader), loss.item() )) 

            if epoch +1 >= 40 and i == 15 :
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
                        #print(xx, yy)
                        images [ i, yy , xx ] = 1
                x_concat = torch.cat([inputs.view(128, 1, 28, 28), images.view(128, 1, 28, 28)], dim=3) 
                save_image(x_concat, ("%s/reconstructed-%d.png" % (result_dir, epoch + 1))) 
                txt_save(x_line,("%s/x_line-%d.txt" % (result_dir, epoch + 1)))
                txt_save(y_line,("%s/y_line-%d.txt" % (result_dir, epoch + 1)))

                inputser = np.array(inputs.cpu().detach()) 
                np.save(("%s/images-%d" % (result_dir, epoch + 1)),inputser)

        if (epoch + 1) % 5 == 0 :
            epoch_num  = 'epoch_{}'.format(epoch +1)
            if os.path.exists(model_path):
               shutil.rmtree(model_path)
            os.makedirs(model_path)
            torch.save(model.cpu().state_dict(), os.path.join(model_path,"%s.pt" % (epoch_num)))
            print('Model saved!')
if __name__ == '__main__':

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