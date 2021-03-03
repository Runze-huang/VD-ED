import argparse
import os
import pickle
import sys
import math
import yaml
import time
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
sys.path.append("./")
import src.midi_functions as mf
from model import VAE

sys.path.append(".")

from src.separator import MidiSeparator

# General settings
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='myconf.yml') 
def main(args):
    conf = None 
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file,Loader=yaml.FullLoader)
        conf = config['combine']
        model_params = config['model']
        preprocess_params = config['preprocessor']
    date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    path = os.path.join(conf['save_path'],date_time)
    path = conf['save_path']

    model = VAE(model_params['roll_dim'], model_params['hidden_dim'], model_params['infor_dim'],  
                 model_params['time_step'],12)

    model.load_state_dict(torch.load(conf['model_path']))
    if torch.cuda.is_available():
        print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
        model.cuda()
    else:
        print('CPU mode')
    model.eval()
    pitch_path = conf['p_path'] + ".txt"
    rhythm_path = conf['r_path'] + ".txt"
    #chord_path = conf['chord_path'] + ".txt"
    name1 = pitch_path.split("/")[-3]
    name2 = rhythm_path.split("/")[-3]
    name = name1+"+"+name2+".mid"
    name2 = name1+"+"+name2+".txt"

    pitch = np.loadtxt(pitch_path)
    print(pitch)
    rhythm = np.loadtxt(rhythm_path)
    print(rhythm)

    print("Importing " + name1+" pitch and "+name2+" rhythm")

    #line_graph(pitch,rhythm)  
    #bar_graph(pitch,rhythm)

    pitch = torch.from_numpy(pitch).float()
    rhythm = torch.from_numpy(rhythm).float()
    recon = model.decoder(pitch, rhythm)

    recon = torch.squeeze(recon, 0)
    recon = mf._sampling(recon)
    recon = np.array(recon.cpu().detach().numpy())
    length = torch.sum(rhythm).int() 
    recon = recon[:length]
    #打印生成的音符分布
    note = recon[:,:-2]
    note = np.nonzero(note)[1]
    note = np.bincount(note,minlength=34).astype(float)    
    recon = mf.modify_pianoroll_dimentions(recon,preprocess_params['low_crop'],preprocess_params['high_crop'],"add")  

    #bar_graph(pitch,rhythm)  
    mf.numpy_to_midi(recon, 120, path ,name, preprocess_params['smallest_note'])

    #pitch_rhythm(recon,path,name2) # write pitch information 

    print("combine succeed")

def txt_save(content1,content2,filename,mode='a'):
    # Try to save a list variable in txt file.
    file = open(filename,mode)
    for i in range(len(content1)):
        file.write(str(i+1)+"  "+str(content1[i])+"  "+str(content2[i])+'\n')
    file.close()

def pitch_rhythm(recon,save_folder,filename):
    path = os.path.join(save_folder,filename)
    note = recon[:,:-2]
    pitch = np.nonzero(note)[1]

    note2 = np.nonzero(note)[0]
    note2 = np.append( note2, recon.shape[0] )
    rhythm = note2[1:] - note2[:-1]
    #infor=infor.astype(np.uint8)  
    txt_save(pitch,rhythm,path)

   
def change_pitch(pitch,change,flag):
    if flag == 'add':
        final = (pitch[::-1]!=0).argmax(axis=0)
        assert(final >= change)
        zero = np.zeros(34)
        zero[change:] = pitch [:-change]
        pitch = zero
    if flag == 'del':
        first = (pitch!=0).argmax(axis=0)
        assert(first >= change)
        zero = np.zeros(34)
        zero[:-change] = pitch[change:]
        pitch = zero
    return pitch 

def coord(rhythm):

    print(rhythm)
    longs = np.ones(17)#rhythm
    rhythm *= 10
    pos = np.zeros((2,len(rhythm)+1))
    for i in range(len(rhythm)):
        angle = rhythm [i] * math.pi / 180
        pos[1,i+1] = pos [1,i] + longs[i] * math.cos(angle)
        pos[0,i+1] = pos [0,i] + longs[i] * math.sin(angle)
    return pos

def bar_graph(pitch, rhythm):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.figure(1)
    x = np.linspace(0,33,34)
    y1 = pitch.astype(float)
    y2 = pitch.astype(int)
    for i in range(y1.shape[0]):
        if y1[i] == 0 :
            y1[i] = 0.05
    #plt.bar(x=x, height=y1, label='number of pitch', color='steelblue', alpha=0.8)
    plt.bar(x=x, height=y1, label='音高数量', color='steelblue', alpha=0.8)

    #plt.axis('equal')
    for x1, yy in zip(x, y2):
        if yy == 0 : continue
        plt.text(x1-0.4, yy+0.05, str(yy), fontsize=10, rotation=0)
    plt.ylim(0, 5)
    plt.xlim(-0.5, 33.5)

    plt.xlabel('音高')
    plt.ylabel("数量")
    plt.legend()
    plt.savefig("./songs/combine_2/pitch.png",dpi = 300,bbox_inches='tight')
    plt.show()

def line_graph(pitch,rhythm):
    pos = coord(rhythm)
    plt.plot(
             pos[1],pos[0],alpha = 0.5,linestyle = '-',linewidth = 4,
             color = 'b',marker = 'o',markeredgecolor = 'g',markeredgewidth = 4.0,
             markerfacecolor = 'w',markersize = 3
             )
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./songs/combine_2/rhythm_4.png",dpi = 300,bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
