import argparse
import os
import pickle
import sys
import yaml
import time
import torch
from model import VAE

sys.path.append(".")

from src.separator import MidiSeparator

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='myconf.yml') 
def main(args):
    conf = None 
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file,Loader=yaml.FullLoader)
        conf = config['separate']
        model_params = config['model']
        preprocess_params = config['preprocessor']
    date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    conf['save_path'] = os.path.join(conf['save_path'],date_time)

    if not os.path.isdir(conf['save_path']):
        os.mkdir(conf['save_path'])

    separator = MidiSeparator(conf['songs_path'],conf['save_path'],conf['save_reconstructed'],
                              model_params['roll_dim'],model_params['time_step'],
                              preprocess_params['low_crop'],preprocess_params['high_crop'],preprocess_params['note_num'],preprocess_params['longest'])
    model = VAE(model_params['roll_dim'], model_params['hidden_dim'], model_params['infor_dim'], 
                                                 model_params['time_step'],12)

    model.eval()
    separator.import_midi_from_folder(model)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
