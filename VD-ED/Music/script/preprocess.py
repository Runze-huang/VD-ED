import argparse
import os
import pickle
import sys
import yaml
import time

sys.path.append(".")

from src.preprocesser import MidiPreprocessor

# General settings
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='myconf.yml') 
parser.add_argument('--import_dir', default='./nice_data',type=str)
parser.add_argument('--save_imported_midi_as_pickle', type=bool, default=True)
parser.add_argument('--save_preprocessed_midi', type=bool, default=True) 
def main(args):
    conf = None 
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file,Loader=yaml.FullLoader)
        conf = config['preprocessor']
    date_time = time.strftime('%Y_%m_%d_%H_%M', time.localtime())
    conf['pickle_store_folder'] = os.path.join(conf['pickle_store_folder'],date_time)
    print(conf['pickle_store_folder'])

    processor = MidiPreprocessor(**conf)
    processor.import_midi_from_folder(args.import_dir,
                                     args.save_imported_midi_as_pickle,
                                     args.save_preprocessed_midi)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)