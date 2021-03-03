import pretty_midi as pretty_midi
import sys
sys.path.append("./")
import src.midi_functions as mf
import os
import sys
import numpy as np
import torch
import pickle
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import argparse
import yaml



now_time = str(int(round(time.time())))
class MidiPreprocessor:
    
    def __init__(self, 
                 classes,   
                 pickle_store_folder, 
                 note_num = 17,
                 longest = 100,
                 low_crop=55,        
                 high_crop=89,       
                 num_notes=128,       
                 smallest_note=16,   
                 max_velocity=127,   
                 input_length=32,    
                 test_fraction=0.01):     
        self.classes = classes
        self.pickle_store_folder = pickle_store_folder
        self.note_num = note_num
        self.longest = longest
        self.low_crop = low_crop
        self.high_crop = high_crop
        self.num_notes = num_notes
        self.smallest_note = smallest_note
        self.input_length = input_length
        self.test_fraction = test_fraction


    def import_midi_from_folder(self, 
                                folder, 
                                save_imported_midi_as_pickle, 
                                save_preprocessed_midi):
        X_list = []
        chroma_list = []
        sources = []
        infors = []
        lengths = []
        folder = "./nice_data/Nottingham/xmas13.mid"
        save_folder =os.path.join(self.pickle_store_folder ,'reconstructed_midi')
        print(folder)
        flag = 1
        source = self.load_rolls(folder, save_folder ,save_preprocessed_midi)
        bar_graph(source)
        return None

    def load_rolls(self, path, save_folder, save_preprocessed_midi):
        infor_list = []
        source_list = []
        length_list = []
        try:
            mid = pretty_midi.PrettyMIDI(path) 
        except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError, AttributeError) as e:
            exception_str = 'Unexpected error in ' + name  + ':\n', e, sys.exc_info()[0]
            print(exception_str)
            return None, None ,None

        tempo_change_times, tempo_change_bpm = mid.get_tempo_changes()
        song_start = 0
        song_end = mid.get_end_time()
        if len(tempo_change_times) == 1:
            tempo = tempo_change_bpm[0]
        else :
            print ("tempo changes")
            return None ,None, None
        quarter_note_length = 1. / (tempo/60.)    
        unit_time = quarter_note_length * 4. / self.smallest_note 
        if len(mid.instruments) ==1 :
            print("this song has no chords")
            return None ,None , None

        notes = mid.instruments[0].notes
        t = 0.
        roll = list()
        flag  = list()
        X = torch.tensor ([])
        num1 = 1
        num2 = 1
        number = 0
        for note in notes: 
            elapsed_time = note.start - t  
            
            if elapsed_time < -0.03: 
                print("the %d notes overlap %f" % (number,-elapsed_time))
                return None ,None , None
            
            if elapsed_time > unit_time:
                steps = np.zeros((int(round(elapsed_time / unit_time)), 130))
                steps[range(int(round(elapsed_time / unit_time))), 129] += 1. 
                roll.append(steps)
                X= np.vstack(roll)

            now_steps = X.shape[0]
            now_time = now_steps * unit_time
            if now_time - note.start >= num1 * unit_time:
                flag.append((now_steps-1,"add"))
                num1 += 1  
                num2 -= 1
      
            if note.start - now_time >= num2 * unit_time:
                flag.append((now_steps-1,"del"))
                num2 += 1
                num1 -= 1

            n_units = int(round((note.end - note.start) / unit_time)) 
            if n_units ==0: 
                n_units =1
            steps = np.zeros((n_units, 130)) 
            steps[0, note.pitch] += 1  
            steps[range(1, n_units), 128] += 1 
            roll.append(steps) 
            t = note.end 
            X= np.vstack(roll) 
            number += 1

        notes = mid.instruments[1].notes

        max_end = 0.
        for note in notes:
            if note.end > max_end:
                max_end = note.end
        chroma = np.zeros((int(round(max_end / unit_time)), 12))
        for note in notes:
            idx = int(round((note.start / unit_time)))
            n_unit = int(round((note.end - note.start) / unit_time))
            chroma[idx:idx + n_unit, note.pitch % 12] += 1


        differ = X.shape[0] - chroma.shape[0]
        if differ != 0 :
            if differ > 0 :
                chroma = np.pad(chroma, ((0,differ),(0, 0)), 'constant', constant_values=(0, 0))
            else :
                chroma = chroma[:differ]

        if save_preprocessed_midi: 
            mf.numpy_to_midi(X, tempo, save_folder ,name, self.smallest_note)

        cut_flag = 0
        lists = X.argmax(1)
        for step in range(X.shape[0]):  
            if lists[step] != 129:
                cut_flag = step
                break
        X = X[cut_flag:,:]
        chroma = chroma[cut_flag:,:]
 
        cut_flag = -1
        lists = X.argmax(1)
        for step in range(1,X.shape[0]):  
            if lists[-step] != 129:
                cut_flag = -step
                break
        if cut_flag != -1:
            print('end delete end delete end delete end delete')
            X = X[:cut_flag,:]
            chroma = chroma[:cut_flag,:]

        silence = X[:,-1]
        #print(silence)
        pos_silence = np.nonzero(silence)[0] 
        if pos_silence.shape[0] > 0:
            print('silence',pos_silence.shape[0])
        X[pos_silence,-1] = 0
        X[pos_silence,-2] = 1

        X= mf.modify_pianoroll_dimentions(X,low =self.low_crop,high = self.high_crop, operate = "del")

        assert(X.shape[0] == chroma.shape[0])
        for step in range(X.shape[0]):
            if np.sum(X[step]) == 0:
                X[step, -2] = 1
        for step in range(X.shape[0]):
            assert(np.sum(X[step,:]) == 1)

        return X

def bar_graph(melody):
    print(melody.shape)
    note = melody[:,:-2]
    note1 = np.nonzero(note)[1]
    pitch = np.bincount(note1,minlength=34)
    print("sum :",np.sum(pitch))

    x = np.linspace(0,33,34)
    y1 = pitch.astype(float)
    y2 = pitch.astype(int)
    for i in range(y1.shape[0]):
        if y1[i] == 0 :
            y1[i] = 0.1

    plt.bar(x=x, height=y1, label='number of pitch', color='steelblue', alpha=0.8)
    #plt.axis('equal')
    for x1, yy in zip(x, y2):
        if yy == 0 : continue
        plt.text(x1-0.4, yy+0.05, str(yy), fontsize=10, rotation=0)
    plt.title("Distribution of pitchs.")
    plt.xlabel('pitch')
    plt.ylabel("number")
    plt.legend()
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='myconf.yml')   
parser.add_argument('--import_dir', default='./nice_data',type=str)
parser.add_argument('--save_imported_midi_as_pickle', type=bool, default=True)  
parser.add_argument('--save_preprocessed_midi', type=bool, default=True)   
args = parser.parse_args()
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