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
from sklearn.model_selection import train_test_split
import time

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
        save_folder =os.path.join(self.pickle_store_folder ,'reconstructed_midi')
        print(folder)
        for path, subdirs, files in os.walk(folder):    
            for name in files:   
                _path = path.replace('\\', '/') + '/'   
                _name = name.replace('\\', '/')
                if _name.endswith('.mid') or _name.endswith('.midi'):
                    shortpath = _path[len(folder):]    
                    found = False
                    for i, c in enumerate(self.classes):
                        if c.lower() in shortpath.lower():   
                            found = True
                            print("Importing " + c + " song called " + _name)  # importing jazz song called _name.mid
                            infor , source ,length = self.load_rolls(_path, _name, save_folder ,save_preprocessed_midi)
                            if infor is not None :
                                infors = infors + infor
                                sources = sources + source
                                lengths = lengths + length 
        '''
        data = train_test_split(infors,
                                sources,
                                lengths, 
                                test_size=self.test_fraction, 
                                random_state=42)
        '''

        X_train = infors
        source_train = sources
        length_train = lengths  

        train_set_size = len(X_train)
        print('totall length is :',train_set_size)
        print(len(source_train))
        print(len(length_train))

        if save_imported_midi_as_pickle:
            if not os.path.exists(self.pickle_store_folder):
                os.makedirs(self.pickle_store_folder)
                
            pickle.dump(X_train,open(self.pickle_store_folder+'/X_train.pickle', 'wb'))
            pickle.dump(source_train,open(self.pickle_store_folder+'/source_train.pickle', 'wb'))
            pickle.dump(length_train,open(self.pickle_store_folder+'/length_train.pickle', 'wb'))

        return None

    def infor_source_length(self,X):
        infor_list , source_list , long_list = [],[],[]
        note = X[:,:-2]
        note = np.nonzero(note)[0]
        amount = note.shape[0]
        note = np.append(note,X.shape[0]) 
        segments = math.floor( amount / self.note_num )
        remain = amount % self.note_num
        token = 0
        for i in range(1 , segments + 1):
            pos = note[17*i]
            part = X[token:pos] 
            token = pos
            pitch,rhythm = self.pitch_rhythm(part)
            infor = np.append(pitch,rhythm)
            infor_list.append(infor)
            if part.shape[0] > self.longest:
                part = part[:self.longest]

            longs = part.shape[0]
            if longs != self.longest :
                pads = self.longest - longs
                part = np.pad(part, ((0,pads),(0, 0)), 'constant', constant_values=(0, 0))
                part[-pads:,-1] = 1
            source_list.append(part)
            long_list.append(longs)

        if remain > self.note_num / 2 : 
            part = X[token:]

            pitch,rhythm = self.pitch_rhythm(part)
            infor = np.append(pitch,rhythm) 
            infor_list.append(infor)
            if part.shape[0] > self.longest:
                part = part[:self.longest]

            longs = part.shape[0]
            if longs != self.longest:
                pads = self.longest - longs
                part = np.pad(part, ((0,pads),(0, 0)), 'constant', constant_values=(0, 0)) 
                part[-pads:,-1] = 1
            source_list.append(part)
            long_list.append(longs)

        return infor_list , source_list , long_list

    def load_rolls(self, path, name, save_folder, save_preprocessed_midi):
        infor_list = []
        source_list = []
        length_list = []
        try:
            mid = pretty_midi.PrettyMIDI(path + name)  
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

        if len(flag) > 0 :
            chroma = self.align(flag ,chroma)


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

        note = X[:,:-2]
        note = np.nonzero(note)[0]
        X1 = X [note[1] : ]
        X2 = X [note[2] : ]
        X3 = X [note[3] : ]
        infor0 , source0 , length0 = self.infor_source_length(X)
        infor1 , source1 , length1 = self.infor_source_length(X1)
        infor2 , source2 , length2 = self.infor_source_length(X2)
        infor3 , source3 , length3 = self.infor_source_length(X3)

        infor_list = infor0 + infor1 + infor2 + infor3
        source_list = source0 + source1 + source2 + source3
        length_list = length0 + length1 + length2 + length3
        return infor_list , source_list ,length_list

    def align(self ,flag, chroma):
        for (pos, operate) in flag:
            if operate == "add":
                if pos < chroma.shape[0]:
                  temp = chroma [pos-1]
                  chroma = np.insert(chroma, pos, values= temp, axis=0)
            if operate == "del":
                if pos < chroma.shape[0]:
                  chroma = np.delete(chroma, pos-1, axis = 0)
        return chroma

    def pitch_rhythm(self , part):

        note = part[:,:-2]
        note1 = np.nonzero(note)[1]
        pitch = np.bincount(note1,minlength=34)
        note2 = np.nonzero(note)[0]
        note2 = np.append( note2, part.shape[0] )
        rhythm = note2[1:] - note2[:-1]
        if rhythm.shape[0] < 17 : 
            zeros = np.zeros(17)
            zeros[:rhythm.shape[0]] = rhythm
            rhythm = zeros
        return pitch,rhythm

