import pretty_midi as pretty_midi
import src.midi_functions as mf
import sys
import numpy as np
import torch
import pickle
import math
from sklearn.model_selection import train_test_split
import time
import os

now_time = str(int(round(time.time())))
class MidiSeparator:
    
    def __init__(self, 
                 songs_path,
                 save_path,
                 save_reconstructed,
                 input_dim,
                 input_length,
                 low_crop,
                 high_crop,
                 note_num = 17,
                 longest = 100 ): 
        self.songs_path = songs_path  
        self.save_path= save_path # separate/time
        self.save_reconstructed = save_reconstructed
        self.input_dim = input_dim
        self.input_length = input_length
        self.smallest_note = 16
        self.low_crop = low_crop
        self.high_crop = high_crop
        self.note_num = note_num
        self.longest = longest

    def import_midi_from_folder(self,model):
        folder = self.songs_path
        paths = []
        for paths, subdirs, files in os.walk(folder):   
            for name in files:   
                _path = paths.replace('\\', '/') + '/'  
                _name = name.replace('\\', '/')
                if _name.endswith('.mid') or _name.endswith('.midi'):
                    print("Importing song called " + _name)  
                    name = name.split('.')[-2] 
                    path = os.path.join(self.save_path,name) 
                    infor , source , length = self.load_rolls(_path, _name, path)
                    infor = np.array(infor)

                    pitch = infor[:,:34]
                    rhythm = infor[:,34:51]
                    pitch_pos = infor[:,51:] + 55

                    if pitch is not None :

                        if not os.path.isdir(path): 
                              os.mkdir(path)

                        file_z1 = os.path.join(path,"pitch")
                        if not os.path.isdir(file_z1): 
                              os.mkdir(file_z1)
                        file_z2 = os.path.join(path,"rhythm")
                        if not os.path.isdir(file_z2): 
                              os.mkdir(file_z2)
                        file_z3 = os.path.join(path,"pitch_pos")
                        if not os.path.isdir(file_z3): 
                              os.mkdir(file_z3)
                        
                        for i in range(pitch.shape[0]):
                            self.save_data(pitch[i], rhythm[i], pitch_pos[i], path, i ,name)
    
        return None

    def save_data(self, pitch, rhythm,pitch_pos, path, i, name):
        seq = str(i)
        file_p_name = "pitch/"+"p_"+seq+".txt"
        file_r_name = "rhythm/"+"r_"+seq+".txt"
        file_p_p_name = "pitch_pos/"+"p_"+seq+".txt"
        p_path = os.path.join(path,file_p_name)
        p_p_path = os.path.join(path,file_p_p_name)
        r_path = os.path.join(path,file_r_name)
        np.savetxt(p_path,pitch)
        np.savetxt(r_path,rhythm)
        np.savetxt(p_p_path,pitch_pos)

    def load_rolls(self, path, name, save_folder):
        infor_list = []
        source_list = []
        long_list = []
        try:
            mid = pretty_midi.PrettyMIDI(path + name)  
        except (ValueError, EOFError, IndexError, OSError, KeyError, ZeroDivisionError, AttributeError) as e:
            exception_str = 'Unexpected error in ' + name  + ':\n', e, sys.exc_info()[0]
            print(exception_str)
            return None, None

        tempo_change_times, tempo_change_bpm = mid.get_tempo_changes()
        song_start = 0
        song_end = mid.get_end_time()
        if len(tempo_change_times) == 1:
            tempo = tempo_change_bpm[0]
        else :
            print ("tempo changes")
            return None ,None
        quarter_note_length = 1. / (tempo/60.)   
        unit_time = quarter_note_length * 4. / self.smallest_note  
        if len(mid.instruments) ==1 :
            print("this song has no chords")
            return None ,None
        notes = mid.instruments[0].notes
        t = 0.
        roll = list()
        flag  = list()
        X = torch.tensor([])
        num1 = 1
        num2 = 1
        number = 0
        for note in notes:  
            
            elapsed_time = note.start - t  
            if elapsed_time < -0.03: 
                print("the %d notes overlap %f" % (number,-elapsed_time))
                return None ,None
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
            pitch,rhythm,pitch_pos = self.pitch_rhythm(part)
            infor1 = np.append(pitch,rhythm)
            infor = np.append(infor1,pitch_pos)

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

            pitch,rhythm,pitch_pos = self.pitch_rhythm(part)
            infor1 = np.append(pitch,rhythm)
            infor = np.append(infor1,pitch_pos)
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
        return infor_list , source_list ,long_list

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
        temp = np.zeros((17))
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
        lenth = len(note1)
        temp[:lenth] = note1
        return pitch,rhythm,temp
