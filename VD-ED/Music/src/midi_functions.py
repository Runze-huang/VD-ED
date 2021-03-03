import numpy as np
import _pickle as pickle
import os
import sys
import pretty_midi as pm
import mido
import operator
import torch

def numpy_to_midi(sample_roll, tempo, save_folder, filename, smallest_note=16):
    if not os.path.exists(save_folder): 
        os.makedirs(save_folder)
    music = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program('Acoustic Grand Piano')
    piano = pm.Instrument(program=piano_program)

    quarter_note_length = 1. / (tempo/60.)   
    unit_time = quarter_note_length * 4. / smallest_note
    t = 0
    for i in sample_roll:
        if 'torch' in str(type(i)):
            pitch = int(i.max(0)[1])
        else:
            pitch = int(np.argmax(i))
        if pitch < 128:  
            note = pm.Note(
                velocity=100, pitch=pitch, start=t, end=t + unit_time)
            t += unit_time
            piano.notes.append(note)
        elif pitch == 128:  
            if len(piano.notes) > 0:
                note = piano.notes.pop()
            else:
                p = np.random.randint(60, 72)
                note = pm.Note(
                    velocity=100, pitch=int(p), start=0, end=t)
            note = pm.Note(
                velocity=100,
                pitch=note.pitch,
                start=note.start,
                end=note.end + unit_time)
            piano.notes.append(note)
            t += unit_time
        elif pitch == 129: 
            t += unit_time
    music.instruments.append(piano)
    music.write(os.path.join(save_folder,filename))
def modify_pianoroll_dimentions(X,low,high,operate):
    if operate =="del":
        cut = range(high,128)           
        X= np.delete(X,cut,axis = 1)
        cut =range(0,low)         
        X= np.delete(X,cut,axis = 1)
        return X

    elif operate =="add":
        X = np.pad(X, ((0,0),(low, 0)), 'constant', constant_values=(0, 0)) 
        X = np.pad(X, ((0,0),(0, 128-high)), 'constant', constant_values=(0, 0))
        X[:,[high,high+1,128,129]]= X[:,[128,129,high,high+1]] 
        return X
    else:
        print("operate error")

def _sampling(x): 
    idx = x.max(1)[1] 
    x = torch.zeros_like(x)
    arange = torch.arange(x.size(0)).long()
    if torch.cuda.is_available():
        arange = arange.cuda()
    x[arange, idx] = 1
    return x

