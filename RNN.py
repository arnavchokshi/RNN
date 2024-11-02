from tkinter import N
from types import NoneType
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
import random
import pandas as pd
import pretty_midi
import glob
import collections
import pathlib
import datetime
import fluidsynth
import pathlib
import seaborn as sns


from IPython import display
from matplotlib import pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple


filenames = glob.glob('/Users/arnavchokshi/Desktop/data2/bach/mond_3.mid')


sample_file = filenames[0]
pm = pretty_midi.PrettyMIDI(sample_file)


instrument = pm.instruments[0]
instrument_name = pretty_midi.program_to_instrument_name(instrument.program)


def midi_to_notes(midi_file: str) -> pd.DataFrame:
 pm = pretty_midi.PrettyMIDI(midi_file)
 instrument = pm.instruments[0]
 notes = collections.defaultdict(list)


 # Sort the notes by start time
 sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
 prev_start = sorted_notes[0].start


 for note in sorted_notes:
   start = note.start
   end = note.end
   notes['pitch'].append(note.pitch)
   notes['start'].append(start)
   notes['end'].append(end)
   notes['step'].append(start - prev_start)
   notes['duration'].append(end - start)
   prev_start = start


 return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


raw_notes = midi_to_notes(sample_file)


def notes_to_midi(
 notes: pd.DataFrame,
 out_file: str,
 instrument_name: str,
 velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:


 pm = pretty_midi.PrettyMIDI()
 instrument = pretty_midi.Instrument(
     program=pretty_midi.instrument_name_to_program(
         instrument_name))


 prev_start = 0
 for i, note in notes.iterrows():
   start = float(prev_start + note['step'])
   end = float(start + note['duration'])
   note = pretty_midi.Note(
       velocity=velocity,
       pitch=int(note['pitch']),
       start=start,
       end=end,
   )
   instrument.notes.append(note)
   prev_start = start


 pm.instruments.append(instrument)
 pm.write(out_file)
 return pm




x = 0
x_train = []
y_train = []
x_train = np.array(x_train)
y_train = np.array(y_train)


comp_size = 24
raw_notes = raw_notes[:comp_size]
note_size = 24


output_file = notes_to_midi(
   raw_notes, out_file='out.midi', instrument_name=instrument_name)


notes_raw_notes = raw_notes['pitch']
notes_raw_notes = np.append(notes_raw_notes, notes_raw_notes)
notes_raw_notes = np.append(notes_raw_notes, notes_raw_notes)
notes_raw_notes = np.append(notes_raw_notes, notes_raw_notes)
notes_raw_notes = np.append(notes_raw_notes, notes_raw_notes)
notes_raw_notes = np.append(notes_raw_notes, notes_raw_notes)
notes_raw_notes = np.append(notes_raw_notes, notes_raw_notes)


while x + note_size + 1 < notes_raw_notes.shape[0]:
   x_train = np.append(x_train, notes_raw_notes[x:x+note_size])
   y_train = np.append(y_train, notes_raw_notes[x+note_size:x+note_size+1])
   x = x + 1


x_train = x_train.reshape(y_train.shape[0], note_size, 1)
nodes = 1
dense_size = 1


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience = 1000, restore_best_weights = True) #uses best weights


data_loss = []


time = 0
while time != 101:
   nodes = 1
   while nodes != 25:
       model = Sequential()
       model.add(SimpleRNN(nodes, activation = 'relu'))
       model.add(Dense(dense_size, activation = None))


       model.compile(
           loss='MeanAbsoluteError',
           optimizer='Adam',
           metrics=['Accuracy'])


       history = model.fit(x_train, y_train, epochs=500, callbacks=[callback])
       data_loss = np.append(data_loss, model.evaluate(x_train, y_train)[0])

       nodes = nodes + 1
   time = time + 1



