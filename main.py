#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 09:12:13 2018

@author: phongdo
"""

import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from matplotlib import cm
import subprocess
import commands
import os
import sys
import cPickle
import keras as K
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, LSTM, Dropout, Input, Flatten, InputLayer 
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from datetime import datetime

path = 'train'
model_name= 'model_' + datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M")
#print model_name
#sys.exit()

def convert_to_wav(folder):
    for folder in sorted(os.listdir(folder)):
        if "pkl" in folder:
            continue
        print folder
        curDir = os.path.join(os.getcwd(), path, folder)
        for (i,filename) in enumerate(os.listdir(curDir)):
            print i
            path_to_file = os.path.join(curDir,filename)
            if not filename.endswith('wav'):
                #convert to wav
                command = "ffmpeg -i {}".format(path_to_file)
                extension = os.path.splitext(filename)[-1]
                new_file = path_to_file.replace(extension, '.wav')
                command = "ffmpeg -i {} {}".format(path_to_file, new_file)
                os.system(command)
                #print commands.getoutput(command)
                os.system('rm {}'.format(path_to_file))     #remove old file not in *.wav format
    
def get_mfcc_feature(path_to_folder):
    train_pkl = os.path.join(path_to_folder,'0_female_central.pkl')
    if os.path.isfile(train_pkl):
        [training_set, maxlen, max_cepstrum, min_cepstrum] = cPickle.load(open(train_pkl, 'rb'))
        return training_set, maxlen, max_cepstrum, min_cepstrum
    
    training_set = []
    max_cepstrum = np.zeros(13)
    min_cepstrum = np.zeros(13)
    maxlen = 0
    list_folders = sorted(os.listdir(path_to_folder))
    list_folders = [folder for folder in list_folders if "pkl" not in folder and 'zip' not in folder]
    #print list_folders
    #sys.exit()
    for (order, folder) in enumerate(list_folders):
        #if "pkl" in folder or 'zip' in folder:
        #    continue
        if order <= 1:
            continue
#        print folder
        folder_pkl = os.path.join(path, str(order) + '_' + folder + '.pkl')
        print folder_pkl, 
        #continue
        gender = 0 if 'female' in folder else 1
        if 'north' in folder:
            accent = 0
        elif 'central' in folder:
            accent = 1
        else:
            accent = 2
        print accent, gender
        curDir = os.path.join(os.getcwd(), path, folder)
        for (i,filename) in enumerate(os.listdir(curDir)):
            print i
            path_to_file = os.path.join(curDir,filename)
            #print path_to_file
#            if not filename.endswith('wav'):
#                #convert to wav
#                command = "ffmpeg -i {}".format(path_to_file)
#                #print commands.getoutput(command)
#                extension = os.path.splitext(filename)[-1]
#                new_file = path_to_file.replace(extension, '.wav')
#                command = "ffmpeg -i {} {}".format(path_to_file, new_file)
#                os.system(command)
#                #command = "ffmpeg -i {}".format(new_file)
#                #print commands.getoutput(command)
#                
#                (rate,sig) = wav.read(new_file)
#                os.system('rm {}'.format(new_file))     #remove new *.wav file
#            else:
            (rate,sig) = wav.read(path_to_file)
                #print sig
                #plt.plot(mfcc_feat)
                #fig, ax = plt.subplots()
                #mfcc_data= np.swapaxes(mfcc_feat, 0 ,1)
                #cax = plt.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
                #plt.show()                
                #sys.exit()
            mfcc_feat = mfcc(sig,rate, nfft = 1024)
            maxlen = max(maxlen, mfcc_feat.shape[0])        #find max time-length
            
            max_current_mfcc = np.max(mfcc_feat, axis = 0)
            max_cepstrum = np.maximum(max_cepstrum, max_current_mfcc)
            
            min_current_mfcc = np.min(mfcc_feat, axis = 0)
            min_cepstrum = np.minimum(min_cepstrum, min_current_mfcc)
            #print sig.shape
            #print mfcc_feat.shape
            sample = {'feature': mfcc_feat,
                      'gender': np.zeros(2, np.int16),
                      'accent': np.zeros(3, np.int16)}
            sample['gender'][gender] = 1
            sample['accent'][accent] = 1
            training_set.append(sample)
            #break
        print len(training_set)
        cPickle.dump([training_set, maxlen, max_cepstrum, min_cepstrum], open(folder_pkl,'wb'), -1)
    #del training_set            
        #break
    #cPickle.dump([training_set, maxlen, max_cepstrum, min_cepstrum], open(train_pkl,'wb'), -1)
    #print len(training_set)
    return training_set, maxlen, max_cepstrum, min_cepstrum       
        
def normalize_data(dataset, min_cepstrum, max_cepstrum):
    diff = max_cepstrum - min_cepstrum
    for data in dataset:
        data['feature'] = (data['feature'] - min_cepstrum) / diff    #(x-min)/(max-min)
    return dataset

def padding_data(dataset, maxlen):
    for data in dataset:
        data['feature'] = np.pad(data['feature'], [(0,maxlen - len(data['feature'])),(0,0)], mode='constant', constant_values=0)
    
def split_data(training_set):
    train, test = train_test_split(training_set, test_size = 0.3, random_state=42)
    X_train = np.array([element['feature'] for element in train])
    Y_train_gender = np.array([element['gender'] for element in train])
    Y_train_accent = np.array([element['accent'] for element in train])
    assert (X_train.shape[0] == Y_train_gender.shape[0]) and (X_train.shape[0] == Y_train_accent.shape[0])
    
    X_test = np.array([element['feature'] for element in test])
    Y_test_gender = np.array([element['gender'] for element in test])
    Y_test_accent = np.array([element['accent'] for element in test])
    assert (X_test.shape[0] == Y_test_gender.shape[0]) and (X_test.shape[0] == Y_test_accent.shape[0])
    
    return X_train, Y_train_gender, Y_train_accent, X_test, Y_test_gender, Y_test_accent

def model(training_set, maxlen, use_dropout = True):
    X_train, Y_train_gender, Y_train_accent, X_test, Y_test_gender, Y_test_accent =  split_data(training_set)
    
    inputs = Input(shape=(maxlen,13,))
#    lstm1 = LSTM(16, return_sequences=True)(inputs)
#    lstm1_flatten = Flatten()(lstm1)
#    dense1 = Dense(32, activation='relu')(lstm1_flatten)
    lstm1 = LSTM(16)(inputs)
    dense1 = Dense(8, activation='relu')(lstm1)
    output_gender = Dense(2, activation='softmax')(dense1)
    output_accent = Dense(3, activation='softmax')(dense1)
    
    model = Model(input=inputs, output=[output_gender, output_accent])
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    
    '''model'''
#    model = Sequential()
#    model.add(InputLayer(input_shape=(maxlen,13,)))
#    model.add(LSTM(64, return_sequences=True))
#    model.add(Flatten())
#    model.add(Dense(32))
#    model.add(Dense(2,activation='softmax'))
#    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    
    
    #model.fit(X_train, Y_train_gender, batch_size=10, verbose=1, epochs=3)
    model.fit(X_train, [Y_train_gender, Y_train_accent], validation_data=(X_test, [Y_test_gender, Y_test_accent]), \
              batch_size=128, verbose=1, epochs=1)

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name + ".h5")
    print("Saved model to disk")
    
if __name__ == "__main__":
    #convert_to_wav(path)
    training_set, maxlen, max_cepstrum, min_cepstrum = get_mfcc_feature(path)
    print maxlen
    print max_cepstrum
    print min_cepstrum
    training_set = normalize_data(training_set, min_cepstrum=min_cepstrum, max_cepstrum=max_cepstrum)
    padding_data(training_set, maxlen=maxlen)
    model(training_set, maxlen)
    #print training_set[0:2]['gender']
    