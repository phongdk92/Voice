#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 16:12:30 2018

@author: phongdo
"""

'''test'''
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
import keras as K
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Activation, LSTM, Dropout, Input, Flatten, InputLayer 
from keras.preprocessing.sequence import pad_sequences

from main import convert_to_wav, normalize_data, padding_data
model_name = ''

def get_mfcc_feature(model, path):
    test_set = []
    list_filenames = sorted(os.listdir(path))
    for filename in list_filenames:
        path_to_file = os.path.join(path,filename)
        if not filename.endswith('wav'):
            #convert to wav
            command = "ffmpeg -i {}".format(path_to_file)
            #print commands.getoutput(command)
            extension = os.path.splitext(filename)[-1]
            new_file = path_to_file.replace(extension, '.wav')
            command = "ffmpeg -i {} {}".format(path_to_file, new_file)
            os.system(command)
            #command = "ffmpeg -i {}".format(new_file)
            #print commands.getoutput(command)
            
            (rate,sig) = wav.read(new_file)
            os.system('rm {}'.format(new_file))     #remove new *.wav file
        else:
            (rate,sig) = wav.read(path_to_file)
             
        mfcc_feat = mfcc(sig,rate, nfft = 1024)
        test_set.append(mfcc_feat)
    return test_set, list_filenames

def predict_labels(list_filenames, test_set):
    # load json and create model
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name + ".h5")
    print("Loaded model from disk")
    
    labels = loaded_model.predict(test_set)
    
    
if __name__ == "__main__":
    path_to_public_test = sys.argv[1]
    if ".zip" in path_to_public_test:
        os.command("unzip {}".format(path_to_public_test))
        path_to_public_test.replace(".zip", " ")
    convert_to_wav(path_to_public_test)
    #maxlen, max_cepstrum, min_cepstrum must come from training_set
    test_set, maxlen, max_cepstrum, min_cepstrum = get_mfcc_feature()
    normalize_data(test_set, min_cepstrum=min_cepstrum, max_cepstrum=max_cepstrum)
    padding_data(test_set)
    predict_labels(test_set)