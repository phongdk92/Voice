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
import cPickle

from main import normalize_data, get_same_length_data, get_new_max_length
model_name = ''

def get_mfcc_feature(path):
    print 'extract feature of test set'
    test_pkl = 'test.pkl'
    if os.path.isfile(test_pkl):
        [test_set, list_filenames] = cPickle.load(open(test_pkl, 'rb'))
        return test_set, list_filenames
    test_set = []
    list_filenames = sorted(os.listdir(path))
    for filename in list_filenames:
        path_to_file = os.path.join(path,filename)
        if not filename.endswith('wav'):
            #convert to wav
            command = "ffmpeg -i {}".format(path_to_file)
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
             
        mfcc_feat = mfcc(sig,rate, nfft = 2048)
        test_set.append(mfcc_feat)
    cPickle.dump([test_set, list_filenames], open(test_pkl, 'wb'), -1)
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
    
def convert_to_wav_test(folder):
    for (i,filename) in enumerate(os.listdir(folder)):
        print i
        path_to_file = os.path.join(folder,filename)
        if not filename.endswith('wav'):
            #convert to wav
            command = "ffmpeg -i {}".format(path_to_file)
            extension = os.path.splitext(filename)[-1]
            new_file = path_to_file.replace(extension, '.wav')
            command = "ffmpeg -i {} {}".format(path_to_file, new_file)
            os.system(command)
            os.system('rm {}'.format(path_to_file))     #remove old file not in *.wav format    
            
if __name__ == "__main__":
    path_to_public_test = sys.argv[1]
    if ".zip" in path_to_public_test:
        #os.system("unzip {}".format(path_to_public_test))
        path_to_public_test = path_to_public_test.replace(".zip", "")
    #convert_to_wav_test(path_to_public_test)
    train_pkl = 'train/train.pkl'
    [training_set, maxlen, max_cepstrum, min_cepstrum] = cPickle.load(open(train_pkl, 'rb'))
    maxlen = get_new_max_length(training_set)
    print maxlen
    
    test_set, list_filenames = get_mfcc_feature(path_to_public_test)
    print len(test_set)
    test_set = normalize_data(test_set, min_cepstrum=min_cepstrum, max_cepstrum=max_cepstrum)
    test_set = get_same_length_data(test_set, maxlen)
    
#    predict_labels(test_set)