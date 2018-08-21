#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 09:57:43 2018

@author: phongdk
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
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import pandas as pd

from utils import normalize_data, get_same_length_data, get_new_max_length

win = 0.025
step = 0.01
#model_name = 'model/2018_08_17_17_20'

def extract_features(path):
    print 'extract feature of test set'
    test_pkl = 'test34.pkl'
    if os.path.isfile(test_pkl):
        [test_set, list_filenames] = cPickle.load(open(test_pkl, 'rb'))
        return test_set, list_filenames
    test_set = []
    list_filenames = sorted(os.listdir(path))
    for filename in list_filenames:
        path_to_file = os.path.join(path,filename)
        [rate,sig] = audioBasicIO.readAudioFile(path_to_file)
        if (rate == -1 and sig == -1):
            #convert to wav
            #command = "ffmpeg -i {}".format(path_to_file)
            extension = os.path.splitext(filename)[-1]
            new_file = path_to_file.replace(extension, '.wav')
            command = "ffmpeg -i {} {}".format(path_to_file, new_file)
            os.system(command)
            [rate,sig] = audioBasicIO.readAudioFile(new_file)
            os.system('rm {}'.format(path_to_file))     #remove old file not in *.wav format
        if sig.ndim >= 2:           #merge multichannels into mono channel
            sig = np.mean(sig,axis=1)
        features = audioFeatureExtraction.stFeatureExtraction(sig, rate, win*rate, step*rate);
        features = features.reshape((features.shape[1],-1))
        test_set.append(features)
    cPickle.dump([test_set, list_filenames], open(test_pkl, 'wb'), -1)
    return test_set, list_filenames

def predict_labels(test_set):
    test_set_arr = [element for element in test_set]
    test_set = np.array(test_set_arr)
    
#    # load json and create model
    #X_test, Y_test_gender, Y_test_accent = cPickle.load(open('validation.pkl','rb'))
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name + ".h5")
    print("Loaded model from disk")
    
    loaded_model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
#    acc = loaded_model.evaluate(X_test, [Y_test_gender, Y_test_accent], batch_size=128)
#    print('Test accuracy:', acc)
    labels = loaded_model.predict(test_set)
    #print('Test score:', score)
    
    gender = np.argmax(labels[0], axis=1)
    accent = np.argmax(labels[1], axis=1)
    print labels[1][0:5]
    #print gender
    #print accent
    return gender, accent   
 
            
if __name__ == "__main__":
    path_to_public_test = sys.argv[1]
    model_name = sys.argv[2]
    if ".zip" in path_to_public_test:
        os.system("unzip {}".format(path_to_public_test))
        path_to_public_test = path_to_public_test.replace(".zip", "")
    #convert_to_wav_test(path_to_public_test)
    train_pkl = 'train/train34.pkl'
    [training_set, maxlen, max_cepstrum, min_cepstrum] = cPickle.load(open(train_pkl, 'rb'))
    print maxlen
    maxlen = get_new_max_length(training_set)
    print maxlen
    
    test_set, list_filenames = extract_features(path_to_public_test)
#    print min_cepstrum
#    print max_cepstrum
#    print len(test_set)
#    print test_set[10]
    test_set = normalize_data(test_set, min_cepstrum=min_cepstrum, max_cepstrum=max_cepstrum)
    test_set = get_same_length_data(test_set, maxlen)
    for element in test_set:
        assert element.shape[0] == maxlen
#    print test_set[10].shape
#    print test_set[10]
    gender, accent = predict_labels(test_set)
    #create new df 
    df = pd.DataFrame({'id':list_filenames, 'gender':gender,'accent':accent}) 
    df.set_index('id', inplace=True)
    df.to_csv("submission.csv")
