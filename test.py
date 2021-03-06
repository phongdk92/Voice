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
import pandas as pd

from utils import get_new_max_length, get_same_length_data, normalize_data
model_name = 'model/2018_08_16_09_14'

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
    #print gender
    #print accent
    return gender, accent    
    
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
    train_pkl = 'train_wav/train.pkl'
    [training_set, maxlen, max_cepstrum, min_cepstrum] = cPickle.load(open(train_pkl, 'rb'))
    maxlen = get_new_max_length(training_set)
    print maxlen
    
    test_set, list_filenames = get_mfcc_feature(path_to_public_test)
    print len(test_set)
    test_set = normalize_data(test_set, min_cepstrum=min_cepstrum, max_cepstrum=max_cepstrum)
    test_set = get_same_length_data(test_set, maxlen)
    gender, accent = predict_labels(test_set)
    #create new df 
    df = pd.DataFrame({'id':list_filenames, 'gender':gender,'accent':accent}) 
    df.set_index('id', inplace=True)
    df.to_csv("submission.csv")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    