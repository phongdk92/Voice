#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:59:48 2018

@author: phongdk
"""

import numpy as np

def normalize_data(dataset, min_cepstrum, max_cepstrum):
    diff = max_cepstrum - min_cepstrum
    if isinstance(dataset[0], dict):
        for data in dataset:
            data['feature'] = (data['feature'] - min_cepstrum) / diff    #(x-min)/(max-min)
    else:
        for (i, data) in enumerate(dataset):
            dataset[i] = (data - min_cepstrum) / diff    #(x-min)/(max-min)
    return dataset

def get_same_length_data(dataset, maxlen):
    if isinstance(dataset[0], dict):
        for data in dataset:
            if data['feature'].shape[0] < maxlen:           #padding data
                data['feature'] = np.pad(data['feature'], [(0,maxlen - len(data['feature'])),(0,0)], mode='constant', constant_values=0)
            else:
                data['feature'] = data['feature'][:maxlen]          #clip data
    else:
        for (i,data) in enumerate(dataset):
            if data.shape[0] < maxlen:           #padding data
                dataset[i] = np.pad(data, [(0,maxlen - len(data)),(0,0)], mode='constant', constant_values=0)
            else:
                dataset[i] = np.array(data[:maxlen,:])          #clip data
    return dataset

def get_new_max_length(dataset):
    l = [data['feature'].shape[0] for data in dataset]
    iqr = np.subtract(*np.percentile(l, [75, 25]))
    maxlen = int(np.percentile(l, 75) + 1.5 * iqr)       #set new max length for audio since the longest is over 200K
    return maxlen
