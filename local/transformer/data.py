'''
This file contains the helper functions related to reading from files and processing the data.
It also contains the iterable dataset that will be used to build the data loader.
This is my own work. (fs2776)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kaldiio
import os

from random import choice

class MFCCDataset(torch.utils.data.IterableDataset):

    def __init__(self, data_dic, max_len, n_feature, label2num, n_class):
        super(MFCCDataset, self).__init__()
        self.data = data_dic
        self.max_len = max_len
        self.n_feature = n_feature
        self.label2num = label2num
        self.n_class = n_class

    def get_label(self, key):
        label = key.split("_")[-1]
        if label not in self.label2num:
            return 'other'
        return label

    # Return zero-padded X and one-hot encoded y during each iteration
    def __iter__(self): 
        for key, val in self.data.items():
            d1, d2 = val.shape[0], val.shape[1]
            if d2 != self.n_feature:
                continue
            cls = torch.ones(1, self.n_feature)
            if d1 < self.max_len:
                padding = torch.ones(self.max_len - d1, self.n_feature)
                yield (torch.cat((cls, torch.from_numpy(val), padding), 0), \
                F.one_hot(torch.tensor(self.label2num[self.get_label(key)]), self.n_class))
            else:
                yield (torch.cat((cls, torch.from_numpy(val)), 0), \
                F.one_hot(torch.tensor(self.label2num[self.get_label(key)]), self.n_class))


# Define all dictionaries and vocabularies
class_8 = {'down':0, 'go':1, 'left':2, 'no':3, 'right':4, 'stop':5, 'up':6, 'yes':7}
class_12 = {'yes':0, 'no':1, 'up':2, 'down':3, 'left':4, 'right':5, \
                        'on':6, 'off':7, 'stop':8, 'go': 9, 'noise': 10,'other':11}
vocab_v1 = ['stop', 'seven', 'yes', 'zero', 'no', 'up', 'two', \
            'four', 'go', 'one', 'six', 'on', 'right', 'nine', \
            'down', 'five', 'off', 'three', 'left', 'eight', 'house', \
            'dog', 'marvin', 'wow', 'happy', 'sheila', 'cat', 'tree', \
            'bird', 'bed']

vocab_v2 = ['five', 'zero', 'yes', 'seven', 'no', 'nine', 'down', \
            'one', 'go', 'two', 'stop', 'six', 'on', 'left', \
            'eight', 'right', 'off', 'four', 'three', 'up', 'dog', \
            'wow', 'house', 'marvin', 'bird', 'happy', 'cat', 'sheila', \
            'bed', 'tree', 'backward', 'visual', 'follow', 'learn', 'forward']

train = {}
validation = {}
test = {}
noise = {}

# Sampling audio clips from the noise data
def get_noise():
    sample_frames = 98
    noise_file = choice(list(noise.keys()))
    noise_frames = noise[noise_file].shape[0]
    onset = choice(list(range(noise_frames - sample_frames)))
    return noise[noise_file][onset:onset+sample_frames,:]
# Add noise data to train, validation or test set
def get_noise_data(ind, sample_size):
    counter = 0
    for i in range(sample_size):
        key = str(counter) + '_0_' + str(ind) + '_noise'
        if ind == 0:
            train[key] = get_noise()
        elif ind == 1:
            validation[key] = get_noise()
        else:
            test[key] = get_noise()
        counter += 1

def ref_from_filename(filename):
    if filename.find('train') != -1:
        return train
    if filename.find('validation') != -1:
        return validation
    if filename.find('test') != -1:
        return test
    print("Error:",filename)
    return None

# Load train, validation and test data
def load_data(version, data_dir, n_class):
    
    # Make dictionary
    if n_class == 8:
        label2num = class_8
    elif n_class == 12:
        label2num = class_12
    else:
        vocab = vocab_v1 if version == 1 else vocab_v2
        if n_class == 21:
            label2num = {key: value for value, key in enumerate(vocab[:20])}
            label2num['other'] = 20
        else:
            label2num = {key: value for value, key in enumerate(vocab)}
    
    # Read from file
    for dirpath, _, filenames in os.walk(data_dir): 
        for filename in filenames:
            if filename.endswith('.ark'):
                p = ref_from_filename(filename)
                d = kaldiio.load_ark(os.path.join(dirpath, filename))
                for key, numpy_array in d:
                    if n_class == 8:
                        if key.split("_")[-1] in label2num:
                            p[key] = numpy_array
                    elif 'noise' not in key and len(key.split("_")) == 4:
                        p[key] = numpy_array
                    if n_class == 12 and 'noise' in key:
                        noise[key] = numpy_array
    
    # Generate noise class. I chose sample size of 1/120 of the entire train, validation and test set.
    if n_class == 12:
        train_noise = len(train) // 120
        val_noise = len(validation) // 120
        test_noise = len(test) // 120
        get_noise_data(0, train_noise)
        get_noise_data(1, val_noise)
        get_noise_data(2, test_noise)

    return train, validation, test, label2num

# Load subset of test data. Default size is 32. No noise class.
def load_test(version, data_dir, n_class, size = 32):
    
    # Make dictionary
    if n_class == 8:
        label2num = class_8
    elif n_class == 12:
        label2num = class_12
    else:
        vocab = vocab_v1 if version == 1 else vocab_v2
        if n_class == 21:
            label2num = {key: value for value, key in enumerate(vocab[:20])}
            label2num['other'] = 20
        else:
            label2num = {key: value for value, key in enumerate(vocab)}
    
    # Read from file
    count = 0
    for dirpath, _, filenames in os.walk(data_dir): 
        for filename in filenames:
            if filename.endswith('.ark'):
                if filename.find('test') != -1:
                    d = kaldiio.load_ark(os.path.join(dirpath, filename))
                    for key, numpy_array in d:
                        if n_class == 8:
                            if key.split("_")[-1] in label2num:
                                test[key] = numpy_array
                        elif 'noise' not in key and len(key.split("_")) == 4:
                            test[key] = numpy_array
                        count += 1
                        if count >= size:
                            return test, label2num

    return test, label2num


