#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:07:35 2019

@author: qq
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import keras
from keras.datasets import mnist
from keras import optimizers
import random
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
import matplotlib.image as mp
#from PIL import Image
#from skimage import io
import boundary


def random_select(amount,x_test,y_test):
    x_bound = np.zeros((amount,28,28))
    y_bound = np.zeros((amount,))
    
    lst = range(len(x_test))
    rnd_lst = random.sample(lst,amount)
    for i in range(amount):
        x_bound[i] = x_test[rnd_lst[i]]
        y_bound[i] = y_test[rnd_lst[i]]
    return x_bound,y_bound


def train(filename='beta_mnist_shift_rotate(left2)',bound_ratio=2,ramdon=True,samplesize=100):
#(x_train, y_train), (__, __) = mnist.load_data()
    npzfile=np.load('../mnist.npz')  
    x_train= npzfile['x_train']
    y_train= npzfile['y_train']  
    #(x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    npzfile=np.load('../backup/'+filename+'.npz') 
    y_test= npzfile['y_test']
    x_test= npzfile['x_test']
    
    
    model_path='../ModelB_raw.hdf5'
    model=load_model(model_path)
    len_bound=0
    #bound_ratio=2
    if ramdon==False:
        x_bound,y_bound = boundary.get_bound_data_mnist(model,x_test,y_test,bound_ratio)
        #x_train = x_bound
        #y_train = y_bound
        x_train = np.concatenate((x_bound, x_train), axis=0)
        y_train = np.concatenate((y_bound, y_train), axis=0)
        len_bound =len(x_bound)
    else:
        x_random,y_random = random_select(samplesize,x_test,y_test)
        #x_train = x_random
        #y_train = y_random
        x_train = np.concatenate((x_random, x_train), axis=0)
        y_train = np.concatenate((y_random, y_train), axis=0)
       
    #x_train = x_bound
    #y_train = y_bound
    
    x_train = x_train.astype('float32').reshape(-1,28,28,1)
    x_test = x_test.astype('float32').reshape(-1,28,28,1)
    
    x_train = x_train / 255
    x_test = x_test / 255
    
    
    y_test = keras.utils.to_categorical(y_test, 10)
    y_train = keras.utils.to_categorical(y_train, 10)
        
    
    score = model.evaluate(x_test, y_test)
    #print('Test Loss: %.4f' % score[0])
    print('Test accuracy: %.4f'% score[1])
    origin_acc=score[1]
    
    '''
    npzfile=np.load('./backup/beta_mnist_shift_up2.npz') 
    y_beta= npzfile['y_test']
    x_beta= npzfile['x_test'] 
    '''
    
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    model.fit(x_train, y_train, batch_size=100, epochs=5, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test)
    print('Test accuracy: %.4f'% score[1])
    
    return score[1],len_bound,origin_acc
#####################################################

lstbound_acc=[]
lstramdon_acc=[]
len_bound=0
origin_acc=0
filename='beta_mnist_rotate(up2)'
for i in range(5):
    testbound_acc, len_bound,origin_acc = train(filename=filename,bound_ratio=2,ramdon=False)
    lstbound_acc.append(testbound_acc)
    print("random select............................")
    testramdon_acc, __,__ = train(filename=filename,bound_ratio=2,ramdon =True,samplesize=len_bound)
    lstramdon_acc.append(testramdon_acc)
    
    
resultfilename = '../result.txt'
with open(resultfilename,'w') as file_object:
    file_object.write('ModelB\n')
    file_object.write(filename)
    file_object.write('\n')
    file_object.write('origin_accuracy: ')
    file_object.write(str(origin_acc))
    file_object.write('\n')
    file_object.write('length_of_bound: ')
    file_object.write(str(len_bound))
    file_object.write('\n')
    file_object.write('bound_accuracy:\n')
    file_object.write(str(lstbound_acc))
    file_object.write('\n')
    file_object.write('ramdon_accuracy:\n')
    file_object.write(str(lstramdon_acc))
    file_object.write('\n')