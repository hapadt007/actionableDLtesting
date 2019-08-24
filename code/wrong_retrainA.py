#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:07:35 2019

@author: qq
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#os.environ["TF_CPP_MIN_LOG_LEVEL"]=2

import keras
from keras.datasets import mnist
from keras.datasets import cifar10
from keras import optimizers
import random
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
import pandas as pd
import csv
import boundary
from keras.utils import np_utils
from keras.models import Model,Input,load_model

def wrong_random_select(notpasslist,amount,x_test,y_test):
    x_bound = np.zeros((amount,28,28))
    y_bound = np.zeros((amount,))
    
    #lst = range(len(x_test))
    rnd_lst = random.sample(notpasslist,amount)
    for i in range(amount):
        x_bound[i] = x_test[rnd_lst[i]]
        y_bound[i] = y_test[rnd_lst[i]]
    return x_bound,y_bound


#返回不通过的测试用例list
def find_notpass(model,image,test_label):
    image = image.astype('float32').reshape(-1,28,28,1)
    image = image / 255
    pred=model.predict(image)
    pred=list(map(lambda x:np.argmax(x),pred))
    notpasslist=[]
    for i in range(len(image)):
        if pred[i]!=test_label[i]:
            notpasslist.append(i)
    #print 'notpass:',len(notpasslist)
    return notpasslist
#最大/第二大，比值越大，说明离边界越远。越接近越好
    

def save_ratio_order(filename):
    npzfile=np.load('./mnist.npz')  
    x_train= npzfile['x_train']
    y_train= npzfile['y_train']  
    
    
    npzfile=np.load('../backup/'+filename+'.npz') 
    y_test= npzfile['y_test']
    x_test= npzfile['x_test']   
    
    model_path='../ModelA_raw.hdf5'
    model=load_model(model_path)
    bound_data_lst =[]
    out_index=len(model.layers)-1
    model_layer=Model(inputs=model.input,outputs=model.layers[out_index].output)
    x_test=x_test.astype('float32').reshape(-1,28,28,1)
    x_test/=255
    act_layers=model_layer.predict_on_batch(x_test)
    order_lst,order_ratio_lst=boundary.order_output(act_layers)
    dataframe = pd.DataFrame({'index':order_lst,'ratio':order_ratio_lst})
    dataframe.to_csv('ratio_order_A.csv') 
    return

'''
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
model_path='../ModelA_raw.hdf5'
model=load_model(model_path)
bound_data_lst =[]
out_index=len(model.layers)-1
model_layer=Model(inputs=model.input,outputs=model.layers[out_index].output)
x_test=x_test.astype('float32').reshape(-1,28,28,1)
x_test/=255
act_layers=model_layer.predict_on_batch(x_test)
order_lst,order_ratio_lst=order_output(act_layers)
dataframe = pd.DataFrame({'index':order_lst,'ratio':order_ratio_lst})
dataframe.to_csv('ratio_order_A.csv') 
''' 
def retrain(filename,wrong=True,samplesize=100):
    #(x_train, y_train), (__, __) = mnist.load_data() 
    npzfile=np.load('./mnist.npz')  
    x_train= npzfile['x_train']
    y_train= npzfile['y_train']  
    
    npzfile=np.load('../backup/'+filename+'.npz') 
    y_test= npzfile['y_test']
    x_test= npzfile['x_test']
    
    model_path='../ModelA_raw.hdf5'
    model=load_model(model_path)
    notpass_num=0
    #bound_ratio=10
 
    if wrong==False:
        path='./ratio_order_A.csv'
        temp = []
        with open(path) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader: 
                temp.append(row)
                
                
        #csv_data = pd.read_csv(path,usecols=[1])
        #temp = csv_data.values
        bound_lst=[]
        #for i in range(samplesize):
            #bound_lst.append(temp[i][0])
        for i in range(samplesize):
            bound_lst.append(int(temp[i+1][1]))
        
        notpasslist = find_notpass(model,x_test,y_test)
        
        for i in range(samplesize):
            if bound_lst[i] in notpasslist:
                notpass_num+=1
        print("wrong case ratio:")
        print(1.0*notpass_num/samplesize)
        print("\n")
        
        
        x_bound = np.zeros((samplesize,28,28))
        y_bound = np.zeros((samplesize,))

        for i in range(samplesize):
            x_bound[i] = x_test[bound_lst[i]]
            y_bound[i] = y_test[bound_lst[i]]

        x_train = np.concatenate((x_bound, x_train), axis=0)
        y_train = np.concatenate((y_bound, y_train), axis=0)
        #x_train = x_bound
        #y_train = y_bound
    else:
        notpasslist = find_notpass(model,x_test,y_test)
        x_random,y_random = wrong_random_select(notpasslist,samplesize,x_test,y_test)

        x_train = np.concatenate((x_random, x_train), axis=0)
        y_train = np.concatenate((y_random, y_train), axis=0)  
        #x_train = x_random
        #y_train = y_random

    
    #x_train = x_bound
    #y_train = y_bound
    
    x_train = x_train.astype('float32').reshape(-1,28,28,1)
    x_test = x_test.astype('float32').reshape(-1,28,28,1)
    
    x_train = x_train / 255
    x_test = x_test / 255
    
    
    y_test = np_utils.to_categorical(y_test, 10)
    y_train = np_utils.to_categorical(y_train, 10)
                
            
    score = model.evaluate(x_test, y_test,verbose=0)
    #print('Test Loss: %.4f' % score[0])
    print('Test accuracy: %.4f'% score[1])
    origin_acc=score[1]
    if wrong==False:
        return score[1],origin_acc,notpass_num
    '''
    npzfile=np.load('./backup/beta_mnist_shift_up2.npz') 
    y_beta= npzfile['y_test']
    x_beta= npzfile['x_test'] 
    '''
    
    sgd = optimizers.SGD(lr=0.005)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    model.fit(x_train, y_train, batch_size=100, epochs=5, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test)
    print('Test accuracy: %.4f'% score[1])
    return score[1],origin_acc,notpass_num
#####################################################
#print("random select............................")

lstorder_acc=[]
lstramdon_acc=[]
origin_acc=0
length_of_order=1000
length_of_wrong=0
notpass_num=0
datafilename_lst=['beta_mnist_rotate(down2)',
                  'beta_mnist_rotate(up2)',
                  'beta_mnist_shift_down2',
                  'beta_mnist_shift_down3',
                  'beta_mnist_shift_left3',
                  'beta_mnist_shift_right3',
                  'beta_mnist_shift_rotate(left2)',
                  'beta_mnist_shift_up3']
for f in range(len(datafilename_lst)):
    filename=datafilename_lst[f]
    print(filename)
    print("\n")
    save_ratio_order(filename)
    lstorder_acc=[]
    lstramdon_acc=[]
    __,origin_acc,notpass_num = retrain(filename,wrong=False,samplesize=length_of_order)
    length_of_wrong=int(length_of_order-length_of_order*origin_acc)
    for i in range(5):
        #__,origin_acc,notpass_num = retrain(filename,wrong=False,samplesize=length_of_order)
        #lstorder_acc.append(testbound_acc)
        print("random select............................")
        
        testramdon_acc, __,__ = retrain(filename,wrong =True,samplesize=length_of_wrong)
        lstramdon_acc.append(testramdon_acc)
        
        
    resultfilename = '../wrong_resultA.txt'
    with open(resultfilename,'a') as file_object:
        file_object.write('ModelA\n')
        file_object.write('Data:')
        file_object.write(filename)
        file_object.write('\n')
        file_object.write('origin_accuracy: ')
        file_object.write(str(origin_acc))
        file_object.write('\n')
        file_object.write('length_of_order: ')
        file_object.write(str(length_of_order))
        file_object.write('\n')
        file_object.write('length_of_wrong: ')
        file_object.write(str(length_of_wrong))
        file_object.write('\n')
        file_object.write('order_accuracy:\n')
        file_object.write(str(lstorder_acc))
        file_object.write('\n')
        file_object.write('wrong_accuracy:\n')
        file_object.write(str(lstramdon_acc))
        file_object.write('\n')
    
    